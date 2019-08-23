from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dropout import create_adversarial_dropout_mask, calculate_jacobian
from losses import EntropyLoss
from metrics import AverageMeterSet
from ramps import linear_rampup
from utils import disable_tracking_bn_stats
from vat import VirtualAdversarialPerturbationGenerator


class JointCnnFcTrainerSplit(object):
    def __init__(self, models, dataloaders, optimizers, criterions, num_epochs, args,
                 log_period_as_iter=40000, train_loggers=None, val_loggers=None, device='cuda',
                 lr_schedulers=None, pretrain_epochs=0):
        self.feature_extractor = models['feature_extractor'].to(device)
        self.feature_extractor = nn.DataParallel(self.feature_extractor) if args.num_gpu > 1 else self.feature_extractor

        self.classifier = models['classifier'].to(device)
        self.fc_drop_size = self.classifier.drop_size
        self.classifier = nn.DataParallel(self.classifier) if args.num_gpu > 1 else self.classifier

        self.optimizers = optimizers

        self.criterions = criterions
        self.class_criterion = self.criterions['classifier']
        self.source_consistency_criterion = self.criterions['source_consistency']
        self.target_consistency_criterion = self.criterions['target_consistency']
        self.entmin_criterion = self.criterions['entmin']
        self.cls_balance_criterion = self.criterions['class_balance']

        self.entropy_loss = EntropyLoss()

        self.target_fc_consistency_weight = args.target_fc_consistency_weight
        self.target_cnn_consistency_weight = args.target_cnn_consistency_weight
        self.source_fc_consistency_weight = args.source_fc_consistency_weight
        self.source_cnn_consistency_weight = args.source_cnn_consistency_weight
        self.entmin_weight = args.entmin_weight
        self.drop_prob = args.drop_prob
        self.cnn_delta = args.cnn_delta
        self.fc_delta = args.fc_delta
        self.source_delta = args.source_delta
        self.source_delta_fc = args.source_delta_fc

        self.rampup_length = args.rampup_length
        self.source_rampup_length = args.source_rampup_length

        self.use_vat = args.use_vat
        self.source_vat_loss_weight = args.source_vat_loss_weight
        self.target_vat_loss_weight = args.target_vat_loss_weight
        self.vat = VirtualAdversarialPerturbationGenerator(self.feature_extractor, self.classifier, xi=args.xi,
                                                           eps=args.eps, ip=args.ip)

        self.args = args

        self.dataloaders = dataloaders
        self.lr_schedulers = lr_schedulers if lr_schedulers else {}
        self.num_epochs = num_epochs
        self.device = device

        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.log_period_as_iter = log_period_as_iter
        self.validation_period_as_iter = args.validation_period_as_iter
        self.pretrain_epochs = pretrain_epochs

    def train(self):
        accum_iter = 0
        self.validate(0, self.dataloaders['val'], self.dataloaders['source_val'], accum_iter)
        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    _, accum_iter = self.train_one_epoch(epoch, self.dataloaders[phase], accum_iter)
                else:
                    self.validate(epoch, self.dataloaders['val'], self.dataloaders['source_val'], accum_iter)

        self._complete_logging({
            'state_dict': (self._create_state_dict())
        })

    def _complete_logging(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    @staticmethod
    def _call_loggers(loggers, log_data):
        for logger in loggers:
            logger.log(**log_data)

    def update_grad(self, keys=None):
        keys = keys if keys else self.optimizers.keys()
        for key in keys:
            self.optimizers[key].step()

    def reset_grad(self, keys=None):
        keys = keys if keys else self.optimizers.keys()
        for key in keys:
            self.optimizers[key].zero_grad()

    # TODO: Refactor this
    def train_one_epoch(self, epoch, dataloader, accum_iter):
        self.feature_extractor.train()
        self.classifier.train()

        for scheduler in self.lr_schedulers.values():
            scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(dataloader, ncols=150)

        for batch_idx, ((source_inputs, source_labels), (target_inputs, target_labels)) in enumerate(tqdm_dataloader):
            batch_size = source_inputs.size(0)
            source_inputs, source_labels = source_inputs.to(self.device), source_labels.to(self.device)
            target_inputs, target_labels = target_inputs.to(self.device), target_labels.to(self.device)

            self.reset_grad()

            feature_extractor_grads = OrderedDict()
            classifier_grads = OrderedDict()

            # VAT
            if self.use_vat:
                vat_adv, clean_vat_logits = self.vat(target_inputs)
                vat_adv_inputs = target_inputs + vat_adv

                with disable_tracking_bn_stats(self.feature_extractor):
                    with disable_tracking_bn_stats(self.classifier):
                        adv_vat_features, _ = self.feature_extractor(vat_adv_inputs)
                        adv_vat_logits = self.classifier(adv_vat_features)

                target_vat_loss = self.target_vat_loss_weight * self.target_consistency_criterion(adv_vat_logits,
                                                                                                  clean_vat_logits)
                target_vat_loss.backward()
                # TODO: should be compatible with single gpu. remove .module
                feature_extractor_grads = self.feature_extractor.module.stash_grad(feature_extractor_grads)
                classifier_grads = self.classifier.module.stash_grad(feature_extractor_grads)
            else:
                target_vat_loss = torch.tensor(0).to(self.device)

            # Target Consistency Loss
            # target_features1 = clean, target_features2 = adv drop
            target_features1, target_features2 = self.feature_extractor(target_inputs)
            target_logits1 = self.classifier(target_features1)

            # Target AdD
            target_elementwise_jacobian1, target_elementwise_jacobian2, clean_target_logits = calculate_jacobian(
                target_features2.detach(), target_logits1.detach(), self.fc_drop_size, self.classifier,
                self.target_consistency_criterion, self.reset_grad)

            target_channelwise_jacobian = F.avg_pool2d(target_elementwise_jacobian1,
                                                       kernel_size=target_elementwise_jacobian1.size()[2:])

            cnn_drop_delta = linear_rampup(epoch, self.rampup_length) * self.cnn_delta
            target_dropout_mask_cnn, _ = create_adversarial_dropout_mask(
                torch.ones_like(target_channelwise_jacobian),
                target_channelwise_jacobian, cnn_drop_delta)

            fc_drop_delta = linear_rampup(epoch, self.rampup_length) * self.fc_delta
            target_dropout_mask_fc, _ = create_adversarial_dropout_mask(
                torch.ones_like(target_elementwise_jacobian2),
                target_elementwise_jacobian2, fc_drop_delta)

            _, target_predicted = clean_target_logits.max(1)
            average_meter_set.update('target_accuracy', target_predicted.eq(target_labels).sum().item(),
                                     n=batch_size)
            average_meter_set.update('delta', cnn_drop_delta)

            target_logits_cnn_drop = self.classifier(target_dropout_mask_cnn * target_features2)
            target_logits_fc_drop = self.classifier(target_features2, target_dropout_mask_fc)
            target_consistency_loss = self.target_cnn_consistency_weight * self.target_consistency_criterion(
                target_logits_cnn_drop,
                target_logits1)
            target_consistency_loss += self.target_fc_consistency_weight * self.target_consistency_criterion(
                target_logits_fc_drop,
                target_logits1)
            target_entropy_loss = self.entmin_weight * self.entmin_criterion(target_logits1)
            target_loss = target_consistency_loss + target_entropy_loss

            # Class balance
            cls_balance_loss = 0.01 * (self.cls_balance_criterion(target_logits1) +
                                       self.cls_balance_criterion(target_logits_cnn_drop))
            target_loss += cls_balance_loss
            target_loss.backward()
            feature_extractor_grads = self.feature_extractor.module.stash_grad(feature_extractor_grads)
            classifier_grads = self.classifier.module.stash_grad(classifier_grads)

            average_meter_set.update('target_loss', target_loss.item())
            average_meter_set.update('target_consistency_loss', target_consistency_loss.item())
            average_meter_set.update('target_entropy_loss', target_entropy_loss.item())
            average_meter_set.update('target_vat_loss', target_vat_loss.item())
            average_meter_set.update('cls_balance_loss', cls_balance_loss.item())

            # Source CE Loss
            source_features1, source_features2 = self.feature_extractor(source_inputs)
            source_logits1 = self.classifier(source_features1)

            # Source pi model
            source_dropout_mask_cnn = torch.ones((*source_features2.size()[:2], 1, 1)).to(self.device)
            source_dropout_mask_fc = torch.ones(source_features1.size(0), self.fc_drop_size).to(self.device)

            source_logits_cnn_drop = self.classifier(source_dropout_mask_cnn * source_features2)
            source_logits_fc_drop = self.classifier(source_features2, source_dropout_mask_fc)
            ce_loss = self.class_criterion(source_logits1, source_labels)
            source_consistency_loss = self.source_cnn_consistency_weight * self.source_consistency_criterion(
                source_logits_cnn_drop, source_logits1)
            source_consistency_loss += self.source_fc_consistency_weight * self.source_consistency_criterion(
                source_logits_fc_drop, source_logits1)
            source_loss = ce_loss + source_consistency_loss
            source_loss.backward()

            average_meter_set.update('source_loss', source_loss.item())
            average_meter_set.update('ce_loss', ce_loss.item())
            average_meter_set.update('source_consistency_loss', source_consistency_loss.item())
            loss = source_loss.item() + target_loss.item()
            average_meter_set.update('loss', loss)
            average_meter_set.update('ce_loss_ratio', source_loss.item() / loss)
            average_meter_set.update('target_loss_ratio', target_loss.item() / loss)
            _, predictions_adv = source_logits_cnn_drop.max(1)
            _, predictions_clean = source_logits1.max(1)

            average_meter_set.update('adv_correct', predictions_adv.eq(source_labels).sum().item(), n=batch_size)
            average_meter_set.update('clean_correct', predictions_clean.eq(source_labels).sum().item(), n=batch_size)

            self.feature_extractor.module.restore_grad(feature_extractor_grads)
            self.classifier.module.restore_grad(classifier_grads)
            self.update_grad()
            self.reset_grad()

            tqdm_dataloader.set_description(
                ("Epoch {}, loss {:.3f}, ce_loss {:.3f}, " +
                 "target_loss {:.3f}, accuracy {:.3f} ").format(
                    epoch + 1, average_meter_set['loss'].avg, average_meter_set['ce_loss'].avg,
                    average_meter_set['target_loss'].avg, 100 * average_meter_set['clean_correct'].avg))

            accum_iter += batch_size

            if self._is_logging_needed(accum_iter):
                tqdm_dataloader.set_description("Logging to Tensorboard...")
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter
                }
                log_data.update(average_meter_set.averages())
                self._call_loggers(self.train_loggers, log_data)

            if self._is_validation_needed(accum_iter):
                self.validate(epoch, self.dataloaders['val'], self.dataloaders['source_val'], accum_iter)
                self.feature_extractor.train()
                self.classifier.train()

        return average_meter_set['loss'].avg, accum_iter

    def _is_logging_needed(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.batch_size and accum_iter != 0

    def _is_validation_needed(self, accum_iter):
        if self.validation_period_as_iter is None:
            return False
        else:
            return accum_iter % self.validation_period_as_iter < self.args.batch_size and accum_iter != 0

    def validate(self, epoch, dataloader, source_dataloader, accum_iter):
        self.feature_extractor.eval()
        self.classifier.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(dataloader)
            for batch_idx, (target_inputs, target_labels) in enumerate(tqdm_dataloader):
                target_inputs, target_labels = target_inputs.to(self.device), target_labels.to(self.device)

                target_features = self.feature_extractor(target_inputs)[0]
                target_logits = self.classifier(target_features)

                _, target_predictions = target_logits.max(1)
                average_meter_set.update('target_ce_loss', self.class_criterion(target_logits, target_labels))
                average_meter_set.update('target_correct', target_predictions.eq(target_labels).sum().item(),
                                         n=target_inputs.size(0))

            print("\ntarget_ce_loss: {:.3f}, target_accuracy: {:.3f}".format(
                average_meter_set['target_ce_loss'].avg, 100 * average_meter_set['target_correct'].avg,
            ))

            tqdm_source_dataloader = tqdm(source_dataloader)
            for batch_idx, (source_inputs, source_labels) in enumerate(tqdm_source_dataloader):
                source_inputs, source_labels = source_inputs.to(self.device), source_labels.to(self.device)

                source_logits = self.classifier(self.feature_extractor(source_inputs)[0])

                _, source_predictions = source_logits.max(1)
                average_meter_set.update('source_ce_loss', self.class_criterion(source_logits, source_labels))
                average_meter_set.update('source_correct', source_predictions.eq(source_labels).sum().item(),
                                         n=source_inputs.size(0))

                tqdm_source_dataloader.set_description("source_ce_loss1: {:.3f}, source_correct1: {:.3f},".format(
                    average_meter_set['source_ce_loss'].avg, 100 * average_meter_set['source_correct'].avg,
                ))

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'target_ce_loss': average_meter_set['target_ce_loss'].avg,
                'target_accuracy': 100 * average_meter_set['target_correct'].avg,
                'source_ce_loss': average_meter_set['source_ce_loss'].avg,
                'source_accuracy': 100 * average_meter_set['source_correct'].avg,
            }
            self._call_loggers(self.val_loggers, log_data)

    def _create_state_dict(self):
        return {
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': {k: optimizer.state_dict() for k, optimizer in self.optimizers.items()},
        }
