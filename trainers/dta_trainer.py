from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from dropout import create_adversarial_dropout_mask, calculate_jacobians
from losses import EntropyLoss
from metrics import AverageMeterSet
from models.visda_architectures import ResNetLower, ResNetUpper
from ramps import linear_rampup
from utils import disable_tracking_bn_stats
from vat import VirtualAdversarialPerturbationGenerator


class DTATrainer(object):
    def __init__(self, models, dataloaders, optimizers, criterions, num_epochs, args,
                 log_period_as_iter=40000, train_loggers=None, val_loggers=None, device='cuda',
                 lr_schedulers=None, pretrain_epochs=0):
        assert isinstance(models['feature_extractor'], ResNetLower)
        assert isinstance(models['classifier'], ResNetUpper)

        self.feature_extractor = nn.DataParallel(models['feature_extractor']).to(device)
        self.classifier = nn.DataParallel(models['classifier']).to(device)

        self.optimizers = optimizers

        self.class_criterion = criterions['classifier']
        self.source_consistency_criterion = criterions['source_consistency']
        self.target_consistency_criterion = criterions['target_consistency']
        self.entmin_criterion = criterions['entmin']
        self.cls_balance_criterion = criterions['class_balance']

        self.entropy_loss = EntropyLoss()

        self.target_fc_consistency_weight = args.target_fc_consistency_weight
        self.target_cnn_consistency_weight = args.target_cnn_consistency_weight
        self.source_fc_consistency_weight = args.source_fc_consistency_weight
        self.source_cnn_consistency_weight = args.source_cnn_consistency_weight
        self.entmin_weight = args.entmin_weight
        self.cls_balance_weight = args.cls_balance_weight
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
        # self.validate(0, self.dataloaders['val'], accum_iter)
        for epoch in range(self.num_epochs):
            _, accum_iter = self.train_one_epoch(epoch, self.dataloaders['train'], accum_iter)
            self._step_schedulers()
            self.validate(epoch, self.dataloaders['val'], accum_iter)

        self._complete_logging({
            'state_dict': (self._create_state_dict())
        })

    def _step_schedulers(self):
        for scheduler in self.lr_schedulers.values():
            scheduler.step()

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

    def train_one_epoch(self, epoch, dataloader, accum_iter):
        self.feature_extractor.train()
        self.classifier.train()

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
                feature_extractor_grads = self.feature_extractor.module.stash_grad(feature_extractor_grads)
                classifier_grads = self.classifier.module.stash_grad(classifier_grads)
            else:
                target_vat_loss = torch.tensor(0).to(self.device)

            # Target Consistency Loss
            # target_features1 = clean, target_features2 = adv drop
            target_features1, target_features2 = self.feature_extractor(target_inputs)
            target_logits1 = self.classifier(target_features1)

            # Target AdD
            jacobian_for_cnn_adv_drop, jacobian_for_fc_adv_drop, clean_target_logits = calculate_jacobians(
                target_features2.detach(), target_logits1.detach(), self.classifier, self.classifier.module.drop_size,
                self.target_consistency_criterion, self.reset_grad)

            cnn_drop_delta = linear_rampup(epoch, self.rampup_length) * self.cnn_delta
            target_cnn_dropout_mask, _ = create_adversarial_dropout_mask(
                torch.ones_like(jacobian_for_cnn_adv_drop),
                jacobian_for_cnn_adv_drop, cnn_drop_delta)

            fc_drop_delta = linear_rampup(epoch, self.rampup_length) * self.fc_delta
            target_fc_dropout_mask, _ = create_adversarial_dropout_mask(
                torch.ones_like(jacobian_for_fc_adv_drop),
                jacobian_for_fc_adv_drop, fc_drop_delta)

            _, target_predicted = clean_target_logits.max(1)
            average_meter_set.update('target_accuracy', target_predicted.eq(target_labels).sum().item(),
                                     n=batch_size)
            average_meter_set.update('delta', cnn_drop_delta)

            target_logits_cnn_drop = self.classifier(target_cnn_dropout_mask * target_features2)
            target_logits_fc_drop = self.classifier(target_features2, target_fc_dropout_mask)
            target_consistency_loss = self.target_cnn_consistency_weight * self.target_consistency_criterion(
                target_logits_cnn_drop,
                target_logits1)
            target_consistency_loss += self.target_fc_consistency_weight * self.target_consistency_criterion(
                target_logits_fc_drop,
                target_logits1)
            target_entropy_loss = self.entmin_weight * self.entmin_criterion(target_logits1)
            target_loss = target_consistency_loss + target_entropy_loss

            # Class balance
            cls_balance_loss = self.cls_balance_weight * (self.cls_balance_criterion(target_logits1) +
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
            source_logits1, source_logits2 = self.classifier(source_features1), self.classifier(source_features2)

            # Source pi model
            ce_loss = self.class_criterion(source_logits1, source_labels)
            source_consistency_loss = 2 * self.source_cnn_consistency_weight * self.source_consistency_criterion(
                source_logits2, source_logits1)
            source_loss = ce_loss + source_consistency_loss
            source_loss.backward()

            average_meter_set.update('source_loss', source_loss.item())
            average_meter_set.update('ce_loss', ce_loss.item())
            average_meter_set.update('source_consistency_loss', source_consistency_loss.item())
            loss = source_loss.item() + target_loss.item()
            average_meter_set.update('loss', loss)
            _, predictions_clean = source_logits1.max(1)

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
                self.validate(epoch, self.dataloaders['val'], accum_iter)
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

    def validate(self, epoch, dataloader, accum_iter):
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

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'target_ce_loss': average_meter_set['target_ce_loss'].avg,
                'target_accuracy': 100 * average_meter_set['target_correct'].avg,
            }
            self._call_loggers(self.val_loggers, log_data)

    def _create_state_dict(self):
        return {
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': {k: optimizer.state_dict() for k, optimizer in self.optimizers.items()},
        }
