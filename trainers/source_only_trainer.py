import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import AverageMeterSet


class SourceOnlyTrainer(object):
    def __init__(self, models, dataloaders, optimizers, criterions, num_epochs, args,
                 log_period_as_iter=40000, train_loggers=None, val_loggers=None, device='cuda',
                 lr_schedulers=None, pretrain_epochs=0):
        self.feature_extractor = models['feature_extractor'].to(device)
        self.feature_extractor = nn.DataParallel(self.feature_extractor) if args.num_gpu > 1 else self.feature_extractor

        self.classifier = models['classifier'].to(device)
        self.classifier = nn.DataParallel(self.classifier) if args.num_gpu > 1 else self.classifier

        self.args = args

        self.dataloaders = dataloaders
        self.optimizers = optimizers
        self.criterions = criterions
        self.class_criterion = self.criterions['classifier']
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
        self.validate(0, self.dataloaders['val'], accum_iter)
        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    _, accum_iter = self.train_one_epoch(epoch, self.dataloaders[phase], accum_iter)
                else:
                    self.validate(epoch, self.dataloaders['val'], accum_iter)

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

    def train_one_epoch(self, epoch, dataloader, accum_iter):
        self.feature_extractor.train()
        self.classifier.train()

        for scheduler in self.lr_schedulers.values():
            scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(dataloader, ncols=150)

        for batch_idx, ((source_inputs, source_labels), (_, _)) in enumerate(tqdm_dataloader):
            batch_size = source_inputs.size(0)
            source_inputs, source_labels = source_inputs.to(self.device), source_labels.to(self.device)

            self.reset_grad()

            # Source CE Loss

            source_features1, _ = self.feature_extractor(source_inputs)
            source_logits1 = self.classifier(source_features1)

            ce_loss = self.class_criterion(source_logits1, source_labels)
            ce_loss.backward()

            average_meter_set.update('ce_loss', ce_loss.item())
            _, predictions_clean = source_logits1.max(1)

            average_meter_set.update('clean_correct', predictions_clean.eq(source_labels).sum().item(), n=batch_size)

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
        return accum_iter % self.validation_period_as_iter < self.args.batch_size and accum_iter != 0

    def validate(self, epoch, target_dataloader, accum_iter):
        self.feature_extractor.eval()
        self.classifier.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(target_dataloader)
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
