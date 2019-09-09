import pprint as pp

import os
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from consistency_losses import get_consistency_loss
from datasets import dataloaders_factory
from loggers import MetricGraphPrinter, RecentModelCheckpointLogger, BestModelTracker
from losses import EntropyLoss, ClassBalanceLoss
from misc import set_up_gpu, fix_random_seed_as, create_experiment_export_folder, export_experiments_config_as_json
from models import create_feature_extractor, create_class_classifier
from options import get_parsed_args, parser
from trainers.dta_trainer import DTATrainer


def main(args, trainer_cls):
    export_root, args = _setup_experiments(args)

    dataloaders = dataloaders_factory(args)

    feature_extractor, classifier = create_feature_extractor(args), create_class_classifier(args)
    models = {
        'feature_extractor': feature_extractor,
        'classifier': classifier,
    }

    writer = SummaryWriter(os.path.join(export_root, 'logs'))
    train_loggers, val_loggers = setup_loggers(args, export_root, writer)
    export_configs_to_tensorboard(args, writer)

    criterions = {
        'classifier': nn.CrossEntropyLoss(),
        'source_consistency': get_consistency_loss(args.source_consistency_loss),
        'target_consistency': get_consistency_loss(args.target_consistency_loss),
        'entmin': EntropyLoss(),
        'class_balance': ClassBalanceLoss()
    }

    optimizers = _create_optimizers(args, feature_extractor, classifier)

    schedulers = {
        'feature_extractor': optim.lr_scheduler.StepLR(optimizers['feature_extractor'], step_size=args.decay_step,
                                                       gamma=args.gamma),
        'classifier': optim.lr_scheduler.StepLR(optimizers['classifier'], step_size=args.decay_step,
                                                gamma=args.gamma),
    }

    trainer = trainer_cls(models, dataloaders, optimizers, criterions, args.epoch, args,
                          log_period_as_iter=args.log_period_as_iter, train_loggers=train_loggers,
                          val_loggers=val_loggers, lr_schedulers=schedulers, pretrain_epochs=10)
    trainer.train()
    writer.close()


def setup_loggers(args, export_root, writer):
    loggers_for_train_status = [
        MetricGraphPrinter(writer, key='clean_correct', namespace='train_status'),
        MetricGraphPrinter(writer, key='loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='ce_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='source_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='source_consistency_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='target_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='target_consistency_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='target_entropy_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='target_vat_loss', namespace='train_status'),
        MetricGraphPrinter(writer, key='target_accuracy', namespace='train_status'),
        MetricGraphPrinter(writer, key='cls_balance_loss', namespace='train_status')
    ]
    extra_analysis = [
        MetricGraphPrinter(writer, key='delta', namespace='analysis'),
    ]
    train_loggers = [MetricGraphPrinter(writer, key='epoch')] + loggers_for_train_status + extra_analysis
    val_loggers = [
        MetricGraphPrinter(writer, key='target_ce_loss', namespace='val_target'),
        MetricGraphPrinter(writer, key='target_accuracy', namespace='val_target'),
        RecentModelCheckpointLogger(os.path.join(export_root, 'models'), checkpoint_period=args.checkpoint_period),
        BestModelTracker(os.path.join(export_root, 'models'), metric_key='target_accuracy')
    ]
    return train_loggers, val_loggers


def export_configs_to_tensorboard(args, writer):
    config_str = pp.pformat(vars(args), width=1)
    config_str = config_str.replace('\n', '  \n')
    writer.add_text('config', config_str, 0)


def _create_optimizers(args, feature_extractor, classifier):
    if args.optimizer == 'Adam':
        return {
            'feature_extractor': optim.Adam(feature_extractor.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay),
            'classifier': optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        }

    return {
        'feature_extractor': optim.SGD(feature_extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                       momentum=args.momentum),
        'classifier': optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum),
    }


def _setup_experiments(args):
    set_up_gpu(args)
    fix_random_seed_as(args.random_seed)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint(vars(args), width=1)
    return export_root, args


if __name__ == "__main__":
    parsed_args = get_parsed_args(parser)
    main(parsed_args, DTATrainer)
