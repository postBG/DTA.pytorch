import os
import pprint as pp

import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from consistency_losses import get_consistency_loss
from datasets import dataloaders_factory
from loggers import MetricGraphPrinter, RecentModelCheckpointLogger, BestAccuracyModelTracker
from losses import EntropyLoss, ClassBalanceLoss
from misc import set_up_gpu, fix_random_seed_as, create_experiment_export_folder, export_experiments_config_as_json
from models import create_feature_extractor, create_class_classifier
from options import args as parsed_args
from trainers.dta_trainer import DTATrainer
from trainers.source_only_trainer import SourceOnlyTrainer


def main(args, trainer_cls):
    export_root, args = setup_experiments(args)

    dataloaders = dataloaders_factory(args)

    feature_extractor, classifier = create_feature_extractor(args), create_class_classifier(args)
    models = {
        'feature_extractor': feature_extractor,
        'classifier': classifier,
    }

    writer = SummaryWriter(os.path.join(export_root, 'logs'))

    update_feature_extractor_and_classifier = [
        MetricGraphPrinter(writer, key='clean_correct',
                           graph_label='clean_accuracy', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='loss',
                           graph_label='loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='ce_loss',
                           graph_label='ce_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='source_loss',
                           graph_label='source_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='source_consistency_loss',
                           graph_label='source_consistency_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='target_loss',
                           graph_label='target_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='target_consistency_loss',
                           graph_label='target_consistency_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='target_entropy_loss',
                           graph_label='target_entropy_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='target_vat_loss',
                           graph_label='target_vat_loss', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='target_accuracy',
                           graph_label='target_accuracy', group_name='update_feature_extractor_and_classifier'),
        MetricGraphPrinter(writer, key='cls_balance_loss', graph_label='class balance loss',
                           group_name='update_feature_extractor_and_classifier'),
    ]
    extra_analysis = [
        MetricGraphPrinter(writer, key='delta', graph_label='delta', group_name='analysis'),
    ]
    train_loggers = [MetricGraphPrinter(writer, key='epoch',
                                        graph_label='Epoch')] + update_feature_extractor_and_classifier + extra_analysis

    val_target_loggers = [
        MetricGraphPrinter(writer, key='target_ce_loss',
                           graph_label='target_ce_loss', group_name='val_target'),
        MetricGraphPrinter(writer, key='target_accuracy',
                           graph_label='target_accuracy', group_name='val_target'),
    ]
    val_loggers = [
                      RecentModelCheckpointLogger(os.path.join(export_root, 'experiments'),
                                                  checkpoint_period=args.checkpoint_period),
                      BestAccuracyModelTracker(os.path.join(export_root, 'experiments'),
                                               metric_key='target_accuracy'),
                  ] + val_target_loggers

    criterions = {
        'classifier': nn.CrossEntropyLoss(),
        'source_consistency': get_consistency_loss(args.source_consistency_loss),
        'target_consistency': get_consistency_loss(args.target_consistency_loss),
        'entmin': EntropyLoss(),
        'class_balance': ClassBalanceLoss()
    }

    optimizers = create_optimizers(args, feature_extractor, classifier)

    schedulers = {
        'feature_extractor': optim.lr_scheduler.StepLR(optimizers['feature_extractor'], step_size=args.decay_step,
                                                       gamma=args.gamma),
        'classifier': optim.lr_scheduler.StepLR(optimizers['classifier'], step_size=args.decay_step,
                                                gamma=args.gamma),
    }

    config_str = pp.pformat(vars(args), width=1)
    config_str = config_str.replace('\n', '  \n')
    writer.add_text('config', config_str, 0)

    trainer = trainer_cls(models, dataloaders, optimizers, criterions, args.epoch, args,
                          log_period_as_iter=args.log_period_as_iter, train_loggers=train_loggers,
                          val_loggers=val_loggers, lr_schedulers=schedulers, pretrain_epochs=10)
    trainer.train()
    writer.close()


def create_optimizers(args, feature_extractor, classifier):
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


def setup_experiments(args):
    set_up_gpu(args)
    fix_random_seed_as(args.random_seed)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint(vars(args), width=1)
    return export_root, args


if __name__ == "__main__":
    if parsed_args.train_mode == 'source_only':
        main(parsed_args, SourceOnlyTrainer)
    elif parsed_args.train_mode == 'dta':
        main(parsed_args, DTATrainer)
