import argparse
import json

from datasets import DATA_SETS

parser = argparse.ArgumentParser(description='Options for DTA.pytorch')

#########################
# Load Template
#########################
parser.add_argument('--config_path', type=str, default='', help='config json path')

#########################
# General Train Settings
#########################
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--epoch', type=int, default=80, help='epoch (default: 100)')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs')
parser.add_argument('--device_idx', type=str, default='0', help='Gpu idx')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization lambda (default: 0)')
parser.add_argument('--decay_step', type=int, default=15, help='num epochs for decaying learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay gamma')
parser.add_argument('--log_period_as_iter', type=int, default=12800, help='num iter')
parser.add_argument('--validation_period_as_iter', type=int, default=30000, help='validation period in iterations')
parser.add_argument('--test', type=bool, default=False, help='is Test')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
parser.add_argument('--source_dataset_code', type=str, default='stl9', choices=list(DATA_SETS.keys()),
                    help='Source DataSet Code')
parser.add_argument('--target_dataset_code', type=str, default='cifar9', choices=list(DATA_SETS.keys()),
                    help='Target DataSet Code')
parser.add_argument('--transform_type', type=str, default='standard', help='Transform type')
parser.add_argument('--classifier_ckpt_path', type=str, default='', help='Domain Classifier Checkpoint Path')
parser.add_argument('--encoder_ckpt_path', type=str, default='', help='Encoder Checkpoint Path')
parser.add_argument('--pretrain', type=str, default='',
                    choices=['class_classifier', 'domain_classifier', ''], help='Pretrain mode')
parser.add_argument('--freeze_encoder', type=bool, default=False, help='Freeze Encoder')
parser.add_argument('--ct_threshold', type=float, default=0.968, help='Confidence Threshold')
parser.add_argument('--use_ct', type=bool, default=False, help='Use Confidence Threshold')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--model', type=str, default='resnet50', help='Model: resnet50 | resnet101')
parser.add_argument('--rampup_length', type=int, default=1, help='Ramp up length')
parser.add_argument('--source_rampup_length', type=int, default=1, help='Source Ramp up length')
parser.add_argument('--random_seed', type=int, default=0, help='Random seed value')
parser.add_argument('--target_consistency_loss', type=str, default='l1', choices=['l1', 'l2', 'kld'],
                    help='Target Consistency Loss')
parser.add_argument('--source_consistency_loss', type=str, default='l2', choices=['l1', 'l2', 'kld'],
                    help='Source Consistency Loss')
parser.add_argument('--train_mode', type=str, default='dta', choices=['dta', 'source_only'],
                    help='Train mode')

#########################
# Adversarial Dropout Settings
#########################
parser.add_argument('--drop_prob', type=float, default=0.0, help='Probability of dropping unit')
parser.add_argument('--target_consistency_weight', type=float, default=1, help='Target Consistency Weight')
parser.add_argument('--source_consistency_weight', type=float, default=1, help='Source Consistency Weight')
parser.add_argument('--target_fc_consistency_weight', type=float, default=2, help='Target FCdrop consistency weight')
parser.add_argument('--target_cnn_consistency_weight', type=float, default=2, help='Target CNNdrop consistency weight')
parser.add_argument('--source_fc_consistency_weight', type=float, default=1, help='Source FCdrop Consistency Weight')
parser.add_argument('--source_cnn_consistency_weight', type=float, default=1, help='Source CNNdrop Consistency Weight')
parser.add_argument('--cls_balance_weight', type=float, default=0.01, help='Class balance loss Weight')

parser.add_argument('--entmin_weight', type=float, default=0., help='Target EntMin Weight')
parser.add_argument('--delta', type=float, default=0.01, help='delta')
parser.add_argument('--cnn_delta', type=float, default=0.01, help='cnn delta')
parser.add_argument('--fc_delta', type=float, default=0.1, help='fc delta')
parser.add_argument('--source_delta', type=float, default=0.1, help='Delta value for source')
parser.add_argument('--source_delta_fc', type=float, default=0.1, help='Delta value for source, fc layer')

#########################
# VAT settings
#########################
parser.add_argument('--use_vat', type=bool, default=False, help='Use VAT or not')
parser.add_argument('--xi', type=float, default=1e-6, help='VAT xi')
parser.add_argument('--ip', type=int, default=1, help='VAT ip')
parser.add_argument('--eps', type=float, default=3.5, help='VAT eps')
parser.add_argument('--source_vat_loss_weight', type=float, default=0., help='vat loss source weight')
parser.add_argument('--target_vat_loss_weight', type=float, default=0., help='vat loss target weight')

#########################
# Experiment Logging Settings
#########################
parser.add_argument('--experiment_dir', type=str, default='', help='Experiment save directory')
parser.add_argument('--experiment_description', type=str, default='svhn_mnist', help='Experiment description')
parser.add_argument('--checkpoint_period', type=int, default=1, help='epoch / checkpoint_period = checkpoint num')
parser.add_argument('--checkpoint_path', type=str, default='./jigsaw_2018-11-02_2/experiments/best_acc_model.pth',
                    help='Checkpoint path')


def _load_experiments_config_from_json(args, json_path, arg_parser):
    with open(json_path, 'r') as f:
        config = json.load(f)

    for config_name, config_val in config.items():
        if config_name in args.__dict__ and getattr(args, config_name) == arg_parser.get_default(config_name):
            setattr(args, config_name, config_val)

    print("Config at '{}' has been loaded".format(json_path))


def get_parsed_args(arg_parser: argparse.ArgumentParser):
    args = arg_parser.parse_args()
    if args.config_path:
        _load_experiments_config_from_json(args, args.config_path, arg_parser)
    return args
