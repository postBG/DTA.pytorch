# TODO: replace with json config
def set_template(args):
    print("Template Setting: {}".format(args.template))

    if args.template == '':
        pass

    elif args.template == 'fc_drop':
        args.device_idx = '0,1,2,3'
        # args.device_idx = '4,5,6,7'
        args.train_mode = 'fc_add'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'fc_experiments'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.02
        args.delta = 0.1
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.rampup_length = 20
        args.random_seed = 1
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'l2'
        args.model = 'resnet50'
        args.source_delta = 0.1
        args.use_vat = True
        args.eps = 15
        args.target_vat_loss_weight = 0.1
        args.experiment_description = 'res50_fcadd_VAT_0.1_entmin0.02'

    elif args.template == 'cnn_drop':
        # args.device_idx = '0,1,2,3'
        args.device_idx = '4,5,6,7'
        args.train_mode = 'cnn_add'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'cnn_reproduce_exp'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.01
        args.delta = 0.01
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.rampup_length = 30
        args.random_seed = 12345
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'kld'
        args.model = 'resnet101'
        args.source_delta = 0.0025
        args.use_vat = True
        args.eps = 15
        args.target_vat_loss_weight = 0.1
        args.experiment_description = 'res101_convadd_best_reproduce'

    elif args.template == 'cnn_drop_source_only':
        args.device_idx = '0,1,2,3'
        # args.device_idx = '4,5,6,7'
        args.train_mode = 'cnn_add'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'source_only'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.00
        args.delta = 0.0
        args.target_consistency_weight = 0
        args.source_consistency_weight = 0
        args.rampup_length = 30
        args.random_seed = 12345
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'kld'
        args.model = 'resnet101'
        args.source_delta = 0.0025
        args.use_vat = False
        args.eps = 15
        args.target_vat_loss_weight = 0
        args.experiment_description = 'res101_source_only'

    elif args.template == 'joint_cnn_fc_drop':
        # args.device_idx = '4,5,6,7'
        args.device_idx = '4,5,6,7'
        args.train_mode = 'cnn_fc_add'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'joint_cnn_fc_experiments'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.01
        args.delta = 0.01
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.rampup_length = 50
        args.random_seed = 12345
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'kld'
        args.model = 'resnet101'
        args.source_delta = 0.0025
        args.use_vat = False
        args.eps = 15
        args.target_vat_loss_weight = 0.1
        args.experiment_description = 'joint_novat_targcons2_r50'

    elif args.template == 'joint_cnn_fc_drop_VAT':
        # args.device_idx = '4,5,6,7'
        args.device_idx = '0,1,2,3'
        args.train_mode = 'cnn_fc_add'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'joint_cnn_fc_experiments'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.01
        args.delta = 0.01
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.rampup_length = 30
        args.random_seed = 12345
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'kld'
        args.model = 'resnet101'
        args.source_delta = 0.0025
        args.use_vat = True
        args.eps = 15
        args.target_vat_loss_weight = 0.1
        args.experiment_description = 'joint_vat_targcons2'

    elif args.template == 'res50_rnd_seed':
        # args.device_idx = '4,5,6,7'
        # args.device_idx = '0,1,2,3'
        args.test = True
        args.train_mode = 'dta'
        args.batch_size = 128
        args.device_idx = '0,1'
        args.lr = 0.001
        args.epoch = 20
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'random_seed_exp'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.entmin_weight = 0.02
        args.delta = 0.01
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.rampup_length = 20
        args.random_seed = -1
        args.target_consistency_loss = 'kld'
        args.source_consistency_loss = 'l2'
        args.model = 'resnet50'
        args.source_delta = 0.0025
        args.use_vat = False
        args.eps = 15
        args.target_vat_loss_weight = 0.2
        args.experiment_description = 'res50_best_randomseed'

    elif args.template == 'source_only':
        args.train_mode = 'source_only'
        # args.device_idx = '4,5,6,7'
        args.device_idx = '0,1,2,3'
        args.batch_size = 128
        args.lr = 0.001
        args.epoch = 15
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 10
        args.gamma = 0.5
        args.weight_decay = 5e-4
        args.experiment_dir = 'source_only'
        args.log_period_as_iter = 10000
        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'
        args.transform_type = 'visda_standard'
        args.model = 'resnet50'
        args.experiment_description = 'res50_source_only'

    elif args.template == 'layer_ablation_vat':
        args.lr = 0.001
        args.epoch = 20
        # args.device_idx = "4,5,6,7"
        args.weight_decay = 5e-4
        args.decay_step = 10
        args.momentum = 0.9
        args.gamma = 0.5
        args.log_period_as_iter = 10000
        args.batch_size = 128
        args.source_dataset_code = "visda_source"
        args.target_dataset_code = "visda_target"
        args.transform_type = "visda_standard"
        args.optimizer = "SGD"
        args.model = "resnet50"
        args.rampup_length = 20
        args.source_rampup_length = 1
        args.random_seed = 12345
        args.target_consistency_loss = "kld"
        args.source_consistency_loss = "l2"
        args.train_mode = "dta"
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.entmin_weight = 0.02
        args.delta = 0.01
        args.source_delta = 0.0025
        args.source_delta_fc = 0.1
        args.use_vat = True
        args.eps = 15
        args.source_vat_loss_weight = 0.0
        args.target_vat_loss_weight = 0.2
        args.experiment_dir = "layer_ablation_res50"

    elif args.template == 'layer_ablation_novat':
        args.lr = 0.001
        args.epoch = 20
        # args.device_idx = "4,5,6,7"
        args.weight_decay = 5e-4
        args.decay_step = 10
        args.momentum = 0.9
        args.gamma = 0.5
        args.log_period_as_iter = 10000
        args.batch_size = 128
        args.source_dataset_code = "visda_source"
        args.target_dataset_code = "visda_target"
        args.transform_type = "visda_standard"
        args.optimizer = "SGD"
        args.model = "resnet50"
        args.rampup_length = 20
        args.source_rampup_length = 1
        args.random_seed = 12345
        args.target_consistency_loss = "kld"
        args.source_consistency_loss = "l2"
        args.train_mode = "dta"
        args.target_consistency_weight = 2
        args.source_consistency_weight = 1
        args.entmin_weight = 0.02
        args.delta = 0.01
        args.source_delta = 0.0025
        args.source_delta_fc = 0.1
        args.use_vat = False
        args.eps = 15
        args.source_vat_loss_weight = 0.0
        args.target_vat_loss_weight = 0.2
        args.experiment_dir = "layer_ablation_res50"

    elif args.template == 'res152_test':
        args.train_mode = 'dta'
        args.decay_step = 10
        args.epoch = 20
        args.eps = 15
        args.experiment_dir = 'test_on_152'
        args.gamma = 0.5
        args.log_period_as_iter = 10000
        args.lr = 0.001
        args.validation_period_as_iter = 30000
        args.weight_decay = 5e-4
        args.model = 'resnet152'
        args.momentum = 0.9
        args.optimizer = 'SGD'
        args.transform_type = 'visda_standard'
        args.batch_size = 64

        args.source_dataset_code = 'visda_source'
        args.target_dataset_code = 'visda_target'

        args.source_consistency_weight = 1
        args.target_consistency_weight = 2.0
        args.source_consistency_loss = 'kld'
        args.target_consistency_loss = 'kld'

        args.delta = 0.01
        args.fc_delta = 0.1
        args.source_delta = 0.0025
        args.source_delta_fc = 0.1

        args.source_rampup_length = 1
        args.source_vat_loss_weight = 0.0
        args.entmin_weight = 0.02
        args.target_vat_loss_weight = 0.2

        args.use_vat = True

    else:
        raise ValueError
