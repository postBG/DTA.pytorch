import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


class AbstractLogger(object):
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        raise NotImplementedError


def _checkpoint_file_path(export_path, filename):
    return os.path.join(export_path, filename)


class RecentModelCheckpointLogger(AbstractLogger):
    def __init__(self, export_path, checkpoint_period, ckpt_filename='checkpoint-recent.pth'):
        self.export_path = export_path
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)
        self.checkpoint_period = checkpoint_period
        self.call_count = 0
        self.ckpt_filename = ckpt_filename
        self.ckpt_final_filename = self.ckpt_filename + '.final'

    def log(self, *args, **kwargs):
        self.call_count += 1

        if self.call_count % self.checkpoint_period == 0:
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            state_dict['accum_iter'] = kwargs['accum_iter']
            torch.save(state_dict, _checkpoint_file_path(self.export_path, self.ckpt_filename))

    def complete(self, *args, **kwargs):
        torch.save(kwargs['state_dict'], _checkpoint_file_path(self.export_path, self.ckpt_final_filename))


class PerEpochModelCheckpointLogger(AbstractLogger):
    def __init__(self, export_path, checkpoint_period, ckpt_filename='checkpoint-recent.pth'):
        self.export_path = export_path
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)
        self.checkpoint_period = checkpoint_period
        self.call_count = 0
        self.ckpt_filename = ckpt_filename
        self.ckpt_final_filename = self.ckpt_filename + '.final'

    def log(self, *args, **kwargs):
        self.call_count += 1

        if self.call_count % self.checkpoint_period == 0:
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            state_dict['accum_iter'] = kwargs['accum_iter']
            torch.save(state_dict, _checkpoint_file_path(self.export_path, self.ckpt_filename + str(self.call_count)))

    def complete(self, *args, **kwargs):
        torch.save(kwargs['state_dict'], _checkpoint_file_path(self.export_path, self.ckpt_final_filename))


class BestAccuracyModelTracker(AbstractLogger):
    def __init__(self, export_path, metric_key='accuracy', ckpt_filename='best_acc_model.pth'):
        self.export_path = export_path
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)

        self.best_accuracy = 0.
        self.metric_key = metric_key
        self.ckpt_filename = ckpt_filename

    def log(self, *args, **kwargs):
        acc = kwargs[self.metric_key]
        if self.best_accuracy < acc:
            print("Update Best Accuracy Model at {}".format(kwargs['epoch']))
            self.best_accuracy = acc
            torch.save(kwargs['state_dict'], _checkpoint_file_path(self.export_path, self.ckpt_filename))

    def complete(self, *args, **kwargs):
        pass


# TODO: Refactor this
class MetricGraphPrinter(AbstractLogger):
    def __init__(self, writer, key='train_loss', graph_label='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_label
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()


# TODO: Refactor this
class ParamsHistogramPrinter(AbstractLogger):
    def __init__(self, writer, model, param_name='fc1.weight', graph_label='Fc1 Weight',
                 group_name='Train Parameters'):
        self.model = model
        self.param_name = param_name
        self.graph_label = graph_label
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        for name, param in self.model.named_parameters():
            if name == self.param_name:
                self.writer.add_histogram(self.group_name + '/' + self.graph_label,
                                          param.clone().data, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()


# TODO: Refactor this
class HistogramPrinter(AbstractLogger):
    def __init__(self, writer, key='source_dropout_frequencies_accum', graph_label='Source Dropout Frequency',
                 group_name='histogram'):
        self.key = key
        self.graph_label = graph_label
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        frequency_data = kwargs[self.key]
        fig = plt.figure()
        fig.tight_layout(pad=0)
        ax1 = fig.add_subplot(111)
        ax1.bar(np.arange(0, len(frequency_data)), frequency_data)

        self.writer.add_figure(self.group_name + '/' + self.graph_label, fig, kwargs['epoch'])

    def complete(self, *args, **kwargs):
        self.writer.close()
