import os
import torch
from abc import ABC
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


class AbstractLogger(ABC):
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


class BestModelTracker(AbstractLogger):
    def __init__(self, export_path, metric_key, ckpt_filename='best_acc_model.pth'):
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


class MetricGraphPrinter(AbstractLogger):
    def __init__(self, writer, key='train_loss', graph_label=None, namespace='metric'):
        self.key = key
        self.graph_label = graph_label if graph_label else self.key
        self.group_name = namespace
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()
