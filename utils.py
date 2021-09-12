import shutil
import torch
import os
from os.path import exists, join, split
import multiprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255



def getDictImageNetClasses(path_imagenet_classes_name='imagenet_list.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    list_imagenet_classname=[]
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        print(line)
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            if cat_name in dict_imagenet_classname2id.keys():
                print(cat_name)
            dict_imagenet_classname2id[id] = cat_name.lower()
            count += 1
            list_imagenet_classname.append(id)
            # print(cat_name, id)
            line = f.readline()
    # print("Total categories categories", count)

    return dict_imagenet_classname2id, list_imagenet_classname



def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.train()

# model.apply(set_bn_eval)


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }


#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'





def objectnet_accuracy_B(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    cnt = 0
    top1_cnt = 0
    top5_cnt = 0

    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)

        output = output.data
        _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)  # Oh forget to sorted it


        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        pred = pred.cpu().numpy()

        # target = target # Labels are splited by -1

        target_num = len(target)
        target = [each.numpy() for each in target]

        for jj in range(batch_size):
            # label_gt_all = target[jj]
            label_gt_list = []

            for kk in range(target_num):
                label_gt_list.append(target[kk][jj])

            cnt += 1

            pred_index = pred[jj]

            for label_gt in label_gt_list:
                if pred_index[0] == label_gt:
                    top1_cnt += 1
                    break

            flag = True
            for nnn in range(5):  # If each batch is from same category, can make it in matrix to speed up
                if flag == False:
                    break
                for label_gt in label_gt_list:
                    if pred_index[nnn] == label_gt:
                        top5_cnt += 1
                        flag = False
                        break

        return top1_cnt * 100. / cnt, top5_cnt * 100. / cnt



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        print(pred)
        print(target)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print('correct size', correct.size())
        # print('correct size', correct[:5].size())
        # print('correct size', correct[:5].reshape(-1).size())
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res





