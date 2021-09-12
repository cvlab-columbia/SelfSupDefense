
import os
import argparse
import logging
import sys
import math, time
import random
''''If you see bad robustness, it is due to use the wrong normalization in L26-32'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from learning.wideresnet import WideResNet, WideResNetBD, WideResNetMed_SSL, WRN34_out_branch
from learning.preactresnet import PreActResNet18Mhead, Res18_out3_model, Res18_out4_model, Res18_out5_model,Res18_out6_model
from utils import *

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# def normalize(X):
#     return (X - mu)/std
def normalize(X):
    return X

def normal_guassian_normalize(T):
    return (T-T.mean()) / T.std()

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)



def attack_constrastive_Mhead(model, model_ssl, scripted_transforms, criterion, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, Ltype=None, reverse=False, n_views=2):
    """Reverse algorithm that optimize the SSL loss via PGD"""

    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):

        new_x = X + delta
        # import pdb; pdb.set_trace()

        # TODO: here the neg sample is fixed, we can also try random neg sample to enlarge and diversify
        loss = -calculate_contrastive_Mhead_loss(new_x, scripted_transforms, model, criterion,
                                                     model_ssl, no_grad=False, n_views=n_views)
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta



def constrastive_loss_func(contrastive_head, criterion, bs, n_views):
    """Loss function for contrastive SSL learning"""
    features = F.normalize(contrastive_head, dim=1)

    labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    temperature = 0.2
    logits = logits / temperature

    xcontrast_loss = criterion(logits, labels)

    correct = (logits.max(1)[1] == labels).sum().item()
    return xcontrast_loss, correct





def calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, submodel, no_grad=True, n_views=2):
    """Creating multiviews for contrastive SSL loss

    Attributes:
        X: input image array.
        scripted_transforms: transformation for generating different views for SSL.
        model: classifier backbone model.
        criterion: cross-entropy.
        submodel: SSL model that do contrastive loss
        no_grad: do not backpropagate
        n_views: views generated for contrastive learning.

    Return:
        closs: SSL loss.
    """
    new_x = X
    bs = X.size(0)

    if n_views == 2:
        X_transformed1 = scripted_transforms(new_x)
        X_transformed2 = scripted_transforms(new_x)
        X_constrastive = torch.cat([X_transformed1, X_transformed2], dim=0)
    elif n_views ==4:
        X_transformed1 = scripted_transforms(new_x)
        X_transformed2 = scripted_transforms(new_x)
        X_transformed3 = scripted_transforms(new_x)
        X_transformed4 = scripted_transforms(new_x)
        X_constrastive = torch.cat([X_transformed1, X_transformed2, X_transformed3, X_transformed4], dim=0)

    if no_grad:
        with torch.no_grad():
            _, out = model(normalize(X_constrastive))
    else:
        _, out = model(normalize(X_constrastive))

    # import pdb; pdb.set_trace()
    output = submodel(out)
    closs, acc = constrastive_loss_func(output, criterion, bs, n_views)

    return closs




def adaptive_attack_pgd(model, X, y, c_head_model, scripted_transforms, criterion, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, n_views=2, lambda_S=1):
    """Defense Aware Attack, where the attacker optimizes to both fool the classifier and decrease contrastive loss,
    So that our reverse algorithm cannot reverse the attack via decreasing the contrastive loss further."""
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon

        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss_classification = F.cross_entropy(output, y)
            loss_ada = -calculate_contrastive_Mhead_loss(X+delta, scripted_transforms, model, criterion,
                                                         c_head_model, no_grad=False, n_views=n_views)
            loss = loss_classification + loss_ada * lambda_S
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_BIM(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        # if norm == "l_inf":
        #     delta.uniform_(-epsilon, epsilon)
        # elif norm == "l_2":
        #     delta.normal_()
        #     d_flat = delta.view(delta.size(0),-1)
        #     n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        #     r = torch.zeros_like(n).uniform_(0, 1)
        #     delta *= r/n*epsilon
        # else:
        #     raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                # loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        elif norm == "l_1":
            pass
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                # loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            elif norm == "l_1":
                g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def attack_CW(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, num_class=10):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    batchsize=X.size(0)
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            label_mask = one_hot_embedding(y, num_class) # this works
            label_mask=label_mask.cuda()

            correct_logit = torch.sum(label_mask*output, dim=1)
            wrong_logit, _ = torch.max((1-label_mask)*output - 1e4*label_mask, axis=1)
            # select the seond best (but of course it is wrong)

            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

# TODO: bug here, even random noise decrease rob acc on medium model.

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='WideResNet')  #WideResNet
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--contrastive_bs', default=512, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lam_res', default=1, type=float)
    parser.add_argument('--adda_times', default=1, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--bd_attack_iters', default=4, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--neg_size', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2', 'l_1'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='train_ssl', type=str)
    import socket
    # if 'cv10' in socket.gethostname():
    #     parser.add_argument('--save_root_path', default='/local/vondrick/chengzhi/SSRobust', type=str)
    if 'cv' in socket.gethostname():
        parser.add_argument('--save_root_path', default='/proj/vondrick/mcz/SSRobust/Ours', type=str)
    else:
        parser.add_argument('--save_root_path', default='/local/rcs/mcz/2021Spring/SSRobdata/', type=str)

    parser.add_argument('--ssl_model_path', default='', type=str)
    parser.add_argument('--attack_type', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--constrastive_head', default=16, type=int)
    parser.add_argument('--md_path', default='', type=str)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--rand', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--MCtimes', default=1, type=int)
    parser.add_argument('--n_views', default=4, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--carmon', action='store_true')
    parser.add_argument('--TRADES', action='store_true')
    parser.add_argument('--Bag', action='store_true')
    parser.add_argument('--res18', action='store_true')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--random_noise', action='store_true')
    parser.add_argument('--foolbox', action='store_true')
    return parser.parse_args()


def main():
    adda_times=1

    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    args.fname = os.path.join(args.save_root_path, args.fname, timestamp + unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    dataset = cifar10(args.data_dir)

    train_set = list(zip(transpose(pad(dataset['train']['data'], 4) / 255.),
                         dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data'] / 255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18' or args.res18:
        from learning.preactresnet import PreActResNet18SSL, PreActResNet18
        model = PreActResNet18SSL()
        c_head_model = Res18_out6_model()

        if args.attack_type=='AA':
            ori_model = PreActResNet18()
            ori_model = nn.DataParallel(ori_model).cuda()

    elif args.model == 'WideResNet':
        # model = WideResNetMed_SSL(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        if args.carmon:
            from learning.unlabel_WRN import WideResNet_2
            model = WideResNet_2(depth=28, widen_factor=10)
        elif args.TRADES or args.Bag:
            from learning.unlabel_WRN import WideResNet_2
            model = WideResNet_2(depth=34, widen_factor=10)

        else:
            model = WideResNetMed_SSL(34, 10, widen_factor=args.width_factor, dropRate=0.0)
            # print(model)
        c_head_model = WRN34_out_branch()

    else:
        raise ValueError("Unknown model")

    if not args.TRADES and not args.Bag:
        model = nn.DataParallel(model).cuda()
    c_head_model = nn.DataParallel(c_head_model).cuda()
    c_head_model.train()


    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params_bkbone = [{'params': decay, 'weight_decay': args.l2},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        # params_bkbone = model.parameters()
        decay, no_decay = [], []
        for name, param in c_head_model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                         {'params': no_decay, 'weight_decay': 0}]


    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return args.lr_max
        elif t / args.epochs < 0.75:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    learning_rate=1e-4
    opt = torch.optim.Adam(params, lr=learning_rate)

    if args.md_path != '':
        # try:
        if args.TRADES or args.Bag:
            tmp=torch.load(args.md_path)
        else:
            tmp=torch.load(args.md_path)['state_dict']

        model.load_state_dict(tmp)
        if args.attack_type=='AA':
            ori_model.load_state_dict(tmp)

    if args.TRADES or args.Bag:
        model = nn.DataParallel(model).cuda()

    # defines transformation for SSL contrastive learning.
    s = 1
    size = 32
    from torchvision.transforms import transforms
    # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([color_jitter], p=0.8),
        transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        transforms.RandomGrayscale(p=0.2),
        # GaussianBlur(kernel_size=int(0.1 * size)),
    )
    scripted_transforms = torch.jit.script(transforms)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    contrastive_attack_loss = 0
    contrastive_clean_loss = 0

    # Train the SSL model first
    if not args.eval_only:
        flag=True
        for epoch in range(0, 201):
            model.eval()
            c_head_model.train()

            start_time = time.time()
            train_loss = 0.

            train_n=0
            for i, batch in enumerate(train_batches):
                X, y = batch['input'], batch['target']

                contrastive_Loss = \
                    calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, c_head_model)

                opt.zero_grad()
                contrastive_Loss.backward()
                opt.step()

                train_loss += contrastive_Loss.item() * y.size(0)
                train_n += y.size(0)

                if args.debug:
                    break

            logger.info('train loss:  %.4f   ', train_loss / train_n)
            # if epoch<20:
            #     continue

            if flag:
                TestX = []
                TestY = []
                Testdelta = []
                test_n = 0
                for i, batch in enumerate(test_batches):
                    if args.debug and i > 0:
                        break

                    X, y = batch['input'], batch['target']
                    TestX.append(X)
                    TestY.append(y)

                    # X_train_support = All_data[random.sample(list1, constrastive_bs)]

                    # Random initialization
                    if args.attack == 'none':
                        delta = torch.zeros_like(X)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                           early_stop=args.eval)
                    delta = delta.detach()
                    Testdelta.append(delta)

                    with torch.no_grad():
                        robust_output, _ = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                        output, _ = model(normalize(X))

                        robust_loss = criterion(robust_output, y)
                        loss = criterion(output, y)

                        Adv_image = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
                        # contrastive_attack = \
                        #     calculate_contrastive_Mhead_loss(Adv_image, scripted_transforms, model, criterion, c_head_model)
                        # contrastive_clean = \
                        #     calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, c_head_model)

                    torch.cuda.empty_cache()

                    # contrastive_attack_loss += contrastive_attack.item() * y.size(0)
                    # contrastive_clean_loss += contrastive_clean.item() * y.size(0)
                    test_robust_loss += robust_loss.item() * y.size(0)
                    test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                    test_loss += loss.item() * y.size(0)
                    test_acc += (output.max(1)[1] == y).sum().item()
                    test_n += y.size(0)
                    print('rob acc', test_robust_acc*1.0/test_n)

                flag=False
                # TODO: we can also try more optimization method, such as LBFGS, Gassian Prior, etc.

                TestX = torch.cat(TestX, dim=0)
                TestY = torch.cat(TestY, dim=0)
                Testdelta = torch.cat(Testdelta, dim=0)


            test_n = 0

            if epoch%args.eval_freq==0 or args.rand:
                c_head_model.eval()
                test_robust_ada_acc = 0
                test_clean_ada_acc = 0
                test_robust_ada_loss = 0
                test_clean_ada_loss = 0

                contrastive_attack_loss = 0
                contrastive_clean_loss = 0
                adaadv_contrastive_loss=0

                bs=512
                num_bs=TestX.size(0)//bs
                if num_bs*bs < TestX.size(0):
                    num_bs+=1
                count_test=0
                for bs_ind in range(num_bs):
                    if args.debug and bs_ind>0:
                        break
                    X = TestX[bs_ind*bs:(bs_ind+1)*bs]
                    y = TestY[bs_ind*bs:(bs_ind+1)*bs]
                    delta = Testdelta[bs_ind*bs:(bs_ind+1)*bs]

                    X = X.cuda()
                    y = y.cuda()
                    delta = delta.cuda()

                    # Need to calculate the contrastive here, i.e., as the training goes, because training change SSL contrastive branch model weights
                    Adv_image = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
                    contrastive_attack = \
                        calculate_contrastive_Mhead_loss(Adv_image, scripted_transforms, model, criterion, c_head_model)
                    contrastive_clean = \
                        calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, c_head_model)
                    contrastive_attack_loss += contrastive_attack.item() * y.size(0)
                    contrastive_clean_loss += contrastive_clean.item() * y.size(0)

                    # Our reversal vector
                    delta2 = attack_constrastive_Mhead(model, c_head_model, scripted_transforms, criterion,
                                                       torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit),
                                                       torch.zeros_like(y), epsilon * adda_times, pgd_alpha,  # 1, 0.2,
                                                       int(args.attack_iters * adda_times) if not args.rand else 0,
                                                       args.restarts, args.norm,
                                                       )
                    delta2 = delta2.detach()

                    robust_output_ada, hidden = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit) + delta2))
                    test_robust_ada_acc += (robust_output_ada.max(1)[1] == y).sum().item()
                    robust_ada_loss = criterion(robust_output_ada, y)
                    test_robust_ada_loss += robust_ada_loss.item() * y.size(0)

                    # calculate the SSL loss after our reversal
                    contrastive_ada_attack = \
                        calculate_contrastive_Mhead_loss(
                            torch.clamp(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit) + delta2, min=lower_limit, max=upper_limit),
                                                         scripted_transforms, model, criterion, c_head_model)
                    adaadv_contrastive_loss += contrastive_ada_attack.item() * y.size(0)

                    # The reversal vector for clean examples, since our algorithm applied reversal regardlessly.
                    delta3 = attack_constrastive_Mhead(model, c_head_model, scripted_transforms,
                                                       criterion,
                                                       X,
                                                       torch.zeros_like(y), epsilon * adda_times, pgd_alpha,  # 1, 0.2,
                                                       int(args.attack_iters * adda_times) if not args.rand else 0,
                                                       args.restarts, args.norm,
                                                       early_stop=args.eval)
                    #     #epsilon * args.adda_times, pgd_alpha
                    delta3 = delta3.detach()

                    # Clean accuracy after reversal, it drops a little due to the reversal.
                    clean_output_ada, hidden = model(
                        normalize(X + delta3))
                    test_clean_ada_acc += (clean_output_ada.max(1)[1] == y).sum().item()

                    clean_ada_loss = criterion(clean_output_ada, y)
                    test_clean_ada_loss += clean_ada_loss.item() * y.size(0)
                    test_n += y.size(0)
                    # print(test_robust_ada_acc, test_robust_ada_loss)

                    torch.cuda.empty_cache()
                    print(bs_ind)

                torch.save({
                    'ssl_model': c_head_model.state_dict(),
                }, os.path.join(args.fname, f'ssl_model_{epoch}.pth'))
                print(
                    'e=%d   \t TestLoss=%.4f TestAcc=%.4f TestCleanAdaAcc=%.4f \t TestRobLoss=%.4f TestRobAcc %.4f \t AdaTestLoss=%.4f AdaTestAcc %.4f' %
                    (epoch,
                     (test_loss / test_n), (test_acc / test_n * 100), (test_clean_ada_acc / test_n * 100),
                     (test_robust_loss / test_n), (test_robust_acc / test_n * 100),
                     (test_robust_loss / test_n), (test_robust_ada_acc / test_n * 100)))
                print('clean contrastive=%.6f \t adv contrastive=%.6f \t adaadv contrastive=%.6f' %
                      ((contrastive_clean_loss / test_n), (contrastive_attack_loss / test_n), (adaadv_contrastive_loss / test_n)))

    else:
        # SSL model has been trained, here we do the evaluation only without training.

        # Load the pretrained SSL model.
        if args.res18:
            # import pdb; pdb.set_trace()
            if args.new:
                tmp = torch.load(args.ssl_model_path)['ssl_model']
            else:
                tmp = torch.load(args.ssl_model_path)
        else:
            tmp = torch.load(args.ssl_model_path)['ssl_model']
        c_head_model.load_state_dict(tmp)
        c_head_model.eval()

        # We use this to allow scripted transforms to be differentiable. Need this due to Pytorch Issue.
        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']
            contrastive_Loss = \
                calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, c_head_model)
            break

        epsilon_list=[8]
        if args.norm=='l_2':
            epsilon_list=[128, 256, 256+128, 256+256, 512+128, 512+256]
            epsilon_list=[256]
        for epsilon in epsilon_list:  #
            lambda_S= 10
            # lambda S is the weight for defense aware attack, if not defense attack, put it to be 0.
            # only work for one Lambda_S now, attack will be at success at all if do a list, but is reasonable for single.

            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            db_rob_acc_all=0
            epsilon = epsilon / 255.
            TestX = []
            TestY = []
            Testdelta = []
            test_n = 0

            db_test_acc_clean_all=0

            print('epsilon', epsilon)

            contrastive_attack_loss=0
            contrastive_clean_loss=0
            # Standard Adversarial Attack Generation
            for i, batch in enumerate(test_batches):
                if args.debug and i > 0:
                    break

                X, y = batch['input'], batch['target']
                TestX.append(X)
                TestY.append(y)

                X = X.cuda()
                y = y.cuda()

                if args.attack_type == 'CW':
                    delta = attack_CW(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                       early_stop=args.eval)
                elif args.attack_type == 'adapt':
                    delta = adaptive_attack_pgd(model, X, y, c_head_model, scripted_transforms, criterion,
                                                epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                       early_stop=args.eval, lambda_S=lambda_S)
                elif args.attack_type == 'BIM':
                    delta = attack_BIM(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                       early_stop=args.eval)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                           early_stop=args.eval)
                delta = delta.detach()
                Testdelta.append(delta)

                with torch.no_grad():
                    robust_output, _ = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    output, _ = model(normalize(X))

                    robust_loss = criterion(robust_output, y)
                    loss = criterion(output, y)

                    Adv_image = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
                    contrastive_attack = \
                        calculate_contrastive_Mhead_loss(Adv_image, scripted_transforms, model, criterion, c_head_model, n_views=4)
                    contrastive_clean = \
                        calculate_contrastive_Mhead_loss(X, scripted_transforms, model, criterion, c_head_model, n_views=4)

                contrastive_attack_loss += contrastive_attack.item() * y.size(0)
                contrastive_clean_loss += contrastive_clean.item() * y.size(0)

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)
                torch.cuda.empty_cache()
                print("test_robust_acc", test_robust_acc/test_n, db_rob_acc_all/test_n, 'clean', db_test_acc_clean_all/test_n)

            print('clean contrastive=%.6f \t adv contrastive=%.6f' %
                                ((contrastive_clean_loss / test_n), (contrastive_attack_loss / test_n)))



            TestX = torch.cat(TestX, dim=0)
            TestY = torch.cat(TestY, dim=0)
            Testdelta = torch.cat(Testdelta, dim=0)

            total_len = TestX.shape[0]
            ind = [i for i in range(total_len)]
            from random import shuffle
            shuffle(ind)
            TestX = TestX[ind]
            TestY = TestY[ind]
            Testdelta = Testdelta[ind]

            bs = args.contrastive_bs

            num_bs = TestX.size(0) // bs
            if num_bs * bs < TestX.size(0):
                num_bs += 1
            count_test = 0


            # Start Reverse Attacks.
            Base_atack_steps = [20]
            for base_attack_step in Base_atack_steps: # 10, 15, 20, 10, 15, used to be 20
                for adda_times in [2]:
                    test_robust_ada_acc = 0
                    test_clean_ada_acc = 0
                    test_robust_ada_loss = 0
                    test_clean_ada_loss = 0
                    # test_clean_loss = 0
                    # test_clean_acc = 0

                    test_n = 0
                    adaadv_contrastive_loss=0

                    print('contrastive bs', bs)

                    for bs_ind in range(num_bs):
                        if args.debug and bs_ind > 0:
                            break
                        X = TestX[bs_ind * bs:(bs_ind + 1) * bs]
                        y = TestY[bs_ind * bs:(bs_ind + 1) * bs]
                        delta = Testdelta[bs_ind * bs:(bs_ind + 1) * bs]

                        X = X.cuda()
                        y = y.cuda()
                        delta = delta.cuda()

                        # Random initialization
                        if args.random_noise:
                            delta2 = torch.zeros_like(X)
                            delta2.uniform_(-epsilon* adda_times, epsilon* adda_times)
                        else:
                            # Reverse Attacks
                            delta2 = attack_constrastive_Mhead(model, c_head_model, scripted_transforms, criterion,
                                                               torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit),
                                                               torch.zeros_like(y), epsilon * adda_times, pgd_alpha,  # 1, 0.2,
                                                               int(base_attack_step * adda_times) if not args.rand else 0,
                                                               args.restarts, args.norm, n_views=args.n_views
                                                               )
                            delta2 = delta2.detach()

                        robust_output_ada, hidden = model(
                            normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit) + delta2))
                        test_robust_ada_acc += (robust_output_ada.max(1)[1] == y).sum().item()
                        robust_ada_loss = criterion(robust_output_ada, y)
                        test_robust_ada_loss += robust_ada_loss.item() * y.size(0)

                        torch.cuda.empty_cache()

                        # New SSL Loss after reversal
                        contrastive_ada_attack = \
                            calculate_contrastive_Mhead_loss(
                                torch.clamp(
                                    torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit) + delta2,
                                    min=lower_limit, max=upper_limit),
                                scripted_transforms, model, criterion, c_head_model)
                        adaadv_contrastive_loss += contrastive_ada_attack.item() * y.size(0)

                        torch.cuda.empty_cache()

                        # Reversal applied to clean examples.
                        if args.random_noise:
                            delta3 = delta2
                        else:
                            delta3 = attack_constrastive_Mhead(model, c_head_model, scripted_transforms,
                                                               criterion,
                                                               X,
                                                               torch.zeros_like(y), epsilon * adda_times, pgd_alpha,  # 1, 0.2,
                                                               int(base_attack_step * adda_times) if not args.rand else 0,
                                                               args.restarts, args.norm,
                                                               early_stop=args.eval, n_views=args.n_views)
                            #     #epsilon * args.adda_times, pgd_alpha
                            delta3 = delta3.detach()

                        clean_output_ada, hidden = model(
                            normalize(X + delta3))
                        test_clean_ada_acc += (clean_output_ada.max(1)[1] == y).sum().item()

                        clean_ada_loss = criterion(clean_output_ada, y)
                        test_clean_ada_loss += clean_ada_loss.item() * y.size(0)
                        test_n += y.size(0)
                        # print(test_robust_ada_acc, test_robust_ada_loss)

                        torch.cuda.empty_cache()
                        # print(bs_ind)

                    print('lambda S', lambda_S)
                    print(
                        'e=%d  scale=%.4f step=%d epsilon=%d \t TestLoss=%.4f TestAcc=%.4f TestCleanAdaAcc=%.4f \t TestRobLoss=%.4f TestRobAcc %.4f \t AdaTestLoss=%.4f AdaTestAcc %.4f' %
                        (0, adda_times, (int(base_attack_step * adda_times)), (epsilon*255),
                         (test_loss / test_n), (test_acc / test_n * 100), (test_clean_ada_acc / test_n * 100),
                         (test_robust_loss / test_n), (test_robust_acc / test_n * 100),
                         (test_robust_loss / test_n), (test_robust_ada_acc / test_n * 100)))
                    print('clean contrastive=%.6f \t adv contrastive=%.6f \t adaadv contrastive=%.6f' %
                          ((contrastive_clean_loss / test_n), (contrastive_attack_loss / test_n),
                           (adaadv_contrastive_loss / test_n)))
        print("\n\n\n")


if __name__ == "__main__":
    main()