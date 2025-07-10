
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
from .utils import flatten



def get_optimizer(model, args):
    if args.optimizer.lower() == 'adam':
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
    return optimizer


def get_scheduler(optimizer, args, max_iterations):
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
    elif args.scheduler.lower() == 'poly':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (1 - step / max_iterations) ** 0.9)
    elif args.scheduler.lower() == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    elif args.scheduler.lower() == 'custom':
        scheduler = CustomLR(optimizer, T_first=5, lr_start_high=10*args.base_lr, lr_end_high=5*args.base_lr, 
                        lr_start_low=args.base_lr, lr_end_low=0, max_epochs=args.max_epochs)
    else:
        raise NotImplementedError(f"Scheduler <{args.scheduler}> not implemented")
    return scheduler


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
            inputs = torch.softmax(inputs, dim=1)
            target = self._one_hot_encoder(target)

            assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                      target.size())

            loss = 0.0
            for i in range(0, self.n_classes):
                loss += self._adaptive_size(inputs[:, i], target[:, i])
            return loss / self.n_classes


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth) / (union + self.smooth)

            losses.append(1 - IoU)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class Criterion(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        loss_type = args.loss_type.split(',')
        loss_weights = args.loss_weights.split(',')
        
        self.lnames, self.losses, self.weights = [], [], []
        for l, w in zip(loss_type, loss_weights):
            self.weights.append(float(w))
            self.lnames.append(l)
            if l == 'dice':
                self.losses.append(DiceLoss(num_classes))
            elif l == 'boundary':
                self.losses.append(BoundaryDoULoss(num_classes))
            elif l == 'ce':
                self.losses.append(CrossEntropyLoss())
            else:
                raise NotImplementedError(f"Loss {l} not implemented")    
    def forward(self, outputs, labels):
        loss = 0.0
        for w, loss_fn, l_name in zip(self.weights, self.losses, self.lnames):
            if l_name == 'ce':
                loss += w*loss_fn(outputs, labels[:].long())
            elif l_name == 'dice':
                loss += w*loss_fn(outputs, labels, softmax=True)
            elif l_name == 'boundary':
                loss += w*loss_fn(outputs, labels[:])
        return loss


class CustomLR(_LRScheduler):
    def __init__(self, optimizer, T_first=3, lr_start_high=0.01, lr_end_high=0.005, 
                 lr_start_low=0.001, lr_end_low=0.00001, max_epochs=50, **kwargs):
        self.T_first = T_first  # First phase epochs
        self.lr_start_high = lr_start_high
        self.lr_end_high = lr_end_high
        self.lr_start_low = lr_start_low
        self.lr_end_low = lr_end_low
        self.max_epochs = max_epochs
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        if self.last_epoch < self.T_first:
            # In the first T_first epochs, linearly decrease from lr_start_high to lr_end_high
            progress = self.last_epoch / self.T_first
            # print([self.lr_start_high - (self.lr_start_high - self.lr_end_high) * progress])
            return [self.lr_start_high - (self.lr_start_high - self.lr_end_high) * progress]
        else:
            # After T_first epochs, cosine annealing from lr_start_low to lr_end_low
            progress = (self.last_epoch - self.T_first) / (self.max_epochs - self.T_first)
            # print([self.lr_start_low - (self.lr_start_low - self.lr_end_low) * (0.5 * (1 - np.cos(np.pi * progress)))])
            return [self.lr_start_low - (self.lr_start_low - self.lr_end_low) * (0.5 * (1 - np.cos(np.pi * progress)))]
