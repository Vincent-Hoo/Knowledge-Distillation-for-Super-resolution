import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        self.label_loss_num = len(args.loss.split('+')) * 2
        self.feature_loss_used = args.feature_loss_used
        
        # SR loss
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function_srhr = nn.MSELoss()
                loss_function_srsr = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function_srhr = nn.L1Loss()
                loss_function_srsr = nn.L1Loss()

            self.loss.append({
                'type': loss_type + "srhr",
                'weight': float(weight),
                'function': loss_function_srhr}
            )
            
            self.loss.append({
                'type': loss_type + "srsr",
                'weight': 1 - float(weight),
                'function': loss_function_srsr}
            )
          
        if args.feature_loss_type == 'L1':
            feature_loss_type = nn.L1Loss()
        elif args.feature_loss_type == 'MSE':
            feature_loss_type = nn.MSELoss()
        
        
        
        # feature loss  
        if args.feature_loss_used == 1:      
            for loss in args.feature_distilation_type.split('+'):
                weight, feature_type = loss.split('*')
                l = {'type': feature_type, 'weight': float(weight), 'function': betweenLoss(eval(args.coef_sloss), loss=feature_loss_type)}
                self.loss.append(l)
                self.loss_module.append(l['function'])
           
            #self.loss.append({'type': 'betweenLoss', 'weight': 1, 'function': betweenLoss(eval(args.coef_sloss), loss=feature_loss_type)})        
        
        
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                #self.loss_module.append(l['function'])
        
        
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.log = torch.Tensor()      
        self.loss_module.to(device)
        
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.resume == 1: 
            self.load(ckp.dir, cpu=args.cpu)



    def forward(self, student_sr, teacher_sr, hr, student_fms, teacher_fms):
        label_loss = 0
        for i in range(self.label_loss_num):
            l = self.loss[i]
            if l['type'].endswith("srhr"):
                loss = l['function'](student_sr, hr)
                effective_loss = l['weight'] * loss
                label_loss += effective_loss
                self.log[-1, i] += effective_loss.item()
            elif l['type'].endswith("srsr"):
                loss = l['function'](student_sr, teacher_sr)
                effective_loss = l['weight'] * loss
                label_loss += effective_loss
                self.log[-1, i] += effective_loss.item()

        loss_sum = label_loss


        
        if self.feature_loss_used == 0:
            pass
        elif self.feature_loss_used == 1:
            assert(len(student_fms) == len(teacher_fms))
            assert(len(student_fms) == len(self.loss_module))
            
            for i in range(len(self.loss_module)):   
                feature_loss = self.loss_module[i](student_fms[i], teacher_fms[i])
                self.log[-1, 2 + i] += feature_loss.item()
                loss_sum += feature_loss
  
        self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()







def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()

def L1_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.l1_loss(softmax_outputs, softmax_targets)

def L2_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.mse_loss(softmax_outputs, softmax_targets)


class betweenLoss(nn.Module):
    def __init__(self, gamma=[1, 1, 1, 1, 1, 1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        tmp = [self.gamma[i] * self.loss(outputs[i], targets[i]) for i in range(length)]
        temp = [i.item() for i in tmp]
        #print(temp)
        res = sum(tmp)
        return res



class Weighted_Loss(nn.Module):
    def __init__(self, loss_type):
        super(Weighted_Loss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, outputs, targets, weight):
        loss_sum = 0
        for b in range(len(weight)):
            if self.loss_type == 'MSE':
                loss_sum += nn.MSELoss()(outputs[b], targets[b]) * weight[b]
            elif self.loss_type == 'L1':
                loss_sum += nn.L1Loss()(outputs[b], targets[b]) * weight[b]
        return loss_sum
