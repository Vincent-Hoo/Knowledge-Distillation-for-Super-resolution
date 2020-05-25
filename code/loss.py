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
        self.feature_loss_module = nn.ModuleList()
        self.feature_loss_used = args.feature_loss_used
        
        # SR loss
        DS_weight = 1 - args.alpha
        TS_weight = args.alpha

        self.loss.append({'type': "DS", 'weight': DS_weight, 'function': nn.L1Loss()})
        self.loss.append({'type': "TS", 'weight': TS_weight, 'function': nn.L1Loss()})
          


        # feature loss
        if args.feature_loss_used == 1:      
            for loss in args.feature_distilation_type.split('+'):
                weight, feature_type = loss.split('*')
                l = {'type': feature_type, 'weight': float(weight), 'function': FeatureLoss(loss=nn.L1Loss())}
                self.loss.append(l)
                self.feature_loss_module.append(l['function'])
       
      
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
        
        
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.log = torch.Tensor()      
        self.feature_loss_module.to(device)
        
        if not args.cpu and args.n_GPUs > 1:
            self.feature_loss_module = nn.DataParallel(
                self.feature_loss_module, range(args.n_GPUs)
            )

        if args.resume == 1: 
            self.load(ckp.dir, cpu=args.cpu)



    def forward(self, student_sr, teacher_sr, hr, student_fms, teacher_fms):
        # DS Loss
        DS_loss = self.loss[0]['function'](student_sr, hr) * self.loss[0]['weight']
        self.log[-1, 0] += DS_loss.item()
        
        # TS Loss
        TS_loss = self.loss[1]['function'](student_sr, teacher_sr) * self.loss[1]['weight']
        self.log[-1, 1] += TS_loss.item()
        
        loss_sum = DS_loss + TS_loss

        
        if self.feature_loss_used == 0:
            pass
        elif self.feature_loss_used == 1:
            assert(len(student_fms) == len(teacher_fms))
            assert(len(student_fms) == len(self.feature_loss_module))
            
            for i in range(len(self.feature_loss_module)):   
                feature_loss = self.feature_loss_module[i](student_fms[i], teacher_fms[i])
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
            return self.feature_loss_module
        else:
            return self.feature_loss_module.module

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
        for l in self.feature_loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()





class FeatureLoss(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss



