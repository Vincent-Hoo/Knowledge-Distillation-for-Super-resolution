from matplotlib import pyplot as plt 
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
from option import args
import edsr, ddbpn, rdn, rcan
from loss import *
import utility
import math
from scipy.spatial.distance import cdist

def spatial_similarity(fm): 
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s


def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]


def visu(layer_activation, name):
    bc, nc, h, w = layer_activation.shape
    layer_activation = layer_activation.cpu().numpy()
    SA_map = layer_activation.squeeze()
    print(SA_map.shape)
    
    SA_map -= SA_map.mean()
    SA_map /= SA_map.std()
    SA_map *= 64
    SA_map += 128
    SA_map = np.clip(SA_map, 0, 255).astype('uint8')
    
    #plt.figure(SA_map)
    plt.grid(False)
    plt.imshow(SA_map[:75, :75], aspect='auto', cmap='viridis')
    plt.savefig('./{}.png'.format(name))

'''
def visu(layer_activation, name):
    bc, nc, h, w = layer_activation.shape
    layer_activation = layer_activation.cpu().numpy()
    print(layer_activation.shape)
    
    images_per_row = 16
    n_cols = nc // images_per_row
    display_grid = np.zeros((h * n_cols, images_per_row * w))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, col * images_per_row + row, :, :]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * h : (col + 1) * h, row * w : (row + 1) * w] = channel_image
        plt.figure(figsize=(1. / w * display_grid.shape[1], 1. / h * display_grid.shape[0]))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig('./{}.png'.format(name))
'''    

teacher = rcan.RCAN(args)
load_from = torch.load('../teacher_checkpoint/RCAN_BIX8.pt')
teacher.load_state_dict(load_from)

args.n_resblocks = 6
student = rcan.RCAN(args)
student_without_kd = rcan.RCAN(args)
student_with_kd = rcan.RCAN(args)
#load_from_without_kd = torch.load('../experiment_result/student_baseline/rcan_baseline/baseline_x4/model/model_best.pt')
#load_from_without_kd = torch.load('../experiment_result/overall_distilation/rcan/SA_x4_epoch200/model/model_best.pt')

print('network has been loaded successfully')

addr = ['../experiment_result/student_baseline/rcan_baseline/baseline_x8/model/model_best.pt',
        '../experiment_result/overall_distilation/rcan/SA_x8/model/model_best.pt']

name = ['baseline', 'SA']

loader = data.Data(args)
train_loader = loader.loader_train
test_loader = loader.loader_test
device = torch.device('cpu' if args.cpu else 'cuda')


flag = 1
for a, n in zip(addr, name):
    load_from = torch.load(a)
    student.load_state_dict_student(load_from)
    print(n)
    
    student = student.to(device)
    teacher = teacher.to(device)
    
    for p in teacher.parameters():
        p.requires_grad = False
        
    for p in student.parameters():
        p.requires_grad = False
    
    for idx_data, d in enumerate(test_loader):
        for idx_scale, scale in enumerate(args.scale):
            d.dataset.set_scale(idx_scale)
            cnt = 0
            for lr, hr, filename, _ in tqdm(d, ncols=80):
                lr, hr = prepare(lr, hr)
                teacher_fms, teacher_sr = teacher(lr)
                student_fms, student_sr = student(lr)
                
                student_fms = [spatial_similarity(fm) for fm in student_fms]
                teacher_fms = [spatial_similarity(fm) for fm in teacher_fms]
                
                if cnt == 2:
                    visu(teacher_fms[2], './visu/body_1/teacher')
                    visu(student_fms[2], './visu/body_1/{}'.format(n))
                    break
                cnt += 1
            break
        break
                
