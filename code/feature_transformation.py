import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def spatial_similarity(fm): 
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm): 
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 1e-8)
    s = norm_fm.bmm(norm_fm.transpose(1,2))
    s = s.unsqueeze(1)
    return s

def batch_similarity(fm):
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0,1))
    normalized_Q = Q / torch.norm(Q,2,dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q


def FSP(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
    
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)
    fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    
    return fsp


def AT(fm):
    eps = 1e-6
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2,3), keepdim=True)
    am = torch.div(am, norm+eps)
    return am


def pooled_spatial_similarity(fm, k, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    s = spatial_similarity(fm)
    return s


def gaussian_rbf(fm, k, P, gamma, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    fm = fm.view(fm.size(0), fm.size(1), -1)
    feat = F.normalize(fm, p=2, dim=1)
    sim_mat = torch.bmm(feat.transpose(1,2), feat)

    corr_mat = torch.zeros_like(sim_mat)
    one = torch.ones_like(sim_mat)
    corr_mat += math.exp(-2*gamma) * (2*gamma)**0 / \
                    math.factorial(0) * one
    for p in range(1, P+1):
        corr_mat += math.exp(-2*gamma) * (2*gamma)**p / \
                    math.factorial(p) * torch.pow(sim_mat, p)
    return corr_mat



def MMD(fm, k, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    fm = fm.view(fm.size(0), fm.size(1), -1)
    mean_fm = torch.mean(fm, dim=1)
    num = mean_fm.shape[1]
    a = mean_fm.unsqueeze(-1).repeat(1,1,num)
    b = mean_fm.unsqueeze(1).repeat(1,num,1)
    mmd = torch.abs(a - b)
    return mmd
