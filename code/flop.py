import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from option import args
import edsr, rcan

with torch.cuda.device(0):
    #net = models.densenet161()
    #args.n_resblocks = 6
    #net = rcan.RCAN(args)
    args.n_resblocks = 16
    args.n_feats = 64
    args.res_scale = 1.0
    net = edsr.EDSR(args)
    flops, params = get_model_complexity_info(net, (3, 48, 48), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)