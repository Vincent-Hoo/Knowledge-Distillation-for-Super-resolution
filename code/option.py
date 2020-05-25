import argparse
import template

parser = argparse.ArgumentParser(description='Knowledge distillation for super-resolutioin')

# GPU setting
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')


# Dataset Setting
parser.add_argument('--dir_data', type=str, default='/data/dataset/dataset/sr_dataset', help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5+Set14+B100+Urban100', help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810', help='train/test data range')
parser.add_argument('--scale', type=str, default='4', help='super resolution scale')
parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')


# data argmentation setting
parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')
parser.add_argument('--patch_size', type=int, default=192, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')


# storage setting
parser.add_argument('--ckp_dir', type=str, default='first try', help='checkpoint directory')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--save_gt', action='store_true', help='save low-resolution and high-resolution images together')
parser.add_argument('--save_results', action='store_true', help='save output results')


# training setting
parser.add_argument('--epochs', type=int, default=700, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--resume', type=int, default=0, help='whether resume from the lastest version')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')
parser.add_argument('--print_every', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--template', default='.', help='You can set various templates in option.py') 


# loss function and optimizer setting
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--decay', type=str, default='150-300-450-600', help='learning rate decay type')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')


# distilation setting
#parser.add_argument('--loss', type=str, default='1*L1', help='DS and TS loss, 1 for DS')
parser.add_argument('--alpha', type=float, default=0.5, help='TS loss coefficient')
parser.add_argument('--model', default='RCAN', help='student model name')
parser.add_argument('--feature_distilation_type', default="1*SA", type=str, help='feature distilation type')
parser.add_argument('--feature_loss_used', default=1, type=int, help='whether to use feature loss')
parser.add_argument('--features', default="[1,2,3]", type=str, help='features selected')
parser.add_argument('--teacher', default="[RCAN]", type=str, help='teachers selected')
parser.add_argument('--student_n_resblocks', type=int, default=6, help='number of residual blocks')



# RDN hyper-parameters
parser.add_argument('--G0', type=int, default=64, help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3, help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B', help='parameters config of RDN. (Use in RDN)')
                    

# RCAN hyper-parameters
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')


# EDSR heyper-parameter
#parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

# Testing
parser.add_argument('--ckp_path', type=str, default='', help='checkpoint path')
parser.add_argument('--TS', default="S", type=str, help='test teacher or student')


args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')
