import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
from models.predesigned_modules import resnet12
import sys
import os
from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from models.util.pos_embed import interpolate_pos_embed
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# fix seed
np.random.seed(1)
torch.manual_seed(1)
import tqdm
from torch.nn.parallel import DataParallel
# torch.backends.cudnn.benchmark = True
from models.models_mae import mae_vit_base_patch16
# from sklearn import svm     #导入算法模块
import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from models import models_vit_fsl

import scipy as sp
import scipy.stats
#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=224, type=int, choices=[32, 84, 112,224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub','fc100','fs'])
parser.add_argument('--data_path', default='/home/jiangweihao/data/mini-imagenet',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
parser.add_argument('--test_n_episode', default=1000, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=4, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--print_freq', default=10, type=int, help='total number of inner frequency')

parser.add_argument('--momentum', default=0.9, type=int, help='parameter of optimization')
parser.add_argument('--weight_decay', default=5.e-4, type=int, help='parameter of optimization')

parser.add_argument('--gpu', default='4')
parser.add_argument('--epochs', default=100)

parser.add_argument('--mlp', action='store_true')
parser.add_argument('--ft', action='store_true')

params = parser.parse_args()
params.ft = True 
# 设置日志记录路径
log_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_path,'save/{}_test_task-{}_shot-{}_mae[{}]_image_complement_mlp[{}]_ft[{}]_encoder'.format(
                                params.dataset,params.test_n_episode,params.n_shot,params.model,params.mlp,params.ft))
ensure_path(log_path)
set_log_path(log_path)
log('log and pth save path:  %s'%(log_path))
# log(params)

# -------------设置GPU--------------------
set_gpu(params.gpu)
# -------------导入数据--------------------
test_file = 'test'
json_file_read = False
if params.dataset == 'mini_imagenet':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 64
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    test_file = 'novel.json'
    json_file_read = True
    params.num_classes = 200
    params.data_path = '/home/jiangweihao/data/CUB_200_2011'
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 351
    params.data_path = '/home/jiangweihao/data/tiered_imagenet'
elif params.dataset == 'fc100':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 64
    # params.image_size = 112
    params.data_path = '/home/jiangweihao/data/FC100'
elif params.dataset == 'fs':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 60
    # params.image_size = 112
    params.data_path = '/home/jiangweihao/data/cifar100'
else:
    ValueError('dataset error')

log(params)
#------------ test data ------------------------
# test_file = 'test'
# json_file_read = False
test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
test_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read, **test_few_shot_params)
test_loader = test_datamgr.get_data_loader(test_file, aug=False)

#   ------查看导入的数据----------
# target, label = next(iter(base_loader))
# print(len(base_loader))
# print(target.size())
# print(label.size())
# print('--------------------')

# ----------- 导入模型 -------------------------
# model = mae_vit_base_patch16()
# state_dict = torch.load('/home/jiangweihao/CodeLab/PytorchCode/mae_fsl/checkpoint/mae_pretrain_vit_base.pth')
# state_dict = state_dict['model']
# model.load_state_dict(state_dict,strict=False)  # 
# model.cuda()

# # from torchinfo import summary
# # summary(model,[5,3,224,224])


def cache_model(support,query,model,mask_ratio=[0, 0.25, 0.5, 0.75],modal='mean'):
    
    with torch.no_grad():
        # Data augmentation for the cache model
        for i, mask in enumerate(mask_ratio):
            
            support_f_m, _, _ = model.forward_encoder(support,mask_ratio=mask)
            query_f_m, _, _ = model.forward_encoder(query,mask_ratio=mask)
            support_cls_token_m = support_f_m[:,0,:]                # 把cls_token分离出来
            query_cls_token_m = query_f_m[:,0,:]
            if modal == 'mean':
                support_f_m = support_f_m[:,1:,:].mean(dim=1,keepdim=True)
                query_f_m = query_f_m[:,1:,:].mean(dim=1,keepdim=True)
            else:
                support_f_m = support_f_m[:,1:,:]
                query_f_m = query_f_m[:,1:,:]
            if i==0:
                support_f = support_f_m 
                query_f = query_f_m 
                support_cls_token = support_cls_token_m
                query_cls_token = query_cls_token_m
            else:
                support_f = torch.cat((support_f,support_f_m),1)
                query_f = torch.cat((query_f,query_f_m),1) 
                support_cls_token = torch.cat((support_cls_token,support_cls_token_m),1)
                query_cls_token = torch.cat((query_cls_token,query_cls_token_m),1) 

    if modal == 'mean':
        support_f = support_f.mean(dim=1).squeeze(1)   
        query_f = query_f.mean(dim=1).squeeze(1) 


    # support_cls_token = support_cls_token.mean(dim=1)  
    # query_cls_token = query_cls_token.mean(dim=1) 

    # 归一化
    # support_f_m = support_f.mean(dim=-1, keepdim=True)
    # support_f = support_f - support_f_m
    support_f /= support_f.norm(dim=-1, keepdim=True)
    support_cls_token /= support_cls_token.norm(dim=-1, keepdim=True)
    # query_f_m = query_f.mean(dim=-1, keepdim=True)
    # query_f = query_f - query_f_m
    query_f /= query_f.norm(dim=-1, keepdim=True)
    query_cls_token /= query_cls_token.norm(dim=-1, keepdim=True)

    return support_f, support_cls_token, query_f, query_cls_token

def catch_feature(query, model, mask_ratio=0):

    with torch.no_grad():    

        feature, _, _ = model.forward_encoder(query,mask_ratio=mask_ratio)

    return feature[:,0,:],feature[:,1:,:]

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

def mean_confidence_interval(data, confidence=0.95):
	a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m,h
    
def test(test_loader,params,model,epoch_index,best_prec1,loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
  

	# switch to evaluate mode
    model.eval()
    accuracies = []


    end = time.time()
    for episode_index, (temp2,target) in enumerate(test_loader):   
    # temp2, _ =next(iter(train_loader))

        # support,query = temp2.split([params.n_shot,params.n_query],dim=1)
        cache_values, q_values = target.split([params.n_shot,params.n_query],dim=1)

        # cache_values = F.one_hot(cache_values).half()
        cache_values = cache_values.reshape(-1)
        q_values = q_values.reshape(-1)
        cache_values, q_values = cache_values.cuda(), q_values.cuda()

        n,k,c,h,w = temp2.shape
        # support = support.reshape(-1,c,h,w)
        # support = support.cuda()
        # query = query.reshape(-1,c,h,w)
        # query = query.cuda()

        # ---------图像组合--------------
        
        #--------方法2：将support取50%，query填充其掩码部分，互补拼接-----------
        # query_patch = patchify(query)          # torch.Size([75, 196, 768])
        # support_patch = patchify(support)  
        # imags = random_compose(query_patch,support_patch)

        # imags = imags.reshape(-1,imags.shape[2],imags.shape[3])
        # imags = unpatchify(imags)
        # # print(imags.shape)
        # label = torch.eq(q_values.unsqueeze(1).repeat(1,params.val_n_way*params.n_shot),cache_values.unsqueeze(0).repeat(params.val_n_way*params.n_query,1)).type(torch.float32)
        # label = label.reshape(-1)
        label = np.repeat(range(params.val_n_way),params.n_query)
        label = torch.from_numpy(np.array(label))
        label = label.cuda()
        
        imags = temp2.reshape(-1,c,h,w).cuda()
        with torch.no_grad():
            outputs = model(imags)

        # outputs = F.sigmoid(outputs).reshape(-1)
        # loss = loss_fn(outputs,label)
        qf = outputs[0]                     #[125,768]
        p,d = qf.shape

        sf = outputs[1]     #[5,768]
        
        outputs = compute_logits(qf, sf, metric='cos')
        outputs = outputs.reshape(params.val_n_way*params.n_query,params.train_n_way,params.train_n_way,params.val_n_way*params.n_query).sum(-1).sum(1)
        loss = loss_fn(outputs,label)

        # pred = outputs.reshape(-1,params.val_n_way*params.n_shot).data.max(1)[1]
        # pred = outputs.reshape(-1,params.val_n_way,params.n_shot).sum(-1).data.max(1)[1]
        pred = outputs.data.max(1)[1]
        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()

        num = params.val_n_way*params.n_query
        pred = pred.eq(y).sum()/num
        
        losses.update(loss.item(), label.size(0))
        top1.update(pred, num)
        accuracies.append(pred)
        
        best_prec1 = max(best_prec1, top1.val)

		# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #============== print the intermediate results ==============#
        if episode_index % params.print_freq == 0 and episode_index != 0:

            log('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
					epoch_index, episode_index, len(test_loader), batch_time=batch_time, loss=losses, top1=top1))
	
        # log(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))

    return top1.avg, accuracies

    


def main():

    model = models_vit_fsl.__dict__[params.model](
        num_classes=0,
        global_pool=params.global_pool,
        img_size = params.image_size
    )


    # checkpoint = torch.load('/home/jiangweihao/CodeLab/PytorchCode/MAE_fsl/save/mini_imagenet_train_task-600_shot-5_mae_image_compolement/model_best.pth.tar')
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

    if params.dataset == 'mini_imagenet':    #----------------------------------------------------mini_imagenet'
        ckp = '/home/jiangweihao/code/MAE_fsl/save/mini_imagenet_train_task-600_shot-{}_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.001_adam-3-cls_100/model_best.pth.tar'.format(params.n_shot)
        # checkpoint = torch.load(ckp)
        # ckp = '/home/jiangweihao/code/MAE_fsl/mae_pretrain_vit_base.pth'       #  no fine-tune
        checkpoint = torch.load(ckp)
    if params.dataset == 'cub':    #----------------------------------------------------cub
        # ckp = '/home/jiangweihao/code/MAE_fsl/save/cub_train_task-600_shot-{}_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.001_adam-2-cls-100/model_best.pth.tar'.format(params.n_shot)
        ckp = '/home/jiangweihao/code/MAE_fsl/save/mini_imagenet_train_task-600_shot-{}_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.001_adam-2-cls-100/model_best.pth.tar'.format(params.n_shot)
        checkpoint = torch.load(ckp)
    elif params.dataset == 'tiered_imagenet':    #-----------------------------------------------------------------------------'tiered_imagenet'
        ckp = '/home/jiangweihao/code/MAE_fsl/save/tiered_imagenet_train_task-600_shot-{}_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.0001_adam-2-cls-100/model_best.pth.tar'.format(params.n_shot)
        checkpoint = torch.load(ckp)
    elif params.dataset == 'fc100':    #--------------------------------------------------------------------------------------------'fc100'
        ckp = '/home/jiangweihao/code/MAE_fsl/save/fc100_train_task-600_shot-1_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.001_adam-2-cls-100-2023-07-01_17-34-10/epoch_88.pth.tar'.format(params.n_shot)
        checkpoint = torch.load(ckp)
    elif params.dataset == 'fs':    #-------------------------------------------------------------------------------------------------'fs'
        ckp = '/home/jiangweihao/code/MAE_fsl/save/fs_train_task-600_shot-{}_mae[vit_base_patch16]_image_compolement_FT[True]_global[False]_encoder_lr0.001_adam-2-cls-100/model_best.pth.tar'.format(params.n_shot)
        checkpoint = torch.load(ckp)
    
    if params.model == 'vit_large_patch16':   
        checkpoint = torch.load('/home/jiangweihao/code/MAE_fsl/save/mini_imagenet_train_task-600_shot-5_mae[vit_large_patch16]_image_compolement_FT[True]onelayer_global[True]/model_best.pth.tar')
    log(ckp)
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint['state_dict'])

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.load_state_dict(checkpoint['model'], strict=True)         # no fine-tune
    model.to('cuda')
    # ---------------------------------------------
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 1
    log('==========start testing ===============')
    
    # loss_all = []
    # pred_all = []
    best_prec1 = 0
    for epoch in range(epochs): 
        
        prec1, accuracies = test(test_loader,params,model,epoch,best_prec1,loss_fn)

        acc, h = mean_confidence_interval(accuracies)
        log('prec1:{:.4f}'.format(prec1))
        log('vag acc:{:.4f}'.format(acc))
        log('confidence_interval:{:.4f}:'.format(h))

    


if __name__ == '__main__':

    start = time.time()
    main()
    log('total test time(s):')
    log(time.time()-start)
    log('===========================testing end!===================================')

