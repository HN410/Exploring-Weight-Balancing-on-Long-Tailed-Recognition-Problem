import numpy as np
import torch
import torch.nn as nn
import math

from utils.utils import *
from utils.model_units import MyModel
from utils.datasets import get_few_threshold, get_many_threshold
# The classes below wrap core functions to impose weight regurlarization constraints in training or finetuning a network.
            
class MaxNorm_via_PGD():
    # learning a max-norm constrainted network via projected gradient descent (PGD) 
    def __init__(self, thresh=1.0, LpNorm=2, tau = 1):
        self.thresh = thresh
        self.LpNorm = LpNorm
        self.tau = tau
        self.perLayerThresh = []
        
    def setPerLayerThresh(self, model: MyModel):
        # set per-layer thresholds
        self.perLayerThresh = []
        
        for curLayerGetter in [model.get_fc_weight, model.get_fc_bias]: #here we only apply MaxNorm over the last two layers
            
            curparam = curLayerGetter().data.detach()
            if len(curparam.shape)<=1: 
                self.perLayerThresh.append(float('inf'))
                continue
            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1).detach().unsqueeze(-1)
            curLayerThresh = neuronNorm_curparam.min() + self.thresh*(neuronNorm_curparam.max() - neuronNorm_curparam.min())
            self.perLayerThresh.append(curLayerThresh)
                
    def PGD(self, model: MyModel):
        if len(self.perLayerThresh)==0:
            self.setPerLayerThresh(model)
        
        for i, curLayerSet in enumerate([(model.get_fc_weight, model.set_fc_weight),
                                         (model.get_fc_bias, model.set_fc_bias)]): #here we only apply MaxNorm over the last two layers
            curparam = curLayerSet[0]().data.detach()


            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)    
            curLayerThresh = self.perLayerThresh[i]
            
            idx = neuronNorm_curparam > curLayerThresh
            idx = idx.squeeze()
            tmp = curLayerThresh / (neuronNorm_curparam[idx].squeeze())**(self.tau)
            for _ in range(len(scalingVect.shape)-1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx] 
            curLayerSet[1](curparam)

class Normalizer(): 
    def __init__(self, LpNorm=2, tau = 1, targetNorm = 1.0, normRate = 1.0):
        self.LpNorm = LpNorm
        self.tau = tau
        self.targetNorm = targetNorm
        self.normRate = normRate
  
    def apply_on(self, model: MyModel): #this method applies tau-normalization on the classifier layer
        curparam = model.get_fc_weight().data.detach()

        curparam_vec = curparam.reshape((curparam.shape[0], -1))
        neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
        normTensor = torch.ones((curparam.shape[0], 1)) -\
                    (self.normRate - 1)/((curparam.shape[0]-1) * self.normRate) *  torch.arange(curparam.shape[0]).unsqueeze(dim=1)
        scalingVect = (normTensor * self.targetNorm).to(curparam.device)
        
        idx = neuronNorm_curparam == neuronNorm_curparam
        idx = idx.squeeze()
        tmp = 1 / (neuronNorm_curparam[idx].squeeze())
        for _ in range(len(scalingVect.shape)-1):
            tmp = tmp.unsqueeze(-1)

        scalingVect[idx] = torch.mul(scalingVect[idx], tmp) 
        curparam[idx] = scalingVect[idx] * curparam[idx]
        model.set_fc_weight(curparam)

def set_norm(model: MyModel, args: BaseArgs):
    norm_type = args.optim.norm_type
    norm_param = args.optim.norm_param if "norm_param" in vars(args.optim) else None
    pgd_func = None
    if norm_type == "L2":
        norm_param = 1.0 if norm_param is None else norm_param
        L2_norm = Normalizer(tau=1, targetNorm=1.0, normRate = norm_param) # tau=1
        applier = L2_norm.apply_on
    elif norm_type == "tau":
        norm_param = 1.9 if norm_param is None else norm_param
        tau_norm = Normalizer(tau=norm_param) # tau=1.9
        applier = tau_norm.apply_on
    elif norm_type == "maxnorm":
        norm_param = 0.1 if norm_param is None else norm_param
        pgd_func = MaxNorm_via_PGD(thresh=norm_param) # 0.1
        applier = pgd_func.setPerLayerThresh # set per-layer thresholds
    elif norm_type == "none":
        return None
    else:
        assert False, "Invalid norm type."
    applier(model)
    return pgd_func

# get labels whoose weights are decayed
def calc_decayed_labels(img_num_per_cls: list, args: BaseArgs):
    if "weight_decay_target" not in vars(args.optim):
        weight_decay_target = [True]*3
    else:
        weight_decay_target = args.optim.weight_decay_target
    if args.model.use_etf and (not args.model.train_etf_norm):
        weight_decay_target = [False]*3
    img_num_per_cls = np.array(img_num_per_cls)
    many_labels = img_num_per_cls > get_many_threshold(args)
    medium_labels = np.logical_and(get_few_threshold(args) <= img_num_per_cls,  img_num_per_cls<= get_many_threshold(args))
    few_labels =  img_num_per_cls < get_few_threshold(args)
    used_fc_labels = np.stack([many_labels, medium_labels, few_labels])[weight_decay_target].sum(axis = 0) > 0
    return used_fc_labels

# pick up appropriate weights and calc the norm
def get_weight_norm_calculator(img_num_per_cls: list, args: BaseArgs):
    if "weight_decay_target" not in vars(args.optim):
        def calc_wn(name, weights, model: MyModel):
            # if name != fc_bias_name:
                return torch.norm(weights, 2) ** 2 / 2
            # else:
            #     return torch.zeros(0, device = args.device)
        return calc_wn
    
    used_fc_labels = calc_decayed_labels(img_num_per_cls, args)
    def calc_wn(name, weights, model: MyModel):
        # if name == fc_bias_name:
        #     return torch.zeros(0, device = args.device)
        if not model.belongs_to_fc_layers(name):
            target_weights = weights
        else:
            target_weights = weights[used_fc_labels]
        return torch.norm(target_weights, 2) ** 2 / 2
    return calc_wn
    


def set_weight_decay(loss_func, img_num_per_cls, args: BaseArgs):
    if args.optim.weight_decay == 0 or args.optim.optimizer == "AdamW":
        return loss_func
    else:
        calc_weights_norm = get_weight_norm_calculator(img_num_per_cls, args)
        def new_loss(logits: torch.tensor, label_list: torch.tensor, model: MyModel, features: torch.Tensor = None, epoch: int = 0):
            loss = loss_func(logits, label_list, model,  features, epoch)
            if args.optim.weight_decay_conv and args.optim.weight_decay_bn and not args.optim.weight_decay_limit and args.optim.weight_decay_identity_bn:
                norm = torch.stack([calc_weights_norm(name, weights, model).sum() 
                                    for name, weights in zip(model.get_trained_params_names_list(args), model.get_trained_params_list(args))
                                    ]).sum()
            else:
                norm_list = []
                for name, module in model.named_modules():
                    if args.optim.weight_decay_limit and (not model.check_valid_wd_target(name, args)):
                        continue
                    if args.optim.weight_decay_conv and isinstance(module, torch.nn.modules.conv.Conv2d):
                        norm_list.append(calc_weights_norm(name, model.get_parameter(name + ".weight"), model).sum())
                    elif args.optim.weight_decay_bn and is_batchnorm(module) and \
                            (args.optim.weight_decay_identity_bn or not model.belongs_to_identity_bn(name)):
                        if args.optim.train_bn_weight:
                            norm_list.append(calc_weights_norm(name, model.get_parameter(name + ".weight"), model).sum())
                        if args.optim.train_bn_bias:
                            norm_list.append(calc_weights_norm(name, model.get_parameter(name + ".bias"), model).sum())
                    elif isinstance(module, torch.nn.Linear):
                        norm_list.append(calc_weights_norm(name, model.get_parameter(name + ".weight"), model).sum())
                        if module.bias is not None:
                            norm_list.append(calc_weights_norm(name, model.get_parameter(name + ".bias"), model).sum())
                norm = torch.stack(norm_list).sum()
            return loss + args.optim.weight_decay * norm 
    return new_loss
            

def set_features_regularization(loss_func, args: BaseArgs):
    if args.optim.features_regularization == 0.0:
        return loss_func
    def new_loss(logits: torch.tensor, label_list: torch.tensor, model: MyModel, features: torch.Tensor = None, epoch: int = 0):
        loss = loss_func(logits, label_list, model,  features, epoch)
        loss = loss + args.optim.features_regularization * (torch.norm(features, 2, dim = 1) ** 2).mean() / 2
        return loss 
    return new_loss   