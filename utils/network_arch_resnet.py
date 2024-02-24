from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import BaseArgs, ModelArgs, OneParamBN
from utils.model_units import LogitAdjuster, ETFClassifier, MyModel


# Reference: https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/models/resnet.py#L59
class ResNetBasicBlock(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d, bn1: nn.BatchNorm2d, bn2: nn.BatchNorm2d,
                 downsample: nn.Module = None, identitySkip: bool = False, bnOneParam: bool = False):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        if bnOneParam:
            bn1 = (OneParamBN(bn1.num_features) if downsample is None or not identitySkip else OneParamBN(bn1.num_features//2)) 
            bn2 = OneParamBN(bn2.num_features)
        
        self.bn1 = bn1 if downsample is None or not identitySkip else (OneParamBN if bnOneParam else nn.BatchNorm2d)(bn1.num_features//2)
        self.bn2 = bn2
        self.downsample = downsample
        self.identitySkip = identitySkip
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        identity = x
        
        if self.identitySkip:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if not self.identitySkip:
            out = self.relu(out)
        return out
        
        

class ResnetEncoder(MyModel):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster,  num_layers=18, isPretrained=False, isGrayscale=False,
                 embDimension=128, poolSize=4, outFeature=False, useETF = False, 
                 outAllFeatures=False, getResidual = False, useDRBalancer = False, imgNumPerCls = None,
                 for224: bool = False, useResNext: bool = False):
        assert args.model.dropout == 0, 'dropout is not supported'
        assert args.model.use_bn, "No support for not using bn"
        assert not args.model.add_skip, "No support for not adding skip connection"
        modelargs : ModelArgs = args.model
        super(ResnetEncoder, self).__init__()
        self.useResNext = useResNext
        if(self.useResNext):
            num_layers = 50
            
        self.path_to_model = '/tmp/models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        self.useETF = useETF
        self.getResidual = getResidual
        self.for224 = for224
        self.construct_encoder(args, num_layers, useResNext)
                    
        if self.embDimension>0:
            if self.useETF:
                self.encoder.fc =  ETFClassifier(self.num_ch_enc[-1], self.embDimension, modelargs.etf_norm_rate,
                                                 args.model.train_etf_norm, use_dr_balancer=useDRBalancer,
                                                 img_num_per_cls=imgNumPerCls)
            else:
                self.encoder.fc =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
                
        self.use_feature_bias = modelargs.use_feature_bias
        if self.use_feature_bias:
            self.feature_bias = torch.nn.Parameter(torch.zeros((1, self.num_ch_enc[-1]), dtype=torch.float32))
            
        self.__init_trained_params_list__(args)
        self.normalizeFeature = modelargs.normalize_feature

        self.logitAdjuster = logit_adjuster
        self.logitNormalization = modelargs.logit_normalization
        self.last_relu = modelargs.last_relu
        
        if self.for224:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.maxpool = lambda x: x
            self.avgpool = lambda x: F.avg_pool2d(x, self.poolSize)
    
    def construct_encoder(self, args: BaseArgs, num_layers:int, useResNext:bool):
        
        resnets = {
            18: models.resnet18, 
            34: models.resnet34,
            50: models.resnet50, 
            101: models.resnet101,
            152: models.resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))
            
        if useResNext:
            self.encoder = models.resnext50_32x4d(self.isPretrained)
            if args.model.identity_skip or args.model.one_param_bn:
                raise NotImplementedError
            if self.isPretrained:
                print("using pretrained model")
        else:
            self.encoder = resnets[num_layers]()
            if args.model.identity_skip or args.model.one_param_bn:
                self.reorder_layers(args)
            if self.isPretrained:
                print("using pretrained model")
                self.encoder.load_state_dict(
                    torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        self.encoder.conv1 = nn.Conv2d(1 if self.isGrayscale else 3, 64, kernel_size=7 if self.for224 else 3 ,
                                           stride=2 if self.for224 else 1,
                                           padding=3 if self.for224 else 1, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
        
    def reorder_layers(self, args: BaseArgs):
        assert not (args.model.one_param_bn and (args.model.modify_bn_weight or args.model.bn_eps != 1e-5)), "Not Implemented"
        if args.model.name != "ResNet34" and args.model.name != "IMResNet34":
            raise NotImplementedError
        encoder = self.encoder
        if args.model.one_param_bn:
            encoder.bn1 = OneParamBN(encoder.bn1.num_features)
        layersList = []
        for layer in [encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]:
            moduleList = []
            for module in layer.children():
                moduleList.append(
                    ResNetBasicBlock(module.conv1, module.conv2, module.bn1, module.bn2,
                                     module.downsample, args.model.identity_skip, args.model.one_param_bn)
                )
            layersList.append(nn.Sequential(*moduleList))
        encoder.layer1 = layersList[0]
        encoder.layer2 = layersList[1]
        encoder.layer3 = layersList[2]
        encoder.layer4 = layersList[3]
                
            
    # from pytorch_memlab import profile
    # @profile
    def forward(self, input_image):
        self.features = []
        if self.outAllFeatures:
            self.features.append(input_image)
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.maxpool(x)
        if self.outAllFeatures:
            self.features.append(x)
        
        x = self.encoder.layer1(x)
        if self.outAllFeatures:
            self.features.append(x)
        
        x = self.encoder.layer2(x)
        if self.outAllFeatures:
            self.features.append(x)
        
        torch.cuda.empty_cache()
        x = self.encoder.layer3(x) 
        if self.outAllFeatures:
            self.features.append(x)
        
        x = self.encoder.layer4(x)
        if self.outAllFeatures:
            self.features.append(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
                
        if self.normalizeFeature:
            x = x / (torch.linalg.norm(x, dim=1, keepdim = True) + 1e-10)
        
        if self.last_relu:
            x = torch.relu(x)
            
        if self.use_feature_bias:
            x = x + self.feature_bias
        
        if self.outFeature:
            if self.outAllFeatures:
                self.features.append(x)
                if self.getResidual:
                    # set nan for the features which has no residual.
                    self.features[-1] += torch.nan
                    self.features[-2] = self.features[-2] - self.encoder.layer4[0].downsample(self.features[-3])
                    self.features[-3] = self.features[-3] - self.encoder.layer3[0].downsample(self.features[-4])
                    self.features[-4] = self.features[-4] - self.encoder.layer2[0].downsample(self.features[-5])
                    self.features[-5] = self.features[-5] - self.features[-6]
                    self.features[0] += torch.nan
                    self.features[1] += torch.nan
                return self.features
            else:
                return x
        
        x = self.forward_last_layer(x)
        return x
    
    def get_fc_weight(self):
        if self.useETF:
            fc: ETFClassifier = self.encoder.fc
            return fc.get_weight()
        else:
            return self.state_dict()["encoder.fc.weight"]
        
    def set_fc_weight(self, tensor: torch.Tensor):
        if self.useETF:
            fc: ETFClassifier = self.encoder.fc
            fc.set_weight(tensor)
        else:
            self.encoder.fc.weight.data = tensor
            return self
        
    def get_fc_bias(self):
        return self.state_dict()["encoder.fc.bias"]
    
    def set_fc_bias(self, tensor: torch.Tensor):
        self.encoder.fc.bias.data = tensor
        return self
    
    def forward_last_layer(self, x: torch.Tensor):
        x = self.encoder.fc(x)
        if self.logitNormalization:
            x =  x / (torch.linalg.norm(x, dim=1, keepdim = True) + 1e-10)
        x = self.logitAdjuster(x)
            
        return x
    
    def get_fc_params_list(self):
        ans = [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
        return ans
    
    def __init_trained_params_list__(self, args: BaseArgs):
        if not self.useETF:
            self.trained_params_list = super().get_trained_params_list(args) 
            self.trained_params_names_list = super().get_trained_params_names_list(args)
            return 
        
        # useETF
        # assert args.optim.train_bn_bias and args.optim.train_bn_weight, " Not supported."
        self.trained_params_list = []
        self.trained_params_names_list = []
        for name, param in self.named_parameters():
            if self.belongs_to_fc_layers(name):
                if "bias" in name or ("norm" in name and args.model.train_etf_norm):
                    self.trained_params_list.append(param) 
                    self.trained_params_names_list.append(name)
                    
            elif args.project_name == "first":
                if (args.optim.train_bn_bias or not self.belongs_to_bn_bias(name)) and \
                        (args.optim.train_bn_weight or not self.belongs_to_bn_weight(name)):
                    self.trained_params_list.append(param) 
                    self.trained_params_names_list.append(name)
               
    def get_trained_params_list(self, args: BaseArgs):
        return self.trained_params_list
    
    def get_trained_params_names_list(self, args: BaseArgs):
        return self.trained_params_names_list
    
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        target_list = args.optim.weight_decay_target_modules
        if(name.startswith("encoder.layer")):
            return int(name[13]) in target_list
        elif name == "encoder.fc":
            return 5 in target_list
        else:
            return 0 in target_list
        
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError
                
            
    
    @classmethod
    def belongs_to_fc_layers(cls, name: str):
        return name.startswith("encoder.fc.")
    
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        return name.endswith(".bias") and (not name.endswith("fc.bias")) or name.endswith(".bias_param")
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        return name.endswith(".weight") and (not name.endswith("fc.weight")) and \
            (not name.endswith("conv1.weight")) and (not name.endswith("conv2.weight")) or name.endswith(".weight_param")
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        return name.startswith("encoder.bn") or ("downsample" in name)
            
    def get_features_n(self):
        return 6

