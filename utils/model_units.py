import torch
from torch import nn
import numpy as np
from scipy.stats import ortho_group
from abc import ABCMeta, abstractmethod, abstractclassmethod
import os
from utils.utils import BaseArgs, ModelArgs, ARGS_FILE_NAME, BEST_PARAM_FILE_NAME

class LogitAdjuster(nn.Module):
    def __init__(self, tau: float= 0, img_num_per_cls: list = None, for_training: bool = False, use_additive: bool = True):
        super().__init__()
        self.tau = tau
        if tau > 0:
            assert img_num_per_cls is not None, "img_num_per_cls should not be None when tau > 0"
            img_num_per_cls = np.array(img_num_per_cls)
            pY = img_num_per_cls / img_num_per_cls.sum()
            pY = torch.log(torch.Tensor(pY)).unsqueeze(0)
            self.register_buffer("pY", pY, persistent=False)
            self.for_training = for_training
            self.use_additive = use_additive
            
    
    def forward(self, x):
        if self.tau > 0:
            if self.use_additive:
                if not self.for_training and not self.training:
                    return x - self.tau * self.pY
                elif self.for_training and self.training:
                    return x + self.tau * self.pY
            else:
                return x * torch.exp(-1 * self.tau * self.pY)
        return x


def get_standard_etf(n_classes, device):
    ans = torch.eye(n_classes) - torch.ones((n_classes, n_classes)) / n_classes
    ans = ans * np.sqrt(n_classes / (n_classes - 1))
    return ans.to(device = device)
def get_general_etf(n_classes, n_features, device):
    ans = get_standard_etf(n_classes, device)
    ans = torch.matmul(
        torch.tensor(ortho_group.rvs(n_features)[:, :n_classes]).to(dtype=torch.float32, device=device), ans)
    return ans.T

class ETFClassifier(nn.Module):
    def __init__(self, in_features, out_features, norm_rate: float = 1, train_norm = True,
                 use_dr_balancer = False, img_num_per_cls: list = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight_vec = get_general_etf(out_features, in_features, "cpu")
        self.weight_vec = nn.Parameter(weight_vec)
        self.norm_rate = norm_rate
        norm_tensor = torch.ones((out_features, 1)) -\
                    (norm_rate - 1)/((out_features-1) * norm_rate) *  torch.arange(out_features).unsqueeze(dim=1)
        if use_dr_balancer:
            N = sum(img_num_per_cls)
            K = len(img_num_per_cls)
            norm_tensor = N / K * norm_tensor / (torch.tensor(img_num_per_cls)[:, None].to(torch.float32))
        self.weight_norm = nn.Parameter(norm_tensor.to(torch.float32))
        

        bias = torch.empty(out_features).uniform_(-1 / in_features, 1 / in_features)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        
        return torch.matmul(x, self.get_weight().T) + self.bias
    
    # return the weight as a linear layer
    def get_weight(self):
        return self.weight_vec * self.weight_norm
    def set_weight(self, tensor: torch.Tensor):
        norm = torch.linalg.norm(tensor, dim=1, keepdim=True)
        self.weight_vec.data = tensor/norm
        self.weight_norm.data = norm
        return self
    
    @classmethod
    def convert_etf_weights_to_linear(cls, model, model_dict):
        # change vec + norm -> weight
        vec_tensor: torch.Tensor
        norm_tensor: torch.Tensor
        vec_name: str
        for name in list(model_dict.keys()):
            if model.belongs_to_fc_layers(name):
                if name.endswith("weight_vec"):
                    vec_tensor = model_dict.pop(name)
                    vec_name = name
                elif name.endswith("weight_norm"):
                    norm_tensor = model_dict.pop(name)
        weight_tensor = vec_tensor * norm_tensor
        model_dict[vec_name[:-4]] = weight_tensor
        return model_dict
    
    @classmethod
    def convert_linear_weights_to_etf(cls, model, model_dict):
        # change vec + norm -> weight
        weights_tensor: torch.Tensor
        weights_name: str
        for name in list(model_dict.keys()):
            if model.belongs_to_fc_layers(name):
                if name.endswith("weight"):
                    weights_tensor = model_dict.pop(name)
                    weights_name = name
        norm_tensor = torch.linalg.norm(weights_tensor, dim=1, keepdim=True)
        vec_tensor = weights_tensor / norm_tensor
        model_dict[weights_name + "_vec"] = vec_tensor
        model_dict[weights_name + "_norm"] = norm_tensor
        return model_dict
        
    

#########################
## Abstract class 
#########################      
        
# Abstract class for NN model
class MyModel(nn.Module, metaclass = ABCMeta):
    def __init__(self):
        super().__init__()
        self.outFeature = False
        self.outAllFeatures = False
        self.fineGrainedFeatures = False
    # input parameter's name and return whether it belongs to final fc layer
    @abstractclassmethod
    def belongs_to_fc_layers(cls, name: str):
        raise NotImplementedError()
    @abstractclassmethod
    def belongs_to_bn_bias(cls, name: str):
        raise NotImplementedError()
    @abstractclassmethod
    def belongs_to_bn_weight(cls, name: str):
        raise NotImplementedError()
    # judge whether the batch normalization maps identity feature (not residual)
    # be careful, cause this function does not judge wheteher the module is batch normalization.
    @abstractclassmethod
    def belongs_to_identity_bn(cls, name: str):
        raise NotImplementedError()
    @abstractmethod
    # return fc_weights in tensor
    def get_fc_weight(self):
        raise NotImplementedError()
    @abstractmethod
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        raise NotImplementedError()
    @abstractmethod
    def get_fc_bias(self):
        raise NotImplementedError()
    @abstractmethod
    def set_fc_bias(self, tensor: torch.Tensor):
        raise NotImplementedError()
    # return params belonging to the fc layer
    @abstractmethod
    def get_fc_params_list(self):
        raise NotImplementedError()
    @abstractmethod
    def forward_last_layer(self, x: torch.Tensor):
        raise NotImplementedError()
    # when wd's target is restricted, check if the module is target or not
    @abstractmethod
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        raise NotImplementedError()
    # the length of features list when outAllFeatures is true, 
    @abstractmethod
    def get_features_n(self):
        raise NotImplementedError()
    @abstractmethod
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError()
    
    # append feature tensor to self.features and return 
    def f_append(self, x: torch.Tensor):
        self.features.append(x)
        return x
    
    def get_trained_params_list(self, args: BaseArgs):
        if args.project_name == "first":
            return [param for name, param in self.named_parameters() 
                    if (args.optim.train_bn_bias or not self.belongs_to_bn_bias(name)) and
                        (args.optim.train_bn_weight or not self.belongs_to_bn_weight(name))]
        elif args.project_name == "second":
            return [param for name, param in self.named_parameters() if self.belongs_to_fc_layers(name)]
    def get_trained_params_names_list(self, args: BaseArgs):
        if args.project_name == "first":
            return [name for name, param in self.named_parameters()
                    if (args.optim.train_bn_bias or not self.belongs_to_bn_bias(name)) and
                        (args.optim.train_bn_weight or not self.belongs_to_bn_weight(name))]
        elif args.project_name == "second":
            return [name for name, param in self.named_parameters() if self.belongs_to_fc_layers(name)]
    

                
    
    # this can load a model which has a different ETFClassifier
    def load_pretrained_model_dict(self, args: BaseArgs):
        pretrained_path = os.path.join(args.model.pretrained_path, str(args.seeds), BEST_PARAM_FILE_NAME)
        model_dict = torch.load(pretrained_path, map_location=args.device)
        loaded_args_path = os.path.join(args.model.pretrained_path, str(args.seeds), ARGS_FILE_NAME)
        loaded_args = BaseArgs.load(loaded_args_path, False)
        if args.model.use_etf == loaded_args.model.use_etf:
            pass
        elif args.model.use_etf:
            # Linear -> ETF
            model_dict = ETFClassifier.convert_linear_weights_to_etf(self, model_dict)
        else:
            # ETF -> Linear
            model_dict = ETFClassifier.convert_etf_weights_to_linear(self, model_dict)
        
        if args.model.load_only_representation:
            for name in list(model_dict.keys()):
                if self.belongs_to_fc_layers(name):
                    model_dict.pop(name)
        self.load_state_dict(model_dict, strict = (not args.model.load_only_representation) \
                             and (not args.model.ignore_restrict_load ))
        return self
    

class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 1, eps: float = 1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        param_size = (out_channels, )
        self.running_mean = nn.Parameter(torch.zeros(param_size, dtype=torch.float32))
        self.running_var = nn.Parameter(torch.ones(param_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(param_size, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(param_size, dtype=torch.float32))
        self.eps = eps
        self.act = nn.ReLU()
        
        self.outFeature = True
        self.outAllFeatures = False
        self.eval()
        self.broad_caster = [...,] + [None,]*2
    
    def forward(self, x):
        assert not self.training, "This model can't be trained (bn's training phase is not implemented)."
        self.features = []
        self.features.append(x)
        
        x = self.conv(x)
        self.features.append(x)
        
        x = (x - self.running_mean[self.broad_caster]) / torch.sqrt(self.running_var[self.broad_caster] + self.eps)
        self.features.append(x)
        
        x = self.weight[self.broad_caster] * x
        self.features.append(x)
        
        x = x + self.bias[self.broad_caster]
        self.features.append(x)
        
        x = self.act(x)
        self.features.append(x)
        
        if self.outAllFeatures:
            return self.features
        else:
            return x
    
    def load_resnet_weight(self, state_dict: dict):
        new_dict = {}
        new_dict["conv.weight"] = state_dict["encoder.conv1.weight"]
        new_dict["running_mean"] = state_dict["encoder.bn1.running_mean"]
        new_dict["running_var"] = state_dict["encoder.bn1.running_var"]
        new_dict["weight"] = state_dict["encoder.bn1.weight"]
        new_dict["bias"] = state_dict["encoder.bn1.bias"]
        
        self.load_state_dict(new_dict)
        
    def get_features_n(self):
        return 6
        
class BNLeNet5(MyModel):
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster, isPretrained=False, 
                 outFeature=False, useETF = False, 
                 outAllFeatures=False, useDRBalancer = False, imgNumPerCls = None):
        super().__init__()
        assert not useDRBalancer, "No Support for DRBalancer"
        assert not useETF, "No support for ETF"
        assert not isPretrained, "No support for pretraining"
        assert args.model.dropout == 0, "No support for dropout"
        assert args.model.use_bn, "No support for not using bn"
        assert not args.model.add_skip, "No support for not adding skip connection"
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias = False)
        self.bn11 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias = False)
        self.bn12 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120, bias = False)
        self.bn21 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias = False)
        self.bn22 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()
        
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.logitAdjuster = logit_adjuster

    def forward(self, x):
        self.features = []
        self.features.append(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.f_append(self.bn11(x))
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.f_append(self.bn12(x))
        
        x = torch.flatten(x, 1)
        x = self.f_append(self.relu(self.bn21(self.fc1(x))))
        x = self.f_append(self.relu(self.bn22(self.fc2(x))))
        
        if self.outFeature:
            if self.outAllFeatures:
                return self.features
            else:
                return x
        else:
            return self.forward_last_layer(x)
    
    def belongs_to_fc_layers(cls, name: str):
        # provisional
        return name.startswith("fc3")
    def belongs_to_bn_bias(cls, name: str):
        return name.startswith("bn") and name.endswith(".bias")
    def belongs_to_bn_weight(cls, name: str):
        return name.startswith("bn") and name.endswith(".weight")
    # return fc_weights in tensor
    def get_fc_weight(self):
        return self.fc3.weight
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        self.fc3.weight.data = tensor
    def get_fc_bias(self):
        return self.fc3.bias
    def set_fc_bias(self, tensor: torch.Tensor):
        self.fc3.bias.data = tensor
    # return params belonging to the fc layer
    def get_fc_params_list(self):
        return [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
    def forward_last_layer(self, x: torch.Tensor):
        return self.logitAdjuster(self.fc3(x))
    # when wd's target is restricted, check if the module is target or not
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        # target_list = args.optim.weight_decay_target_modules
        # if(name.startswith("encoder.layer")):
        #     return int(name[13]) in target_list
        # elif name == "encoder.fc":
        #     return 5 in target_list
        # else:
        #     return 0 in target_list
        raise NotImplementedError()
    def get_features_n(self):
        return 5
    

class BNMLP(MyModel):
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster, isPretrained=False, 
                 outFeature=False, useETF = False, 
                 outAllFeatures=False, useDRBalancer = False, imgNumPerCls = None):
        super().__init__()
        assert not useDRBalancer, "No Support for DRBalancer"
        assert not useETF, "No support for ETF"
        assert not isPretrained, "No support for pretraining"
        assert args.model.dropout == 0, "No support for dropout"
        assert args.model.use_bn, "No support for not using bn"
        assert not args.model.add_skip, "No support for not adding skip connection"
        
        self.hlayers_n = args.model.mlp_hlayer_n
        assert self.hlayers_n >= 1, "Hidden layers should be more than 0."
        linearList = [None]
        linearList[0] = nn.Linear(28*28, args.model.mlp_features_d, False)
        linearList += [nn.Linear(args.model.mlp_features_d, args.model.mlp_features_d, False)
                       for i in range(self.hlayers_n - 1)]
        self.linearList = nn.ModuleList(linearList)
        self.bnList = nn.ModuleList([nn.BatchNorm1d(args.model.mlp_features_d)
                                for i in range(self.hlayers_n)])
        self.fc = nn.Linear(args.model.mlp_features_d, 10)
        self.relu = nn.ReLU()
        
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.logitAdjuster = logit_adjuster
        self.fineGrainedFeatures = False

    def forward(self, x):
        self.features = []
        x = torch.flatten(x, 1)
        self.features.append(x)
        for i in range(self.hlayers_n):
            x = self.linearList[i](x)
            nowBN: nn.BatchNorm1d = self.bnList[i]
            if self.fineGrainedFeatures:
                self.features.append(x)
                self.features.append((x - nowBN.running_mean[None,:]) 
                                     / torch.sqrt(nowBN.eps + nowBN.running_var[None, :]))
            x = nowBN(x)
            if self.fineGrainedFeatures:
                self.features.append(x)
            x = self.f_append(self.relu(x))
        
        if self.outFeature:
            if self.outAllFeatures:
                return self.features
            else:
                return x
        else:
            return self.forward_last_layer(x)
    @classmethod
    def belongs_to_fc_layers(cls, name: str):
        # provisional
        return name.startswith("fc")
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        return name.startswith("bnList") and name.endswith(".bias")
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        return name.startswith("bnList") and name.endswith(".weight")
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        return False
    # return fc_weights in tensor
    def get_fc_weight(self):
        return self.fc.weight
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        self.fc.weight.data = tensor
    def get_fc_bias(self):
        return self.fc.bias
    def set_fc_bias(self, tensor: torch.Tensor):
        self.fc.bias.data = tensor
    # return params belonging to the fc layer
    def get_fc_params_list(self):
        return [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
    def forward_last_layer(self, x: torch.Tensor):
        return self.logitAdjuster(self.fc(x))
    # when wd's target is restricted, check if the module is target or not
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        # target_list = args.optim.weight_decay_target_modules
        # if(name.startswith("encoder.layer")):
        #     return int(name[13]) in target_list
        # elif name == "encoder.fc":
        #     return 5 in target_list
        # else:
        #     return 0 in target_list
        raise NotImplementedError()
    
    def get_features_n(self):
        return 1 + self.hlayers_n * (4 if self.fineGrainedFeatures else 1)
    
    def get_last_layer_weights_names_list(self):
        weights_names = ["linearList.{}.weight", "bnList.{}.weight",
                        "bnList.{}.bias", "bnList.{}.running_var", 
                        "bnList.{}.running_mean"]
        weights_names = [weight_name.format(self.hlayers_n-1) for weight_name in weights_names]
        return weights_names
        

class MLResBlock(MyModel):
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster, isPretrained=False, 
                 outFeature=False, useETF = False, 
                 outAllFeatures=False, useDRBalancer = False, imgNumPerCls = None):
        super().__init__()
        assert not useDRBalancer, "No Support for DRBalancer"
        assert not useETF, "No support for ETF"
        assert not isPretrained, "No support for pretraining"
        assert args.model.dropout == 0, "No support for dropout"
        assert args.model.use_bn, "No support for not using bn"
        assert not args.model.add_skip, "No support for not adding skip connection"
        
        self.hlayers_n = args.model.mlp_hlayer_n
        self.features_d = args.model.mlp_features_d
        assert self.hlayers_n >= 1, "Hidden layers should be more than 0."
        self.identity_skip = args.model.identity_skip # residual block proposed in "Identity Mappings in Deep Residual Networks"
        
        
        self.linear0= nn.Linear(28*28, self.features_d, False)
        self.bn0 = nn.BatchNorm1d(self.features_d)
        
        
        linearList = [nn.Linear(self.features_d, self.features_d, False)
                       for i in range(self.hlayers_n * 2)]
        self.linearList = nn.ModuleList(linearList)
        self.bnList = nn.ModuleList([nn.BatchNorm1d(self.features_d)
                                for i in range(self.hlayers_n*2)])
        self.fc = nn.Linear(args.model.mlp_features_d, 10)
        self.relu = nn.ReLU()
        
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.logitAdjuster = logit_adjuster
        self.fineGrainedFeatures = False
    
    def forward_res_bn_act(self, x: torch.Tensor, skip: torch.Tensor, i: int, activation: bool = True):
        nowBN: nn.BatchNorm1d = self.bnList[i]
        if self.fineGrainedFeatures:
            self.features.append((x - nowBN.running_mean[None,:]) 
                            / torch.sqrt(nowBN.eps + nowBN.running_var[None, :]) + skip)
        x = nowBN(x)
        if self.fineGrainedFeatures:
            self.features.append(x + skip)
        if activation:
            x = self.relu(x)
            if self.fineGrainedFeatures:
                self.features.append(x+skip)
        return x
    def forward_res_linear(self, x: torch.Tensor, skip: torch.Tensor, i: torch.Tensor):
        x = self.linearList[i](x)
        if self.fineGrainedFeatures:
            self.features.append(x + skip)
        return x
    
    def forward_residual_block(self, x, i):
        skip = x
        if self.identity_skip:
            # features_n = 1+3+1+3+1 = 9 or 1
            x = self.forward_res_bn_act(x, skip, 2*i)
            x = self.forward_res_linear(x, skip, 2*i)
            x = self.forward_res_bn_act(x, skip, 2*i+1)
            x = self.forward_res_linear(x, skip, 2*i+1)
            x = skip + x
            return self.f_append(x)

        else: 
            # features_n = 1+3+1+2+1+1 = 9 or 1
            x = self.forward_res_linear(x, skip, 2*i)
            x = self.forward_res_bn_act(x, skip, 2*i, True)
            x = self.forward_res_linear(x, skip, 2*i+1)
            x = self.forward_res_bn_act(x, skip, 2*i+1, False)
            x = skip + x
            if self.fineGrainedFeatures:
                self.features.append(x)
            return self.f_append(self.relu(x))


    def forward(self, x):
        self.features = []
        x = torch.flatten(x, 1)
        self.features.append(x)
        
        x = self.linear0(x)
        if self.fineGrainedFeatures:
            self.features.append(x)
            self.features.append((x - self.bn0.running_mean[None,:]) 
                            / torch.sqrt(self.bn0.eps + self.bn0.running_var[None, :]))
        x = self.bn0(x)
        if self.fineGrainedFeatures:
            self.features.append(x)
        x = self.f_append(self.relu(x))
        
        for i in range(self.hlayers_n):
            x = self.forward_residual_block(x, i)
        
        if self.outFeature:
            if self.outAllFeatures:
                return self.features
            else:
                return x
        else:
            return self.forward_last_layer(x)
    
    @classmethod
    def belongs_to_fc_layers(cls, name: str):
        # provisional
        return name.startswith("fc")
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        return name.startswith("bn") and name.endswith(".bias")
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        return name.startswith("bn") and name.endswith(".weight")
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        return name == "bn0"
    # return fc_weights in tensor
    def get_fc_weight(self):
        return self.fc.weight
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        self.fc.weight.data = tensor
    def get_fc_bias(self):
        return self.fc.bias
    def set_fc_bias(self, tensor: torch.Tensor):
        self.fc.bias.data = tensor
    # return params belonging to the fc layer
    def get_fc_params_list(self):
        return [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
    def forward_last_layer(self, x: torch.Tensor):
        return self.logitAdjuster(self.fc(x))
    # when wd's target is restricted, check if the module is target or not
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        # target_list = args.optim.weight_decay_target_modules
        # if(name.startswith("encoder.layer")):
        #     return int(name[13]) in target_list
        # elif name == "encoder.fc":
        #     return 5 in target_list
        # else:
        #     return 0 in target_list
        raise NotImplementedError()
    
    def get_features_n(self):
        return (5 if self.fineGrainedFeatures else 2) + self.hlayers_n * (9 if self.fineGrainedFeatures else 1)
    
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError



class MLResBlock(MyModel):
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster, isPretrained=False, 
                 outFeature=False, useETF = False, 
                 outAllFeatures=False, useDRBalancer = False, imgNumPerCls = None):
        super().__init__()
        assert not useDRBalancer, "No Support for DRBalancer"
        assert not useETF, "No support for ETF"
        assert not isPretrained, "No support for pretraining"
        assert not args.model.add_skip, "No support for not adding skip connection"
        
        self.hlayers_n = args.model.mlp_hlayer_n
        self.features_d = args.model.mlp_features_d
        assert self.hlayers_n >= 1, "Hidden layers should be more than 0."
        self.identity_skip = args.model.identity_skip # residual block proposed in "Identity Mappings in Deep Residual Networks"
        
        
        self.linear0= nn.Linear(28*28, self.features_d, False)
        self.bn0 = nn.BatchNorm1d(self.features_d)
        
        
        linearList = [nn.Linear(self.features_d, self.features_d, False)
                       for i in range(self.hlayers_n * 2)]
        self.linearList = nn.ModuleList(linearList)
        self.bnList = nn.ModuleList([nn.BatchNorm1d(self.features_d)
                                for i in range(self.hlayers_n*2)])
        self.fc = nn.Linear(args.model.mlp_features_d, 10)
        self.relu = nn.ReLU()
        
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.logitAdjuster = logit_adjuster
        self.fineGrainedFeatures = False
    
    def forward_res_bn_act(self, x: torch.Tensor, skip: torch.Tensor, i: int, activation: bool = True):
        nowBN: nn.BatchNorm1d = self.bnList[i]
        if self.fineGrainedFeatures:
            self.features.append((x - nowBN.running_mean[None,:]) 
                            / torch.sqrt(nowBN.eps + nowBN.running_var[None, :]) + skip)
        x = nowBN(x)
        if self.fineGrainedFeatures:
            self.features.append(x + skip)
        if activation:
            x = self.relu(x)
            if self.fineGrainedFeatures:
                self.features.append(x+skip)
        return x
    def forward_res_linear(self, x: torch.Tensor, skip: torch.Tensor, i: torch.Tensor):
        x = self.linearList[i](x)
        if self.fineGrainedFeatures:
            self.features.append(x + skip)
        return x
    
    def forward_residual_block(self, x, i):
        skip = x
        if self.identity_skip:
            # features_n = 1+3+1+3+1 = 9 or 1
            x = self.forward_res_bn_act(x, skip, 2*i)
            x = self.forward_res_linear(x, skip, 2*i)
            x = self.forward_res_bn_act(x, skip, 2*i+1)
            x = self.forward_res_linear(x, skip, 2*i+1)
            x = skip + x
            return self.f_append(x)

        else: 
            # features_n = 1+3+1+2+1+1 = 9 or 1
            x = self.forward_res_linear(x, skip, 2*i)
            x = self.forward_res_bn_act(x, skip, 2*i, True)
            x = self.forward_res_linear(x, skip, 2*i+1)
            x = self.forward_res_bn_act(x, skip, 2*i+1, False)
            x = skip + x
            if self.fineGrainedFeatures:
                self.features.append(x)
            return self.f_append(self.relu(x))


    def forward(self, x):
        self.features = []
        x = torch.flatten(x, 1)
        self.features.append(x)
        
        x = self.linear0(x)
        if self.fineGrainedFeatures:
            self.features.append(x)
            self.features.append((x - self.bn0.running_mean[None,:]) 
                            / torch.sqrt(self.bn0.eps + self.bn0.running_var[None, :]))
        x = self.bn0(x)
        if self.fineGrainedFeatures:
            self.features.append(x)
        x = self.f_append(self.relu(x))
        
        for i in range(self.hlayers_n):
            x = self.forward_residual_block(x, i)
        
        if self.outFeature:
            if self.outAllFeatures:
                return self.features
            else:
                return x
        else:
            return self.forward_last_layer(x)
    
    @classmethod
    def belongs_to_fc_layers(cls, name: str):
        # provisional
        return name.startswith("fc")
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        return name.startswith("bn") and name.endswith(".bias")
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        return name.startswith("bn") and name.endswith(".weight")
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        return name == "bn0"
    # return fc_weights in tensor
    def get_fc_weight(self):
        return self.fc.weight
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        self.fc.weight.data = tensor
    def get_fc_bias(self):
        return self.fc.bias
    def set_fc_bias(self, tensor: torch.Tensor):
        self.fc.bias.data = tensor
    # return params belonging to the fc layer
    def get_fc_params_list(self):
        return [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
    def forward_last_layer(self, x: torch.Tensor):
        return self.logitAdjuster(self.fc(x))
    # when wd's target is restricted, check if the module is target or not
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        # target_list = args.optim.weight_decay_target_modules
        # if(name.startswith("encoder.layer")):
        #     return int(name[13]) in target_list
        # elif name == "encoder.fc":
        #     return 5 in target_list
        # else:
        #     return 0 in target_list
        raise NotImplementedError()
    
    def get_features_n(self):
        return (5 if self.fineGrainedFeatures else 2) + self.hlayers_n * (9 if self.fineGrainedFeatures else 1)
    
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError

class LinearBNAct(nn.Module):
    def __init__(self, in_features, out_features, activation = nn.ReLU, dropout = 0.0, use_bn = True, 
                 use_skip = False):
        super().__init__()
        self.use_bn = use_bn
        self.use_skip = use_skip
        self.linear = nn.Linear(in_features, out_features, False)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.do = nn.Dropout(dropout)
        self.act = activation()
        
    
    def forward(self, x):
        new_x = self.linear(x)
        if self.use_bn:
            new_x = self.bn(new_x)
        if self.use_skip:
            new_x = new_x + x
        new_x = self.do(new_x)
        new_x = self.act(new_x)
        return new_x

class TableMLP(MyModel):
    LAYER_N = 8
    """Pytorch module for a resnet encoder
    """
    def __init__(self, args: BaseArgs, logit_adjuster: LogitAdjuster, inputDimension=27,
                 embDimension=512, outFeature=False, useETF = False, 
                 outAllFeatures=False, useDRBalancer = False, imgNumPerCls = None, use_skip = False):
        modelargs : ModelArgs = args.model
        super().__init__()
        assert not useDRBalancer, "No Support for DRBalancer"
        
        self.embDimension = embDimension
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures
        self.useETF = useETF
        self.dropout = modelargs.dropout
        self.use_bn = modelargs.use_bn            
        if self.embDimension>0:
            if self.useETF:
                self.fc =  ETFClassifier(self.embDimension, args.data.n_classes, modelargs.etf_norm_rate,
                                                 args.model.train_etf_norm, use_dr_balancer=useDRBalancer,
                                                 img_num_per_cls=imgNumPerCls)
            else:
                self.fc =  nn.Linear(self.embDimension, args.data.n_classes)
        
        # assert not self.use_feature_bias, "No support for feature bias"   
        self.use_feature_bias = modelargs.use_feature_bias
        if self.use_feature_bias:
            self.feature_bias = torch.nn.Parameter(torch.zeros((1, self.num_ch_enc[-1]), dtype=torch.float32))
        self.construct_encoder(args, embDimension, inputDimension, use_skip)
            
        self.__init_trained_params_list__(args)
        self.normalizeFeature = modelargs.normalize_feature

        self.logitAdjuster = logit_adjuster
        self.logitNormalization = modelargs.logit_normalization
        assert not modelargs.last_relu, "No support for last relu"
        
    def construct_encoder(self, args: BaseArgs, embDimension, inputDimension, use_skip):
        module_list = [LinearBNAct(inputDimension, embDimension, dropout = self.dropout,
                                   use_bn = self.use_bn)]
        module_list += [LinearBNAct(embDimension, embDimension,
                                    dropout = self.dropout, use_bn=self.use_bn,
                                    use_skip=(use_skip and i < (self.LAYER_N -1))) for i in range(self.LAYER_N)]
        self.encoder = nn.ModuleList(module_list)
    
    
    def forward(self, input_data):
        self.features = []
        self.features.append(input_data.detach())
        
        x = input_data
        for module in self.encoder:
            # print(module.linear.weight.data[0, :10])
            x = module(x)
            self.features.append(x.detach())
                
        if self.normalizeFeature:
            x = x / (torch.linalg.norm(x, dim=1, keepdim = True) + 1e-10)
        
        if self.use_feature_bias:
            x = x + self.feature_bias
        
        if self.outFeature:
            if self.outAllFeatures:
                self.features.append(x)
                return self.features
            else:
                return x
        
        x = self.forward_last_layer(x)
        return x
    
    def get_fc_weight(self):
        if self.useETF:
            fc: ETFClassifier = self.fc
            return fc.get_weight()
        else:
            return self.state_dict()["fc.weight"]
        
    def set_fc_weight(self, tensor: torch.Tensor):
        if self.useETF:
            fc: ETFClassifier = self.fc
            fc.set_weight(tensor)
        else:
            self.fc.weight.data = tensor
        
    def get_fc_bias(self):
        return self.state_dict()["fc.bias"]
    
    def set_fc_bias(self, tensor: torch.Tensor):
        self.fc.bias.data = tensor
        return self
    
    def forward_last_layer(self, x: torch.Tensor):
        x = self.fc(x)
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
        raise NotImplementedError

    def get_last_layer_weights_names_list(self):
        raise NotImplementedError
                
            
    
    @classmethod
    def belongs_to_fc_layers(cls, name: str):
        return name.startswith("fc.")
    
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        return name.endswith(".bias") and (not name.endswith("fc.bias")) or name.endswith(".bias_param")
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        return name.endswith(".weight") and (not name.endswith("fc.weight")) and \
            (not name.endswith("linear.weight")) or name.endswith(".weight_param")
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        return name.startswith("encoder.bn") or ("downsample" in name)
            
    def get_features_n(self):
        return self.LAYER_N + 3

