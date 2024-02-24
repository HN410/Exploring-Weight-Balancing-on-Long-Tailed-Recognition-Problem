import torch
import os
from utils.utils import *
from utils.network_arch_resnet import *
from utils.model_units import MyModel, BNLeNet5, BNMLP, MLResBlock, TableMLP
from utils.data_units import MINI_IMAGENET_DATA, IMAGENET_DATA, get_input_dimension
class mrelu(torch.nn.Module):
    def __init__(self, bias = -1.0):
        super().__init__()
        self.bias = bias
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.bias + self.relu(x-self.bias)     
    
def get_activation_nn(args: BaseArgs):
    if args.model.activation_type == "lrelu":
        return torch.nn.LeakyReLU(args.model.activation_param)
    elif args.model.activation_type == "selu":
        return torch.nn.SELU()
    elif args.model.activation_type == "elu":
        return torch.nn.ELU()
    elif args.model.activation_type == "mrelu":
        return mrelu()
    else:
        assert False, "Invalid activation type"

# change ReLU to leakyReLU
def exchange_ReLU(model, args: BaseArgs):
    assert args.model.name == "ResNet34", "No support for models other than resnet34"
    model.encoder.relu = get_activation_nn(args)
    
    encoder = model.encoder
    for submodule in [encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]:
        for subsubmodule in submodule:
            subsubmodule.relu = get_activation_nn(args)
    return model
        

def modify_bn_weight(model: MyModel, args: BaseArgs):
    margs = args.model
    if margs.bn_weight_init_distribution == "pretrained":
        weights_dict = torch.load(os.path.join(margs.pretrained_path, str(args.seeds), BEST_PARAM_FILE_NAME), args.device)
        for name, module in model.named_modules():
            if is_batchnorm(module):
                module.weight.data = margs.pretrained_gamma * weights_dict[name+".weight"] + margs.pretrained_beta
    else:
        modifier = None
        if margs.bn_weight_init_distribution == "uniform":
            def modifier(weight: torch.Tensor):
                new_weight = torch.rand_like(weight)
                new_weight = (margs.bn_weight_init_max - margs.bn_weight_init_min) * new_weight + margs.bn_weight_init_min
                return new_weight
        elif margs.bn_weight_init_distribution == "gauss":
            def modifier(weight: torch.Tensor):
                new_weight = torch.randn_like(weight)
                return new_weight * margs.bn_weight_init_std + margs.bn_weight_init_mean
        elif margs.bn_weight_init_distribution == "rectified_gauss":
            def modifier(weight: torch.Tensor):
                new_weight = torch.randn_like(weight)
                return torch.relu(new_weight * margs.bn_weight_init_std + margs.bn_weight_init_mean)
        else:
            assert False, "Not supported bn_weight_init_distribution: " + margs.bn_weight_init_distribution
        for name, module in model.named_modules():
            if is_batchnorm(module) and (args.model.init_identity_bn or not model.belongs_to_identity_bn(name)):
                module.weight.data = modifier(module.weight.data)
    return model

def is_for_224(args: BaseArgs):
    return args.data.name.startswith(MINI_IMAGENET_DATA) or args.data.name.startswith(IMAGENET_DATA)

def get_base_model(args: BaseArgs, outFeature=False, train_img_num_per_class = None, get_residual = False):
    #TODO support outFeature for ResNext50
    fc_bias_param_name = None
    use_etf = args.model.use_etf
    
    model: MyModel
    logit_adjuster = LogitAdjuster(args.model.logit_adjustment_tau, train_img_num_per_class,
                                    args.model.logit_adjustment_for_training, args.model.use_additive_logit_adjustment)
    if args.model.name == 'ResNet34' or args.model.name == 'IMResNet34':
        model = ResnetEncoder(args, logit_adjuster, 34, False,
            embDimension=args.data.n_classes, poolSize=4, outFeature=outFeature, useETF=use_etf, 
            getResidual = get_residual, useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class, for224 = is_for_224(args)).to(args.device)
    elif args.model.name == 'ResNet18' or args.model.name == 'IMResNet18':
        model = ResnetEncoder(args, logit_adjuster, 18, False,
            embDimension=args.data.n_classes, poolSize=4, outFeature=outFeature, useETF=use_etf, 
            getResidual = get_residual, useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class, for224 = is_for_224(args)).to(args.device)
    
    elif args.model.name == "ResNeXt50":
        model = ResnetEncoder(args, logit_adjuster, 50, False,
            embDimension=args.data.n_classes, poolSize=4, outFeature=outFeature, useETF=use_etf, 
            getResidual = get_residual, useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class, for224 = is_for_224(args),
            useResNext=True).to(args.device)
    elif args.model.name == "BNLeNet5":
        model = BNLeNet5(args, logit_adjuster, False,
            outFeature=outFeature, useETF=use_etf, 
            useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class).to(args.device)
    elif args.model.name.startswith("BNMLP"):
        model = BNMLP(args, logit_adjuster, False,
            outFeature=outFeature, useETF=use_etf, 
            useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class).to(args.device)
    elif args.model.name.startswith("MLResBlock"):
        model = MLResBlock(args, logit_adjuster, False,
            outFeature=outFeature, useETF=use_etf, 
            useDRBalancer=args.model.use_dr_balancer,
            imgNumPerCls= train_img_num_per_class).to(args.device)
    elif args.model.name.startswith("TableMLP"):
        model = TableMLP(args, logit_adjuster, get_input_dimension(args), 512, outFeature,
                         use_etf, useDRBalancer=args.model.use_dr_balancer,
                         imgNumPerCls= train_img_num_per_class, use_skip = args.model.add_skip).to(args.device)
    else:
        assert False, "Invalid model name"
    
    if args.model.activation_type != "relu":
        model = exchange_ReLU(model, args)
        
    # rewriting the weight of fc.bias for sigmoid focal loss.
    if args.optim.norm_type == "CB" and "cb_type" in vars(args.optim) and \
        args.optim.cb_type != "softmax":
        
        weight_magnitude = - 1 * np.log(args.data.n_classes -1)
        model.get_fc_bias()[:] = \
            torch.ones(args.data.n_classes, dtype = torch.float32, device=args.device) * weight_magnitude
    
    # init bn's weight
    if args.model.modify_bn_weight:
        model = modify_bn_weight(model, args)
        
    # rewriting bn's eps
    if args.model.bn_eps != DEFAULT_BN_EPS:
        for module in model.modules():
            if is_batchnorm(module):
                module.eps = args.model.bn_eps
    
    # # reduce the learned parameters in BN
    # if args.model.one_param_bn:
    #     assert not args.model.modify_bn_weight, NotImplementedError
    #     for name, module in model.named_modules():
    #         if is_batchnorm(module):
    #             name = name.replace(".0", "[0]").replace(".1", "[1]").replace(".2", "[2]").replace(".3", "[3]").replace(".4", "[4]").replace(".5", "[5]")
    #             new_bn = OneParamBN(module.num_features, device=args.device)
    #             exec("model."+name+" = new_bn")
                

    return model

# def get_models_fc_weights_name(model_name: str):
#     if model_name == 'ResNet34':
#         return "encoder.fc.weight"
#     elif model_name == "ResNeXt50":
#         return "fc.weight"
#     else:
#         assert False, "Invalid model name"

# def get_models_fc_bias_name(model_name: str):
#     if model_name == 'ResNet34':
#         return "encoder.fc.bias"
#     elif model_name == "ResNeXt50":
#         return "fc.bias"
#     else:
#         assert False, "Invalid model name"

# def get_models_last_layer(model: nn.Module, model_name: str):
#     if model_name == 'ResNet34':
#         return model.encoder.fc
#     elif model_name == "ResNeXt50":
#         return model.fc
#     else:
#         assert False, "Invalid model name"

# def load_pretrained_model_dict(args: BaseArgs, model: nn.Module):
#     model_dict = torch.load(args.model.pretrained_path, map_location=args.device)
    
#     if args.model.load_only_representation:
#         for name in list(model_dict.keys()):
#             if name.startswith("encoder.fc"):
#                 model_dict.pop(name)
#     model.load_state_dict(model_dict, strict = not args.model.load_only_representation)
#     return model

# Class for changing dataset's transform using "with"
class OutFeatureChanger():
    def __init__(self, model: MyModel, set_out_feature: bool, set_out_all_features: bool = False):
        self.model = model
        self.before_out_feature = model.outFeature
        self.before_out_all_features = model.outAllFeatures
        self.set_out_feature = set_out_feature
        self.set_out_all_features = set_out_all_features
        
    def __enter__(self):
        self.model.outFeature = self.set_out_feature
        self.model.outAllFeatures = self.set_out_all_features
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.model.outFeature = self.before_out_feature
        self.model.outAllFeatures = self.set_out_all_features
        
