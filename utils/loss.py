import torch.nn as nn
from utils.utils import *
from utils.class_balanced_loss import CB_loss, DotRegressionLoss, NC_loss
from utils.regularizers import set_weight_decay, set_features_regularization
from utils.model_units import MyModel
CROSSENTROPY_NAME = "CE"
CLASSBALANCED_NAME = "CB"
DOTREGRESSION_NAME = "DR"
FOCAL_LOSS = "FL"
NC_LOSS = "NC"

'''
Tensor shapes of the inputs of the loss function
B: batch size, C: the number of the classes, F: the length of the feature vector
- logits ... [B, C]
- labels ... [B]
- features ... [B, F]
'''

# Deferred Re-balancing Optimization Scheduler
# if drw <= epoch, start rebalancing
def check_drw(args: BaseArgs, epoch: int, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"):
    if args.optim.drw <= epoch:
        return None
    else:
        return  nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
    

def get_loss_function(img_num_per_cls: list, args: BaseArgs, reduction: str="mean"):
    loss_type = args.optim.loss_type
    use_features = False
    if loss_type == CROSSENTROPY_NAME:
        ce = nn.CrossEntropyLoss(reduction=reduction)
        def CE_lossFunc(logits, labels, model = None, features = None, epoch = 0):
            return ce(logits, labels)
        loss = CE_lossFunc
    elif loss_type == CLASSBALANCED_NAME:
        if "cb_type" in vars(args.optim):
            use_features = args.optim.cb_type == "dr"
            def CB_lossFunc(logits, labels, model = None, features = None, epoch = 0): #defince CB loss function
                drw = check_drw(args, epoch, logits, labels, reduction)
                if drw is not None:
                    return drw
                return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                args.optim.cb_type, args.optim.cb_beta, args.optim.cb_gamma, args.device,
                model, features, reduction=reduction)
        else:
            def CB_lossFunc(logits, labels, model = None, features = None, epoch= 0): #defince CB loss function
                drw = check_drw(args, epoch, logits, labels, reduction)
                if drw is not None:
                    return drw
                return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                "softmax", 0.9999, 2.0, args.device, model, features, reduction=reduction)
        loss = CB_lossFunc
    elif loss_type == DOTREGRESSION_NAME:
        use_features = True
        def DR_Loss(logits, labels, model, features, epoch = 0):
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return DotRegressionLoss(logits, labels, model, features, reduction)
        loss = DR_Loss
    elif loss_type == FOCAL_LOSS:
        def CB_lossFunc(logits, labels, model = None, features = None, epoch = 0): #defince CB loss function
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                        "focal", 1.0, args.optim.cb_gamma, args.device,
                        model, features, reduction=reduction)
        loss = CB_lossFunc
    elif loss_type == NC_LOSS:
        use_features = True
        def NC_lossFunc(logits, labels, model, features, epoch = 0):
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return NC_loss(logits, labels, features, args.optim.nc_lambda1, args.optim.nc_lambda2, reduction)
        loss = NC_lossFunc
    else:
        assert False, "Invalid loss function type."
    return loss, use_features


def set_loss(img_num_per_cls: list, args: BaseArgs, reduction: str="mean"):
    loss_function, use_features = get_loss_function(img_num_per_cls, args, reduction=reduction)
    loss_function = set_weight_decay(loss_function, img_num_per_cls, args)
    loss_function = set_features_regularization(loss_function, args)
    return loss_function, use_features
