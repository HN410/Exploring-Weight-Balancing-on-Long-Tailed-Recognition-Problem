##This implementation is imported from https://github.com/vandit15/Class-balanced-loss-pytorch and modified.

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
from utils.model_units import MyModel

# Dot Regression Loss in "Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?"
def DotRegressionLoss(logits: torch.Tensor, labels: torch.Tensor, model: MyModel, features: torch.Tensor
                      , reduction: str="mean"):
    gt_logits = torch.stack([logits[i, label] for i, label in enumerate(labels.cpu().detach().tolist())])
    features_norms = torch.linalg.norm(features, dim = 1)
    weights_norms = torch.linalg.norm(model.get_fc_weight()[labels], dim = 1)
    
    norms_product = (features_norms * weights_norms).detach() # separate from comp graph
    net_loss =  torch.pow(gt_logits - norms_product, 2)
    net_loss = net_loss / (2 * norms_product)
    
    if reduction == "mean":
        return net_loss.mean()
    elif reduction == "none":
        return net_loss
    else:
        assert False, "Invalid reduction type."


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    if(torch.isnan(BCLoss).sum().item()):
      print(logits)
      print(labels)
      print(BCLoss)
      assert False, "BCLoss is nan."
      
    # print(logits.min(), logits.max()) 
    if gamma == 0.0:
        modulator = 1.0
    else:
      # default implementation is unstable when logits are too small
        modulator = torch.exp(-gamma * labels * logits - gamma * (torch.log(torch.exp(logits) + 1) - logits))
    assert not torch.isnan(modulator).sum().item(), "modulator is nan."

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss, axis = 1)

    # focal_loss /= torch.sum(labels)

    assert not torch.isnan(focal_loss).sum().item(), "focal_loss is nan."
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device,
            model: MyModel, features: torch.Tensor, reduction: str="mean"):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    if beta != 1.0:
      # effective_num = 1 - torch.pow(torch.tensor([beta]), torch.tensor(samples_per_cls, torch.float32))
      # effective_num = effective_num.numpy()
      effective_num = 1.0 - np.power(np.array([beta]*len(samples_per_cls)), np.array(samples_per_cls, np.float))
      weights = (1.0 - beta) / np.array(effective_num)
    else:
      weights = 1.0 / np.array(samples_per_cls, dtype=np.float)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    labels_one_hot = labels_one_hot.to(device)

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1).to(device)
    weights = weights* labels_one_hot
    weights = weights.sum(1)

    if loss_type == "focal":
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights, reduction="none")
    elif loss_type == "softmax":
        cb_loss = F.cross_entropy(logits, labels_one_hot, reduce=False)
        cb_loss = (weights * cb_loss)
    elif loss_type == "dr":
        cb_loss = DotRegressionLoss(logits, labels, model, features, reduction="none")
        cb_loss = weights * cb_loss
    else:
      assert "Invalid cb loss type."
    
    if reduction == "mean":
      return cb_loss.mean()
    elif reduction == "none":
      return cb_loss
    else:
      assert False, "Invalid reduction type."

def NC_loss(logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor, lambda1: float, lambda2: float, reduction: str = "mean"):
    max_class = labels.max() + 1
    loss = torch.nn.CrossEntropyLoss(reduction = reduction)(logits, labels)
    
    # calc mean features per class
    features_mean = torch.stack([features[labels == i].mean(dim=0) for i in range(max_class)])
    # replace nan
    features_mean = torch.nan_to_num(features_mean)
    labels_num = torch.stack([(labels == i).sum() for i in range(max_class)]).to(torch.float)
    # clac the distance between features and mean features per class
    loss = loss + lambda1 * (torch.norm(features - features_mean[labels], 2, dim=1) ** 2  / labels_num[labels]).sum()
    

    # remove zero vectors
    features_mean = features_mean[torch.norm(features_mean, 2, dim=1) > 0]
    features_mean_norm = features_mean / torch.norm(features_mean, 2, dim=1, keepdim=True)
    cos_sim = torch.mm(features_mean_norm, features_mean_norm.T)
    # replace diagonal elements with -1
    cos_sim[torch.eye(cos_sim.shape[0]).bool()] = -1
    deg_sim =torch.arccos(cos_sim - 1e-2)
    if deg_sim.shape[0] > 0:
        loss -= lambda2 * torch.nan_to_num(torch.min(deg_sim, dim = 1).values, nan=0.0).mean()
    else:
      pass
        # print(features.mean(dim = 0)[:10])
    return loss


if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma, "cpu", "none")
    print(cb_loss)
