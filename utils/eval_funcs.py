from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import sklearn
from utils.dataset_CIFAR import *
from utils.datasets import get_many_threshold, get_few_threshold
from utils.utils import *
from utils.models import OutFeatureChanger
from utils.model_units import MyModel
import traceback


def print_and_save_accuracy(accs_tuple, logger, args, test_aug = True, save_acc = True):
    (acc_avg_class, breakdown_results) = accs_tuple
    logger.info('acc avgClass: '+"{:.1%}".format(acc_avg_class))

    logger.info('Many:' + "{:.1%}".format(breakdown_results[0]) + 
                 'Medium:' + "{:.1%}".format(breakdown_results[1]) + 
                 'Few:' + "{:.1%}".format(breakdown_results[2]))
    
    acc_dict = {"All": acc_avg_class, "Many": breakdown_results[0],
                "Medium": breakdown_results[1], "Few": breakdown_results[2]}
    if save_acc:
        with open(os.path.join(get_work_dir_path(args), ACCURACY_FILE_NAME), "w") as f:
            json.dump(acc_dict, f)
        


def horizontal_flip_aug(model):
    def aug_model(data):
        logits = model(data)
        h_logits = model(data.flip(3))
        return (logits+h_logits)/2
    return aug_model

def mic_acc_cal(preds, labels):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def shot_acc(preds, labels, train_data, many_shot_thr, low_shot_thr, acc_per_cls=False):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def get_confusion_matrix(model: nn.Module, dataloaders: dict, device: str):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloaders['test']:
        with torch.no_grad():
            _, images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)   

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))    
            grndList = np.concatenate((grndList, labels))


    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a  
    
    return confMat  
    
    
def get_per_class_acc(conf_mat, n_classes):
    acc_avgClass = 0
    for i in range(conf_mat.shape[0]):
        acc_avgClass += conf_mat[i,i]

    acc_avgClass /= conf_mat.shape[0]
    
    acc_per_class = [0] * n_classes

    for i in range(n_classes):
        acc_per_class[i] = conf_mat[i,i]
    
    return acc_per_class   
    

def get_features_matrix(dataset: Dataset, model: nn.Module, args: BaseArgs, save: bool = False):
    assert model.outFeature, "Set model to output feature vectors"
    
    with dataset.set_augment(False):
        dataloader =  DataLoader(dataset, batch_size=args.data.batch_size,
                                shuffle=False, num_workers=4) # num_work can be set to batch_size
        features_list = [None] * len(dataloader)
        for i, sample in enumerate(dataloader):
            with torch.no_grad():
                _, images, _ = sample
                images = images.to(args.device)
                features = model(images)
                features_list[i] = features.cpu()
        
    features_matrix = torch.concat(features_list)
    if save:
        np.save(os.path.join(get_work_dir_path(args), FEATURESMAT_FILE_NAME), features_matrix.numpy())
    return features_matrix
    
# calculate Fisher's discriminant ratio
# due to large errors, calc by double precision
# features_tensor: [B, D]
def calc_fdr(features_tensor: torch.Tensor, labels_tensor: torch.Tensor, args: BaseArgs = None, verbose: bool = False):
    if args is None:
        n_classes = torch.unique(labels_tensor).shape[0]
        device = features_tensor.device
    else:
        n_classes = args.data.n_classes
        device = args.device
        
    features_tensor = features_tensor.to(torch.float64)
    # eliminate dimensions whose values are always same
    zero_counts = (torch.abs(features_tensor - features_tensor[0:1]) <= features_tensor.mean().abs()*1e-3).sum(dim = 0)
    features_tensor = features_tensor[:, zero_counts < features_tensor.shape[0]]

    
    img_num_per_cls_tensor = torch.tensor([torch.count_nonzero(labels_tensor == i) for i in range(n_classes)]).to(device)
    features_mean_per_cls = torch.stack([features_tensor[labels_tensor == i].mean(dim=0) for i in range(n_classes)])
    features_mean = features_tensor.mean(axis=0)
    try:
        s_b = features_mean_per_cls - features_mean
        s_b = torch.mm(s_b.T * img_num_per_cls_tensor, s_b)
        if n_classes > 100:
            s_w = torch.zeros((features_tensor.shape[1], features_tensor.shape[1]), dtype=torch.float64).to(device)
            for i in range(n_classes):
                s_w_in = features_tensor[i == labels_tensor] - features_mean_per_cls[i]
                s_w += torch.mm(s_w_in.T, s_w_in)
        else:
            s_w = [features_tensor[i == labels_tensor] - features_mean_per_cls[i] for i in range(n_classes)]
            s_w = torch.stack([torch.mm(s_w_in.T, s_w_in) for s_w_in in s_w]).sum(dim=0)
        
        # zero check
        s_w_0_ind = (s_w.abs() <= features_tensor.mean().abs()*1e-3).sum(dim = 0) == s_w.shape[1]
        if s_w_0_ind.sum() > 0:
            used_ind = torch.logical_not(s_w_0_ind)
            print("Found 0 vec in s_w")
            s_w = s_w[used_ind, :][:,used_ind]
            s_b = s_b[used_ind, :][:, used_ind]
            print(s_w.shape)
    except:
        print(traceback.format_exc())
        # s_b becomes too big
        if verbose:
            return torch.nan, None, None, torch.tensor(torch.nan), torch.tensor(torch.nan)
        return torch.nan
    
    try:
        inv = torch.linalg.solve(s_w, s_b)
        # if verbose:
        #     _, s_s_w, _ = torch.svd(s_w, compute_uv=False)
        #     inv_s_w_norm = torch.sqrt(torch.sum(torch.pow(1 / s_s_w, 2)))
    except:
        torch.save(features_tensor.cpu(), "exp_data/s_w.ckpt")
        print(traceback.format_exc())
        if verbose:
            return torch.nan, None, None, torch.tensor(torch.nan), torch.tensor(torch.nan)
        return torch.nan
    ratio = torch.trace(inv).item()
    if verbose:
        return ratio, s_w, s_b, torch.trace(s_w), torch.trace(s_b)
    else:
        return ratio
        
def print_and_save_indicator(train_indicator: float, test_indicator: float, indicator_name: str,  file_name: str, args: BaseArgs):
    data_dict = {"train": train_indicator, "test": test_indicator}
    args.logger.info("{} --- train: {:.2f}, test: {:.2f}".format(indicator_name, train_indicator, test_indicator))
    with open(os.path.join(get_work_dir_path(args), file_name), "w") as f:
        json.dump(data_dict, f)
    

# def calc_logits_and_error(model: MyModel, loss_func, images_tensor: torch.Tensor, labels_tensor: torch.Tensor, use_fearures: bool):
#     if use_fearures:
#         with OutFeatureChanger(model, True):
#             features = model(images_tensor)
#             logits = model.get_last_layer()(features)
#     else:
#         features=None
#         logits = model(images_tensor)
#     error = loss_func(logits, labels_tensor, model, features)
#     return logits, error

TRAIN_STATIC_KEY = "train_data"
FORGET_SCORE_KEY = "forget_score"
CONFUSION_MAT_KEY = "confusion_mat"
FEATURES_MAT_KEY = "features_mat"
ACCS_KEY = "accs"
LOSS_KEY = "loss"
LABEL_KEY = "labels"
PRED_KEY = "preds"
TF_TENSOR_KEY = "tf_tensor" # used for computing forgetting score 
# Train or test model for all data
# You can specify what data you want

def process_all_data(dataloader: torch.utils.data.DataLoader, args: BaseArgs,
                     model: MyModel, key_set: set, new_label_list: list,
                     optimizer: torch.optim.Optimizer = None,
                     loss_func = None, out_features_all = False, epoch: int = 0):
    is_training = optimizer is not None
    use_loss = is_training or LOSS_KEY in key_set or TRAIN_STATIC_KEY in key_set
    assert not use_loss or loss_func is not None, "Set loss_func if you will train the model"
    iter_count =  sample_count = 0
    running_acc = running_loss = 0.0
    ans_dict = {}
    
    pred_list = [None] * len(dataloader)
    grnd_list = [None] * len(dataloader)
    features_list_list = [[None for j in range(len(dataloader))] for i in range(model.get_features_n() if out_features_all else 1)] 
    loss_list = [None] * len(dataloader)
    data_indices_list = [None] * len(dataloader)
    
    if is_training:
        model.train()
    else:
        model.eval()
    
    for batch_index, sample in enumerate(dataloader):
        # if batch_index % 100 == 0:
        #     args.logger.info(f"{batch_index} / {len(dataloader)}")
        torch.cuda.empty_cache()                 
        data_indices, image_list, label_list = sample
        data_indices_list[batch_index] = data_indices
        image_list = image_list.to(args.device)
        label_list = label_list.type(torch.long).view(-1).to(args.device)

        if is_training:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_training):
            with OutFeatureChanger(model, True, out_features_all):
                features_list = model(image_list)
                features = features_list[-1] if out_features_all else features_list
                logits = model.forward_last_layer(features)
            softmax_scores = logits.softmax(dim=1)

            pred_label = softmax_scores.argmax(dim=1).detach().squeeze().type(torch.float)                  
            acc_rate = (label_list.type(torch.float).squeeze() - pred_label.squeeze().type(torch.float))
            acc_rate = (acc_rate==0).type(torch.float).mean()
            
            pred_list[batch_index] = pred_label.cpu()
            grnd_list[batch_index] = label_list.cpu()
            if out_features_all:
                for i in range(model.get_features_n()):
                    features_list_list[i][batch_index] = features_list[i].detach().cpu()
            else:
                features_list_list[0][batch_index] = features_list.detach().cpu()

            # backward + optimize only if in training phase
            if use_loss:
                error = loss_func(logits, label_list, model, features, epoch)
                loss_list[batch_index] = error.detach().cpu()
                error = error.mean()
                if is_training:
                    error.backward()
                    optimizer.step()
        
        if use_loss:
            # statistics  
            iter_count += 1
            sample_count += label_list.size(0)
            running_acc += acc_rate*label_list.size(0) 
            running_loss += error.item() * label_list.size(0) 
            
            print2screen_avgLoss = running_loss / sample_count
            epoch_error = print2screen_avgLoss
            print2screen_avgAccRate = running_acc / sample_count
        
    pred_list = torch.cat(pred_list)
    grnd_list = torch.cat(grnd_list)
    data_indices_list = torch.cat(data_indices_list)
    if TF_TENSOR_KEY in key_set:
        tf_tensor = torch.zeros_like(data_indices_list, dtype=torch.long)
        tf_tensor[data_indices_list] = (pred_list == grnd_list) + 0
        ans_dict[TF_TENSOR_KEY] = tf_tensor
    if LABEL_KEY:
        ans_dict[LABEL_KEY] = grnd_list
    if PRED_KEY:
        ans_dict[PRED_KEY] = pred_list
    if TRAIN_STATIC_KEY in key_set:
        ans_dict[TRAIN_STATIC_KEY] = (epoch_error, print2screen_avgAccRate)
    if CONFUSION_MAT_KEY in key_set or ACCS_KEY in key_set:
        pred_list = pred_list.numpy()
        grnd_list = grnd_list.numpy()
        conf_mat = sklearn.metrics.confusion_matrix(grnd_list, pred_list)
        a = conf_mat.sum(axis=1).reshape((-1,1))
        conf_mat = conf_mat / a  
        if CONFUSION_MAT_KEY in key_set:
            ans_dict[CONFUSION_MAT_KEY] = conf_mat
        if ACCS_KEY in key_set:
            acc_avg_class = 0
            for i in range(conf_mat.shape[0]):
                acc_avg_class += conf_mat[i,i]
            acc_avg_class /= conf_mat.shape[0]
            breakdown_results = shot_acc(pred_list, grnd_list, np.array(new_label_list),
                                        many_shot_thr=get_many_threshold(args), low_shot_thr=get_few_threshold(args),
                                        acc_per_cls=False)
            ans_dict[ACCS_KEY] = (acc_avg_class, breakdown_results)
    if FEATURES_MAT_KEY in key_set:
        features_tensor_list = [None] * model.get_features_n()
        for i in range(model.get_features_n() if out_features_all else 1):
            now_features_list = torch.cat(features_list_list[i])
            features_tensor = torch.zeros_like(now_features_list)
            features_tensor[data_indices_list] = now_features_list
            features_tensor_list[i] = features_tensor
        ans_dict[FEATURES_MAT_KEY] = features_tensor_list if out_features_all else features_tensor_list[0]
    
    if LOSS_KEY in key_set:
        loss_array = torch.cat(loss_list)
        loss_array = loss_array / loss_array.shape[0]
        ans_dict[LOSS_KEY] = loss_array
        

        
    return ans_dict

def calc_forget_scores(tf_tensor: torch.Tensor):
    # tf_tensor [epochN, N]
    forget_tensor_base = tf_tensor[1:, :] - tf_tensor[:-1, :]
    forget_tensor = (forget_tensor_base.abs() - forget_tensor_base) / 2
    forget_count = forget_tensor.sum(dim = 0)
    zero_check = (tf_tensor.abs().sum(dim = 0) == 0) +0
    
    return forget_count + zero_check * tf_tensor.shape[0]
    
    
# features_tensor: [B, D]
# return: [B, B]
def calc_features_cos_sim_mat(features_tensor: torch.Tensor):
    norms_tensor = torch.norm(features_tensor, dim = 1)
    cos_sim_tensor = torch.mm(features_tensor, torch.transpose(features_tensor, 0, 1)) / norms_tensor.unsqueeze(1) / norms_tensor.unsqueeze(0)
    return cos_sim_tensor


import math
def calc_hoyer_sparsity(tensor):
    dims_n = tensor.shape[1]
    l1 = tensor.norm(1, 1)
    l2 = tensor.norm(2, 1)
    return (math.sqrt(dims_n) - (l1 / l2))/(math.sqrt(dims_n) - 1)

# calculate the sparsity indeices for each feature vector
# features_tensor: [B, D]
# return: [B]
def calc_vec_sparsity(features_tensor: torch.Tensor, use_hoyer=True):
    if use_hoyer:
        return calc_hoyer_sparsity(features_tensor.abs())
    else:
        thresholds = features_tensor.abs().mean(1, keepdim=True)
        return (features_tensor.abs() < thresholds).count_nonzero(1) / features_tensor.shape[1]

def calc_features_norm(features: torch.Tensor, labels: torch.Tensor, data_n: int):
    norm_list = [features[labels == i].norm(dim = 1).mean() for i in range(data_n)]
    return torch.stack(norm_list)

def calc_labels_prob_mean(features: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    prob_mat = torch.exp(torch.mm(features, weight.T))
    prob_mat = prob_mat / prob_mat.sum(dim = 1, keepdim=True)
    return prob_mat[torch.arange(features.shape[0]), labels].mean().item()
    