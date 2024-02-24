from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import seaborn
from utils.eval_funcs import *
from utils.utils import *
from utils.models import *
from utils.loss import *
from utils.regularizers import set_weight_decay, calc_decayed_labels
from utils.datasets import get_few_threshold, get_many_threshold
from utils.model_units import MyModel
from PIL import Image

TF_RECORDS_KEY = "tf_records"
FEATURES_NORM_KEY = "features_norm"
LABELS_PROB_MEAN_KEY = "maxprob_mean"
BEST_LOSS = "best_loss"
BEST_ACC = "best_acc"
LAST_LOSS = "last_loss"
LAST_ACC = "last_acc"

def save_plot(plot_name, args: BaseArgs, tight=False):
    save_dir = os.path.join(get_work_dir_path(args), "imgs")
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    if tight:
        plt.savefig(os.path.join(save_dir, plot_name+".jpg"), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(os.path.join(save_dir, plot_name+".jpg"))
    plt.show()

# return np.ndarray of the pyplot graph image
def image_plot():
    fig = plt.get_current_fig_manager()
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer.buffer_rgba()).transpose((2, 0, 1))
    # img = np.array(Image.fromarray(im))
    return im

def output_plot(plot_name, args: BaseArgs, tight=False, output_image = False):
    if output_image:
        return image_plot()
    else:
        save_plot(plot_name, args, tight)

def plot_per_epoch_accuracy(trackRecords, args: BaseArgs, output_image: bool = False):
    train_acc = trackRecords['acc_train']
    test_acc = trackRecords['acc_test'] 

    plt.figure()    
    plt.title("Training and validation accuracy per epoch")
    plt.plot(torch.Tensor(train_acc).cpu(), label='Train accuracy')
    plt.plot(torch.Tensor(test_acc).cpu(), label='Validation accuracy')

    plt.xlabel('training epochs')
    plt.ylabel('accuracy')
    plt.legend()
    return output_plot("per_epoch_accuracy", args, output_image=output_image)


def plot_per_epoch_labels_prob_mean(trackRecords, args: BaseArgs, output_image: bool = False):
    labels_prob_mean = trackRecords[LABELS_PROB_MEAN_KEY]

    plt.figure()    
    plt.title("Training probability of gt mean per epoch")
    plt.plot(1-np.array(labels_prob_mean))
    plt.yscale("log")

    plt.xlabel('training epochs')
    plt.ylabel('1 - p_c')
    plt.legend()
    return output_plot("per_epoch_labels_prob_mean", args, output_image=output_image)

def plot_features_evolution(trackRecords, args: BaseArgs, output_image: bool = False):
    # visualizing how norms of per-class weights change in the classifier while training
    W = trackRecords[FEATURES_NORM_KEY].cpu().numpy()
    
    repeat_n = max(W.shape[0] // args.data.n_classes, 1)
    W = np.repeat(W, repeat_n, 1)
    
    label_interval = max(1, args.data.n_classes // 10)
    xticks = [i for i in range(0, args.data.n_classes, label_interval)]

    plt.figure()
    plt.imshow(W, cmap= 'jet', vmin = 0, vmax=W.max())
    plt.colorbar()
    plt.xticks([(tick+0.5)*repeat_n for tick in xticks], [str(tick +1) for tick in xticks])
    plt.xlabel('class ID sorted by cardinality')
    plt.ylabel('training epochs')
    plt.title('per-class mean norms of training features')
    return output_plot("features_evolution", args, output_image=output_image)
 

def plot_weights_evolution(trackRecords, args: BaseArgs, output_image: bool = False):
    # visualizing how norms of per-class weights change in the classifier while training
    W = np.concatenate(trackRecords['weightNorm'])
    W = W.reshape((-1, args.data.n_classes))
    
    repeat_n = W.shape[0] // args.data.n_classes
    W = np.repeat(W, repeat_n, 1)
    
    label_interval = max(1, args.data.n_classes // 10)
    xticks = [i for i in range(0, args.data.n_classes, label_interval)]

    plt.figure()
    plt.imshow(W, cmap= 'jet', vmin = 0, vmax=2)
    plt.colorbar()
    plt.xticks([(tick+0.5)*repeat_n for tick in xticks], [str(tick +1) for tick in xticks])
    plt.xlabel('class ID sorted by cardinality')
    plt.ylabel('training epochs')
    plt.title('norms of per-class weights in the classifier')
    return output_plot("weights_evolution", args, output_image=output_image)
 
def plot_weights_norms(model: MyModel, labelnames,  args: BaseArgs, y_range=None, output_image: bool = False):
    # per-class weight norms vs. class cardinality
    W = model.get_fc_weight().cpu()
    tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
    
    if y_range==None:
        max_val, mid_val, min_val = tmp.max(), tmp.mean(), tmp.min()
        c = min(1/mid_val, mid_val)
        y_range = [min_val-c, max_val+c]
    
    
    fig = plt.figure(figsize=(30,8), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(args.data.n_classes)), labelnames, rotation=90, fontsize=16);  # Set text labels.
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('norm', fontsize=16)
    ax1.set_ylim(y_range)

    plt.subplots_adjust(bottom=0.3)
    plt.plot(tmp, linewidth=2)
    plt.title('norms of per-class weights from the learned classifier vs. class cardinality', fontsize=20)
    return output_plot("weights_norms", args, output_image=output_image)

def __plot_conf_mat__(conf_mat: np.ndarray, args: BaseArgs, sub_name: str="", output_image: bool = False):
    plt.figure(figsize=(20, 20), facecolor="white")
    plt.title(args.exp_name, fontsize=28)
    seaborn.heatmap(conf_mat, square= True, annot=False, cmap="Reds", cbar = True)
    plt.xlabel("Prediction", fontsize=20)
    plt.ylabel("GT", fontsize=20)
    
    return output_plot("conf_mat"+sub_name, args, tight=True, output_image=output_image)
    
def plot_conf_mat(conf_mat: np.ndarray, args: BaseArgs, output_image: bool = False):
    return __plot_conf_mat__(conf_mat, args, output_image=output_image)

def plot_conf_mat_normed(conf_mat: np.ndarray, args: BaseArgs, output_image: bool = False):
    conf_mat_max = conf_mat.max(axis=1, keepdims=True)
    conf_mat = np.concatenate([conf_mat / conf_mat_max, conf_mat_max], axis=1)
    return __plot_conf_mat__(conf_mat, args, "_normed", output_image=output_image)


def plot_per_class_accuracy(label_names_list, exp_names_list, result_dict, img_num_per_cls, args: BaseArgs, output_image: bool = False):
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.subplots_adjust(bottom=0.2, top= 0.95, hspace= 0.25)
    plt.figure(figsize=(30,12), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(args.data.n_classes)), label_names_list, rotation=90, fontsize=14);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images', fontsize=20)
    ax1 = plt.gca()    
    ax2=ax1.twinx()
    for label in exp_names_list:
        ax1.bar(list(range(args.data.n_classes)), result_dict[label], alpha=0.7, width=1, label= label, edgecolor = "black")
        
    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='black')
    ax2.plot(img_num_per_cls, linewidth=4, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=16)
    
    ax1.legend(prop={'size': 14})
    return output_plot("per_class_accuracy", args, output_image=output_image)
    # plt.rcParams['figure.subplot.bottom'] = 0.1


def plot_per_class_accuracy_by_models_dict(models_dict, dataloaders, label_names_list, img_num_per_cls, args: BaseArgs,
     device = 'cuda', conf_mats_dict = None, output_image: bool = False):
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        if conf_mats_dict:
            confMat = conf_mats_dict[label]
        else:
            confMat = get_confusion_matrix(model, dataloaders, device)
        acc_per_class = get_per_class_acc(confMat, n_classes= args.data.n_classes)
        result_dict[label] = acc_per_class
    
    return plot_per_class_accuracy(label_names_list, list(models_dict.keys()), result_dict,
                            img_num_per_cls, args, output_image=output_image)
    
    

def plot_per_class_accuracy_with_naive(conf_mat, label_names_list, img_num_per_cls, args: BaseArgs,
      output_image: bool = False):
    exp_names_list = [args.exp_name, "naive"]
    result_dict = {}
    result_dict[args.exp_name] = conf_mat
    
    if args.exp_name != "naive":    
        naive_args = BaseArgs()
        naive_args.data.name = args.data.name
        naive_args.model.name = args.model.name
        naive_args.project_name = "first"
        naive_args.exp_name = "naive"
        result_dict["naive"] = np.load(os.path.join(get_work_dir_path(naive_args), CONFMAT_FILE_NAME))
    
    result_dict = {key: get_per_class_acc(mat, n_classes= args.data.n_classes) for key, mat in result_dict.items()}
    
    return plot_per_class_accuracy(label_names_list, exp_names_list, result_dict, img_num_per_cls, args, output_image=output_image)
    

def plot_per_class_loss(loss_tensor: torch.Tensor, label_tensor: torch.Tensor, model: MyModel, 
                        img_num_per_cls: list, args: BaseArgs, output_image: bool = False):
    class_n = args.data.n_classes

    loss_per_class = torch.stack([loss_tensor[label_tensor == i].sum() for i in range(class_n)]) 
    
    fc_weights = model.get_fc_weight().data.detach()
    fc_bias = model.get_fc_bias().data.detach()
    weights_per_class = (fc_weights ** 2).sum(axis = 1) / 2 * args.optim.weight_decay
    weights_per_class = weights_per_class + ((fc_bias ** 2) / 2 * args.optim.weight_decay).squeeze()
    weights_per_class =  weights_per_class.cpu()
    decayed_labels_list = calc_decayed_labels(img_num_per_cls, args)
    
    other_weights_list = [(torch.norm(weights, 2) ** 2 / 2).sum() 
                                for name, weights in zip(model.get_trained_params_names_list(args), model.get_trained_params_list(args))
                                if not model.belongs_to_fc_layers(name)]
    if other_weights_list:
        other_weights = torch.stack(other_weights_list).sum() * args.optim.weight_decay
        other_weights_magnitude = other_weights.cpu().item()
    else:
        other_weights_magnitude = 0.
        

    plt.figure()   
    plt.bar(np.arange(0, class_n), loss_per_class, label="Net loss")
    plt.bar(np.arange(0, class_n)[decayed_labels_list],
            -1 * weights_per_class[decayed_labels_list], label = "Weight's norm")
    # if decayed_labels_list.sum() != decayed_labels_list.size:
    #     plt.bar(np.arange(0, args.data.n_classes)[np.logical_not(decayed_labels_list)],
    #         -1 * weights_per_class[np.logical_not(decayed_labels_list)], label = "Weight's norm (not decayed)")
        
    # yticks_array = np.arange(np.floor(-1*weights_per_class.min()*1000)/1000 - 0.0005, 
    #                     np.ceil(loss_per_class.max()*1000)/1000+0.0005, 0.0005)
    locs, _ = plt.yticks()
    plt.yticks(locs, [abs(i) for i in locs])
    plt.xlabel("Category number")
    plt.ylabel("Magnitude")
    plt.text(60, plt.ylim()[1], "Other norm's magnitude: {:.2f}\nSum: {:.2f}".format(other_weights_magnitude, 
                                                                weights_per_class[decayed_labels_list].sum() + loss_per_class.sum()))
    plt.title("Composition of losses", pad = 30.0)
    plt.legend()
    
    return output_plot("per_class_loss", args, output_image=output_image)

# Draw the heatmap of weights' cosine similarities
def plot_weights_cos_sim(model: MyModel, args: BaseArgs, output_image: bool = False):
    weight = model.get_fc_weight().data.detach()
    norm = torch.norm(weight, dim=1)
    cos = torch.mm(weight, torch.transpose(weight, 0, 1)) / norm.unsqueeze(0) / norm.unsqueeze(1)
    
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.title(args.exp_name, fontsize=28)
    seaborn.heatmap(cos.cpu().numpy(), square= True, annot=False, cmap="RdBu_r", cbar = True, vmin = -1.0, vmax=1.0)
    return output_plot("weights_cos_sim", args, tight=True, output_image=output_image)


# get mean cosine similarity and norm's product of  i's class feature vectors and j's class weights
def __get_cos_sim_and_norm_ij__(weights_tensor, feature_tensor, labels_tensor, i, j):
    i_tensor = feature_tensor[labels_tensor == i]
    j_tensor = weights_tensor[j:j+1]
    i_norm = i_tensor.norm(dim=1)
    j_norm = j_tensor.norm(dim=1)
    cos_sim_tensor = torch.mm(i_tensor, torch.transpose(j_tensor, 0, 1)) / i_norm.unsqueeze(1) / j_norm.unsqueeze(0)
    return np.array([cos_sim_tensor.mean().item(), (i_norm * j_norm).mean().item()])


def get_features_cos_sim_ij(feature_tensor, labels_tensor, i, j):
    if i > j:
        return 0
    i_tensor = feature_tensor[labels_tensor == i]
    j_tensor = feature_tensor[labels_tensor == j]
    i_norm = i_tensor.norm(dim=1)
    j_norm = j_tensor.norm(dim=1)
    cos_sim_tensor = torch.mm(i_tensor, torch.transpose(j_tensor, 0, 1)) / i_norm.unsqueeze(1) / j_norm.unsqueeze(0)
    return cos_sim_tensor.mean().item()

def plot_features_cos_sim(features_tensor: torch.Tensor, labels_tensor: torch.Tensor, args: BaseArgs, sub_name=""):
    assert features_tensor.shape[0] == labels_tensor.shape[0], "Features and Labels have different data numbers."
    ans = np.array([ get_features_cos_sim_ij(features_tensor, labels_tensor, i, j) for i in range(args.data.n_classes) for j in range(args.data.n_classes)])
    ans = ans.reshape((args.data.n_classes, args.data.n_classes))
    ans_sym = np.triu(ans) + np.triu(ans).T - np.diag(ans.diagonal())
    
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.title(args.exp_name, fontsize=28)
    seaborn.heatmap(ans_sym, square= True, annot=False, cmap="RdBu_r", cbar = True, vmin = -1.0, vmax=1.0)
    plt.xlabel("Features' class", fontsize=20)
    plt.ylabel("Features' class", fontsize=20)
    return output_plot("features_cos_sim"+sub_name, args, tight=True)

def plot_features_and_weights_cos_sim_and_norm(model: MyModel, features_tensor: torch.Tensor, labels_tensor: torch.Tensor, args: BaseArgs, sub_name=""):
    assert features_tensor.shape[0] == labels_tensor.shape[0], "Features and Labels have different data numbers."
    weights_tensor = model.get_fc_weight().data.detach().to(args.device)
    cos_sim_and_norm = np.stack([__get_cos_sim_and_norm_ij__(weights_tensor, features_tensor, labels_tensor, i, j)
                                 for i in range(args.data.n_classes) for j in range(args.data.n_classes)])
    cos_sim_array = cos_sim_and_norm[:, 0].reshape(args.data.n_classes, args.data.n_classes)
    norm_array = cos_sim_and_norm[:, 1].reshape(args.data.n_classes, args.data.n_classes)
    
    for i, img_name, array in zip(list(range(2)), ["features_weights_cos_sim", "features_weights_norm"], 
                                 [cos_sim_array, norm_array]):    
        plt.figure(figsize=(10, 10), facecolor="white")
        plt.title(args.exp_name, fontsize=28)
        if i==0:
            seaborn.heatmap(array, square= True, annot=False, cmap="RdBu_r", cbar = True, vmin = -1.0, vmax=1.0)
        else:
            seaborn.heatmap(array, square= True, annot=False, cmap="RdBu_r", cbar = True)
        plt.xlabel("Weights' class", fontsize=20)
        plt.ylabel("Features' class", fontsize=20)
        output_plot(img_name+sub_name, args, tight=True)

def plot_compressed_conf_mat(conf_mat: np.ndarray, img_num_per_cls,  args: BaseArgs):
    img_num_per_cls = np.array(img_num_per_cls)
    many_cls_list = img_num_per_cls > get_many_threshold(args)
    few_cls_list = img_num_per_cls < get_few_threshold(args)
    medium_cls_list = np.logical_not(np.logical_or(many_cls_list, few_cls_list))

    mask_list = [many_cls_list, medium_cls_list, few_cls_list]
    
    # compress confusion matrix
    conf_mat = [conf_mat[mask].mean(axis=0) for mask in mask_list]
    conf_mat = np.stack([np.array([conf_list[mask].sum() for mask in mask_list]) for conf_list in conf_mat ])

    label_names_list = ["Many", "Medium", "Few"]
    plt.figure(facecolor="white")
    plt.title(args.exp_name)
    seaborn.heatmap(conf_mat, square= True, annot=True, cmap="Reds",
                    cbar = True, xticklabels=label_names_list, yticklabels=label_names_list,
                    vmin =0, vmax=1)
    plt.xlabel("Prediction")
    plt.ylabel("GT")
    return output_plot("conf_mat_compressed", args, tight=True)

def plot_features_norms(features_tensor: torch.Tensor, labels_tensor: torch.Tensor, labelnames: list,  args: BaseArgs,
                        y_range=None, sub_name: str=""):
    # per-class features mean norms vs. class cardinality
    norms_tensor = torch.norm(features_tensor, dim=1)
    norms_tensor = torch.stack([norms_tensor[i == labels_tensor].mean() for i in range(args.data.n_classes)]).cpu()
    
    if y_range==None:
        max_val, mid_val, min_val = norms_tensor.max(), norms_tensor.mean(), norms_tensor.min()
        c = min(1/mid_val, mid_val)
        y_range = [min_val-c, max_val+c]
    
    
    fig = plt.figure(figsize=(30,8), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(args.data.n_classes)), labelnames, rotation=90, fontsize=16);  # Set text labels.
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('norm', fontsize=16)
    ax1.set_ylim(y_range)

    plt.subplots_adjust(bottom=0.3)
    plt.plot(norms_tensor, linewidth=2)
    plt.title('mean norms of per-class features vs. class cardinality', fontsize=20)
    return output_plot("features_norms"+sub_name, args)
    
def plot_per_class_forget_scores(forget_scores: torch.Tensor, labels_tensor: torch.Tensor, args: BaseArgs, output_image: bool = False):
    class_n = args.data.n_classes

    forget_scores_per_class = torch.stack([forget_scores[labels_tensor == i].mean() for i in range(class_n)]) 

    plt.figure()   
    plt.bar(np.arange(0, class_n), forget_scores_per_class, label="forget score")

    plt.xlabel("Category number")
    plt.ylabel("Forget score")
    plt.title("Forget scores per class", pad = 30.0)
    
    return output_plot("per_class_forget_score", args, output_image=output_image)
