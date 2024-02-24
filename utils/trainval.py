import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
from utils.utils import *
from utils.plot_funcs import *
from utils.class_balanced_loss import CB_loss
from utils.regularizers import set_weight_decay, set_norm
from utils.loss import *
from utils.datasets import TRAIN_DATA_KEY, VALID_DATA_KEY, TEST_DATA_KEY
from utils.model_units import MyModel
from torch.utils.tensorboard import SummaryWriter

# get tensorboard's writer
def get_tb_writer(args: BaseArgs):
    writer = SummaryWriter(os.path.join("logs", args.data.name, args.model.name,
                                        args.project_name, args.exp_name))
    return writer

# set model for training or estimation
def set_model(args: BaseArgs, train=True, outFeature=False, train_img_num_per_class = None, load_best = True, get_residual = False):
    assert not (train and get_residual), "No support for get_residual when training"
    base_model: MyModel = get_base_model(args, outFeature=outFeature,
                                         train_img_num_per_class=train_img_num_per_class, get_residual=get_residual)
    
    best_param_path = os.path.join(get_work_dir_path(args), BEST_PARAM_FILE_NAME)
    if os.path.exists(best_param_path) and load_best:
        # This model has already been trained
        state_dict = torch.load(best_param_path, map_location=args.device)
        if(not args.model.load_last_linear):
            for key in base_model.get_last_layer_weights_names_list():
                state_dict[key] = base_model.state_dict()[key]
        base_model.load_state_dict(state_dict)
    else:
        if args.model.is_pretrained:
            if not args.model.load_last_linear: raise NotImplementedError()
            base_model.load_pretrained_model_dict(args)
    base_model.to(args.device)
    # ## Applying Norm
    pgd_func = set_norm(base_model, args)
    
    # for name, module in base_model.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         base_model.state_dict()[name + ".weight"] *= 0.05
        
    if train:
        base_model.train()
    else:
        base_model.eval()
    return base_model, pgd_func, args

CHECK_INTERVAL = 5
def check_statistics_per_epoch(writer: SummaryWriter, model: MyModel, epoch: int, dataloaders_dict: dict,
                               args: BaseArgs, new_label_list: list, img_num_per_class: list,  loss_func):
    if epoch % CHECK_INTERVAL == 0:
        ans_dict_train = process_all_data(dataloaders_dict[TRAIN_DATA_KEY], args, model,
                                          (FEATURES_MAT_KEY, LOSS_KEY, LABEL_KEY),
                                          new_label_list, None, loss_func)
        ans_dict_test = process_all_data(dataloaders_dict[VALID_DATA_KEY], args, model,
                                          (FEATURES_MAT_KEY, CONFUSION_MAT_KEY, ),
                                          new_label_list, None, None)
        per_class_loss_image = plot_per_class_loss(ans_dict_train[LOSS_KEY], ans_dict_train[LABEL_KEY],
                                                   model, img_num_per_class, args, True)
        conf_mat_image = plot_conf_mat(ans_dict_test[CONFUSION_MAT_KEY], args, True)
        fdr_train = calc_fdr(ans_dict_train[FEATURES_MAT_KEY].to(args.device),
                             torch.tensor(dataloaders_dict[TRAIN_DATA_KEY].dataset.labelList).to(args.device),
                             args)
        fdr_test = calc_fdr(ans_dict_test[FEATURES_MAT_KEY].to(args.device),
                             torch.tensor(dataloaders_dict[VALID_DATA_KEY].dataset.labelList).to(args.device),
                             args)
        
        writer.add_scalar("FDR/train", fdr_train, epoch)
        writer.add_scalar("FDR/test", fdr_test, epoch)
        # writer.add_image("loss_per_class", per_class_loss_image, epoch)
        # writer.add_image("conf_mat", conf_mat_image, epoch)
        
        
    
LOG_ADDITIONAL_RES = False
LOG_FDR = True

CHECK_EPOCH_KEY = "epoch"
CHECK_MODEL_KEY = "model"
CHECK_OPT_KEY = "opt"
CHECK_PY_RANDOM_KEY = "py_random"
CHECK_TORCH_RANDOM_KEY = "torch_random"
CHECK_CUDA_RANDOM_KEY = "cuda_random"

def save_check_point(args: BaseArgs, epoch: int, model: MyModel, optimizer: optim.Optimizer):
    check_dict = {}
    check_dict[CHECK_EPOCH_KEY] = epoch
    check_dict[CHECK_MODEL_KEY] = model.state_dict()
    check_dict[CHECK_OPT_KEY] = optimizer.state_dict()
    check_dict[CHECK_PY_RANDOM_KEY] = random.getstate()
    check_dict[CHECK_TORCH_RANDOM_KEY] = torch.random.get_rng_state()
    check_dict[CHECK_CUDA_RANDOM_KEY] = torch.cuda.random.get_rng_state()
    torch.save(check_dict, os.path.join(get_work_dir_path(args), CHECK_POINT_FILE_NAME))

def load_check_point(args: BaseArgs, model: MyModel, optimizer: optim.Optimizer):
    check_dict = torch.load(os.path.join(get_work_dir_path(args), CHECK_POINT_FILE_NAME))
    model.load_state_dict(check_dict[CHECK_MODEL_KEY])
    optimizer.load_state_dict(check_dict[CHECK_OPT_KEY])
    random.setstate(check_dict[CHECK_PY_RANDOM_KEY])
    torch.random.set_rng_state(check_dict[CHECK_TORCH_RANDOM_KEY])
    torch.cuda.random.set_rng_state(check_dict[CHECK_CUDA_RANDOM_KEY])
    return check_dict[CHECK_EPOCH_KEY] + 1

def train_model(args: BaseArgs, dataloaders, new_label_list: list, img_num_per_class: list, model: MyModel, lossFunc, 
                optimizerW, schedulerW, logger, pgdFunc=None,
                num_epochs=50, model_name= 'model', work_dir='./', device='cpu', freqShow=40, clipValue=1, print_each = 1,
                use_features = False, param_search = False, save = True):
    trackRecords = {}
    trackRecords["model_name"] = model_name
    trackRecords['weightNorm'] = []
    trackRecords['acc_test'] = []
    trackRecords['acc_train'] = []
    trackRecords['weights'] = []
    trackRecords[TF_RECORDS_KEY] = []
    trackRecords[LABELS_PROB_MEAN_KEY] = []
    trackRecords[FEATURES_NORM_KEY] = []
    if LOG_FDR:
        trackRecords["fdr_"+TRAIN_DATA_KEY] = []
        trackRecords["fdr_"+VALID_DATA_KEY] = []
        trackRecords["fdr_inner_"+TRAIN_DATA_KEY] = []
        trackRecords["fdr_inner_"+VALID_DATA_KEY] = []
        trackRecords["fdr_inter_"+TRAIN_DATA_KEY] = []
        trackRecords["fdr_inter_"+VALID_DATA_KEY] = []
    
    log_filename = os.path.join(work_dir, TRAIN_LOG_FILE_NAME)    
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.
    best_perClassAcc = 0.0
    
    
    phaseList = [TRAIN_DATA_KEY, VALID_DATA_KEY]
    
    writer = get_tb_writer(args)
    
    if save:
        path_to_save_param = os.path.join(work_dir, BEST_PARAM_FILE_NAME)
        torch.save(model.state_dict(), path_to_save_param)
    
    if args.load_checkpoint:
        start_epoch = load_check_point(args, model, optimizerW)
    else:
        start_epoch = 0
    for epoch in range(start_epoch, num_epochs):  
        if epoch%print_each==0:
            logger.info('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            logger.info('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()


        # Each epoch has a training and validation phase
        for phase in phaseList:
            if epoch%print_each==0:
                logger.info(phase)
            
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == TRAIN_DATA_KEY:
                schedulerW.step()                

            ans_dict = process_all_data(dataloaders[phase], args, model,
                                        (TRAIN_STATIC_KEY, CONFUSION_MAT_KEY, TF_TENSOR_KEY, LABEL_KEY, FEATURES_MAT_KEY) 
                                        if LOG_ADDITIONAL_RES or LOG_FDR else(TRAIN_STATIC_KEY, CONFUSION_MAT_KEY, TF_TENSOR_KEY),
                                        new_label_list, optimizerW if phase == TRAIN_DATA_KEY else None, lossFunc, epoch = epoch)        
            confMat = ans_dict[CONFUSION_MAT_KEY]
            epoch_error, print2screen_avgAccRate = ans_dict[TRAIN_STATIC_KEY]
            
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i,i]
            curPerClassAcc /= confMat.shape[0]
            if epoch%print_each==0:
                logger.info('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            fn = open(log_filename,'a')
            fn.write('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}\n'.format(
                epoch_error, print2screen_avgAccRate, curPerClassAcc))
            fn.close()
            
                
            if phase==TRAIN_DATA_KEY:
                if pgdFunc: # Projected Gradient Descent 
                    pgdFunc.PGD(model)
                      
                trackRecords['acc_train'].append(curPerClassAcc)
                trackRecords[TF_RECORDS_KEY].append(ans_dict[TF_TENSOR_KEY])
                
                ###
                if LOG_ADDITIONAL_RES:
                    trackRecords[FEATURES_NORM_KEY].append(calc_features_norm(ans_dict[FEATURES_MAT_KEY].to(args.device), ans_dict[LABEL_KEY].to(args.device), args.data.n_classes).cpu())
                    trackRecords[LABELS_PROB_MEAN_KEY].append(calc_labels_prob_mean(ans_dict[FEATURES_MAT_KEY].to(args.device), model.get_fc_weight().detach()))
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                W = model.get_fc_weight().cpu().clone()
                tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
                trackRecords['weightNorm'].append(tmp)
                trackRecords['weights'].append(W.detach().cpu().numpy())
                trackRecords[LAST_LOSS] = epoch_error
                trackRecords[LAST_ACC] = print2screen_avgAccRate
            if LOG_FDR:
                features_tensor = ans_dict[FEATURES_MAT_KEY].to(args.device)
                labels_tensor = torch.tensor(dataloaders[phase].dataset.labelList).to(args.device)
                fdr_set = calc_fdr(features_tensor, labels_tensor, args, verbose=True)
                trackRecords["fdr_"+phase].append(fdr_set[0])
                trackRecords["fdr_inner_"+phase].append(fdr_set[3])
                trackRecords["fdr_inter_"+phase].append(fdr_set[4])
                # print(fdr)
                
            if (phase==VALID_DATA_KEY or phase==TEST_DATA_KEY) and curPerClassAcc>best_perClassAcc: #epoch_loss<best_loss:            
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc
                
                trackRecords[BEST_LOSS] = best_loss
                trackRecords[BEST_ACC] = best_acc
                
                if save:
                    path_to_save_param = os.path.join(work_dir, BEST_PARAM_FILE_NAME)
                    torch.save(model.state_dict(), path_to_save_param)
                    
                    file_to_note_bestModel = os.path.join(work_dir, BEST_MODEL_LOG_FILE_NAME)
                    fn = open(file_to_note_bestModel,'a')
                    fn.write('The best model is achieved at epoch-{}: loss{:.5f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}.\n'.format(
                        epoch+1, best_loss, print2screen_avgAccRate, best_perClassAcc))
                    fn.close()
            
        check_statistics_per_epoch(writer, model, epoch, dataloaders, args, new_label_list, img_num_per_class, lossFunc)
        if args.save_checkpoint:
            save_check_point(args, epoch, model, optimizerW)
                
    writer.close()
                
                
    time_elapsed = time.time() - since
    trackRecords['time_elapsed'] = time_elapsed
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
    
    trackRecords[TF_RECORDS_KEY] = torch.stack(trackRecords[TF_RECORDS_KEY]) if trackRecords[TF_RECORDS_KEY] else torch.Tensor()
    if LOG_ADDITIONAL_RES:
        trackRecords[FEATURES_NORM_KEY] = torch.stack(trackRecords[FEATURES_NORM_KEY])
    
    if not param_search:
        torch.save(trackRecords, os.path.join(work_dir, TRACK_RECORDS_FILE_NAME))
        np.save(os.path.join(work_dir, CONFMAT_FILE_NAME), confMat)
    
    return trackRecords

def do_train_cycle(args: BaseArgs, model: MyModel, dataloaders_dict: dict, img_num_per_cls: list, new_label_list: list,
    pgd_func = None, save=True):

    active_layers = model.get_trained_params_list(args)
    if args.project_name == "second":
        for name, param in model.named_parameters(): #freez all model paramters except the classifier layer
            if model.belongs_to_fc_layers(name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.project_name != "first":
        assert False, "Invalid project name"
    
    if args.optim.optimizer == "AdamW":
        assert not args.optim.weight_decay_limit, "No support for weight_decay_limit when using AdamW"
        assert args.optim.weight_decay_bn and args.optim.weight_decay_conv \
            and args.optim.weight_decay_identity_bn\
            ,"AdamW only supports weight decay on all modules"
        optimizer = optim.AdamW([{'params': active_layers, 'lr': args.optim.base_lr}],
            lr=args.optim.base_lr, weight_decay=args.optim.weight_decay)
    elif args.optim.optimizer == "SGD":
        optimizer = optim.SGD([{'params': active_layers, 'lr': args.optim.base_lr}],
            lr=args.optim.base_lr, momentum=0.9)
    else:
        assert False, "Invalid optimizer"

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.optim.total_epoch_num, eta_min=0)
    loss_function, use_features = set_loss(img_num_per_cls, args, reduction="none")

    print_each = 10 #print the accuracy every 10 epochs

    track_records = train_model(args, dataloaders_dict, new_label_list, img_num_per_cls, model, loss_function, optimizer, scheduler, args.logger, pgdFunc=pgd_func,
                            num_epochs=args.optim.total_epoch_num, device = args.device, work_dir=get_work_dir_path(args), 
                            model_name= args.model.name, print_each = print_each, use_features = use_features, save=save)
    return track_records

def check_model_statistic(args: BaseArgs, models_dict: dict, dataloaders_dict: dict,
     new_label_list: list, label_names_list: list, img_num_per_cls, track_records: dict = None, reload_model : bool = True):
    # load model with best epoch accuracy
    path_to_clsnet = os.path.join(get_work_dir_path(args), BEST_PARAM_FILE_NAME)
    model: nn.Module = models_dict[args.exp_name]
    
    if reload_model:
        model.load_state_dict(torch.load(path_to_clsnet, map_location=args.device))
    model.to(args.device)
    model.eval()
    loss_func, use_features = set_loss(img_num_per_cls, args, reduction="none")

    if track_records is None:
        track_records_path = os.path.join(get_work_dir_path(args), TRACK_RECORDS_FILE_NAME)
        if os.path.exists(track_records_path):
            track_records = torch.load(track_records_path)
        else:
            track_records = None
    
    if args.param_search:
        ans_dict_valid = process_all_data(dataloaders_dict[VALID_DATA_KEY], args, model,
                                (ACCS_KEY, ),
                                new_label_list)
        print_and_save_accuracy(ans_dict_valid[ACCS_KEY], args.logger, args)
        return 
            
    ans_dict_test = process_all_data(dataloaders_dict[TEST_DATA_KEY], args, model,
                                (CONFUSION_MAT_KEY, FEATURES_MAT_KEY, ACCS_KEY, LABEL_KEY),
                                new_label_list)
    ans_dict_train = process_all_data(dataloaders_dict[TRAIN_DATA_KEY], args, model,
                                      (LOSS_KEY, LABEL_KEY, FEATURES_MAT_KEY), new_label_list, loss_func=loss_func)
    print_and_save_accuracy(ans_dict_test[ACCS_KEY], args.logger, args)
    conf_mat = ans_dict_test[CONFUSION_MAT_KEY]
    plot_conf_mat(conf_mat, args)
    plot_compressed_conf_mat(conf_mat, img_num_per_cls, args)
    # TODO implement this func for multi-model
    plot_per_class_accuracy_with_naive(conf_mat, label_names_list, img_num_per_cls, args)
    plot_weights_norms(model, label_names_list, args)
    plot_weights_cos_sim(model, args)
    
    train_dataset = dataloaders_dict[TRAIN_DATA_KEY].dataset
    labels_tensor = torch.tensor(train_dataset.labelList).to(args.device)

    if track_records:
        plot_per_class_loss(ans_dict_train[LOSS_KEY], ans_dict_train[LABEL_KEY], model, img_num_per_cls, args)
        plot_per_epoch_accuracy(track_records, args)
        if args.project_name == "first":
            plot_weights_evolution(track_records, args)
        forget_scores = calc_forget_scores(track_records[TF_RECORDS_KEY])
        plot_per_class_forget_scores(forget_scores, labels_tensor, args)
        
        if LOG_ADDITIONAL_RES:
            plot_features_evolution(track_records, args)
            plot_per_epoch_labels_prob_mean(track_records, args)
    
    with OutFeatureChanger(model, True):
        features_tensor = ans_dict_train[FEATURES_MAT_KEY].to(args.device)
        plot_features_cos_sim(features_tensor, labels_tensor, args)
        plot_features_and_weights_cos_sim_and_norm(model, features_tensor, labels_tensor, args)
        plot_features_norms(features_tensor, labels_tensor, label_names_list, args)
        train_fdr = calc_fdr(features_tensor, labels_tensor, args)
        train_sparsity = calc_hoyer_sparsity(features_tensor).mean().item()
        
        test_dataset = dataloaders_dict[TEST_DATA_KEY].dataset
        labels_tensor = torch.tensor(test_dataset.labelList).to(args.device)
        features_tensor = ans_dict_test[FEATURES_MAT_KEY].to(args.device)
        plot_features_cos_sim(features_tensor, labels_tensor, args, "_test")
        plot_features_and_weights_cos_sim_and_norm(model, features_tensor, labels_tensor, args, "_test")
        plot_features_norms(features_tensor, labels_tensor, label_names_list, args, sub_name="_test")
        test_fdr = calc_fdr(features_tensor, labels_tensor, args)
        test_sparsity = calc_hoyer_sparsity(features_tensor).mean().item()
        
        print_and_save_indicator(train_fdr, test_fdr, "FDR", DISCRIMINANT_RATIO_FILE_NAME, args)
        print_and_save_indicator(train_sparsity, test_sparsity, "Sparsity", SPARSITY_FILE_NAME, args)
        
    