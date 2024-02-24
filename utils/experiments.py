# WDを入れて学習するとcone effectが緩和されるか
import sys
import os
sys.path.append(os.pardir)

from utils.utils import *
from utils.trainval import set_model
from utils.eval_funcs import *
from utils.datasets import set_dataset
from utils.models import OutFeatureChanger
from utils.model_units import ConvBnRelu
from utils.loss import set_loss
import matplotlib.pyplot as plt
import torch
import math, glob
import itertools


class CEExpContainer():
    VAR_KEY = "var"
    MEAN_KEY = "mean"
    STATISTICS_KEY_LIST = [MEAN_KEY, VAR_KEY]
    def __init__(self, cos_keys_list):
        self.cossim = {key1: {key0: [] for key0 in cos_keys_list} for key1 in self.STATISTICS_KEY_LIST}
        self.inner_covnorm = []
        self.intra_covnorm = []
        self.features = {key: [] for key in self.STATISTICS_KEY_LIST}
        self.sparsity = {key: [] for key in self.STATISTICS_KEY_LIST}
        self.relufdr = []
        self.fdr = []
        self.cos_keys_list: list = cos_keys_list
    
    def process(self):
        self.cossim = {key0:{key1:torch.tensor(item1) for key1, item1 in item0.items()} for key0, item0 in self.cossim.items()}
        self.features = {key:torch.tensor(item) for key, item in self.features.items()}
        self.sparsity = {key:torch.tensor(item) for key, item in self.sparsity.items()}
    
    def calc_mean_std(self):
        self.process()
        for dictionary in [self.sparsity, self.features]:
            dictionary[self.VAR_KEY] = (dictionary[self.VAR_KEY].sum().sqrt() / len(dictionary[self.VAR_KEY])).item()
            dictionary[self.MEAN_KEY] = dictionary[self.MEAN_KEY].mean().item()
        dictionary = self.cossim
        for key in self.cos_keys_list:
            dictionary[self.VAR_KEY][key] = (dictionary[self.VAR_KEY][key].sum().sqrt() / len(dictionary[self.VAR_KEY])).item()
            dictionary[self.MEAN_KEY][key] = dictionary[self.MEAN_KEY][key].mean().item()
            


class CEExpContainerDict(dict):
    def __init__(self):
        super().__init__()
        self.cossim = {key:{} for key in CEExpContainer.STATISTICS_KEY_LIST}
        self.features = {key:{} for key in CEExpContainer.STATISTICS_KEY_LIST}
        self.sparsity = {key:{} for key in CEExpContainer.STATISTICS_KEY_LIST}
        self.inner_covnorm = {}
        self.intra_covnorm = {}
        self.relufdr = {}
        self.fdr = {}
    
    def add_container(self, key, cos_keys_list):
        self[key]: CEExpContainer = CEExpContainer(cos_keys_list)
        return self[key]
    
    # rebuild into dicts with exp_name as the key
    def rebuild(self):
        for exp_name in self.keys():
            data_container: CEExpContainer = self[exp_name]
            for key in data_container.cos_keys_list:
                for st_key in CEExpContainer.STATISTICS_KEY_LIST:
                    self.cossim[st_key][exp_name + "_" + key] = data_container.cossim[st_key][key].numpy().tolist()
                    self.features[st_key][exp_name] = data_container.features[st_key].numpy().tolist()
                    self.sparsity[st_key][exp_name] = data_container.sparsity[st_key].numpy().tolist()
            self.relufdr[exp_name] = data_container.relufdr
            self.fdr[exp_name] = data_container.fdr
            self.inner_covnorm[exp_name] = data_container.inner_covnorm
            self.intra_covnorm[exp_name] = data_container.intra_covnorm
                
            
    
        


def shuffle_tensor(tensor: torch.tensor):
    idx = torch.randperm(tensor.nelement())
    return tensor.view(-1)[idx].view(tensor.size())

# shuffle the parameters of conv and bn
def shuffle_parameters(state_dict: dict, model: MyModel, exargs: ExBaseArgs):
    shuf_args = exargs.shuffle
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d) and shuf_args.conv_shuffle:
            state_key = name + ".weight"
            weight = state_dict[state_key]
            data: torch.Tensor = weight.data
            conv_shuf_option = shuf_args.conv_shuffle_option
            if conv_shuf_option == ExShuffleArgs.CONV_ALL_SHUFFLE:
                data = shuffle_tensor(data)
                
            elif conv_shuf_option == ExShuffleArgs.CONV_KERNEL_SHUFFLE:
                data_kernel = data.shape[-1]
                ind_tensor_list = [ torch.randperm(data_kernel**2) + data_kernel**2*i
                        for i in range(data.shape[0]*data.shape[1])]
                data = data.view(-1)[torch.concat(ind_tensor_list)].reshape(data.shape)
                
            elif conv_shuf_option == ExShuffleArgs.CONV_OUTPUT_SHUFFLE:
                ind_tensor= torch.randperm(data.shape[0])
                data = data[ind_tensor, ...]
                
            else:
                assert False, "Invalid conv_shuffle_option"
            weight.data = data
        elif is_batchnorm(module) and shuf_args.bn_shuffle:
            keys_list = [name + ".weight", name + ".bias"]
            for state_key in keys_list:
                weight = state_dict[state_key]
                weight.data = shuffle_tensor(weight.data)
    return state_dict

# getter for pyplot color
def make_color_getter(exp_names_list):
    color_list = ["orange", "green", "blue", "red", "black", "brown", "purple", "grey"]
    def get_color(dict_key: str):
        for i, exp_name in enumerate(exp_names_list):
            if dict_key.startswith(exp_name):
                return color_list[i]
        return "black"
    return get_color

# getter for pyplot style
def make_style_getter(cos_types_list):
    style_list = ["-", "--", ":"]
    def get_style(dict_key: str):
        for i, cos_type in enumerate(cos_types_list):
            if dict_key.endswith(cos_type):
                return style_list[i]
        return "-"
    return get_style    
        
INTRA_KEY = "intra"
INNER_KEY = "inner"
COS_TYPES_LIST = [INTRA_KEY, INNER_KEY]

def calc_inner_intra_cossim(labels: torch.Tensor, feature: torch.Tensor, batch_size: int, data_container: CEExpContainer):
    cos_mat = calc_features_cos_sim_mat(feature)
    wo_diag_mask = torch.tril(torch.ones((batch_size, batch_size), dtype=torch.bool, device = feature.device), -1)
    inner_mask = (labels == labels.T)
    inner_mask = inner_mask * wo_diag_mask
    intra_mask = torch.logical_not(inner_mask) * wo_diag_mask
    for key, mask in zip(COS_TYPES_LIST, [intra_mask, inner_mask]):
        cos_mean = (cos_mat[mask]).sum() / mask.sum()
        cos_mean = cos_mean.item()
        
        cos_2 = (cos_mat - cos_mean) ** 2
        cos_2_mean = ((cos_2[mask]).sum() / mask.sum()).item()
        data_container.cossim[CEExpContainer.MEAN_KEY][key].append(cos_mean)
        data_container.cossim[CEExpContainer.VAR_KEY][key].append(cos_2_mean)
    return data_container

# calculate cosine similarities and other statistics, and output to the dict and list
# - split_cos_type ... calc cos siimilarities split by inner and intra class features.
def calc_cos_stsatistics(features_list, labels, batch_size, split_cos_type, data_container:CEExpContainer,
                         args: BaseArgs, exargs: ExBaseArgs, relu_feature: bool = False):
    for feature in features_list:
        feature = torch.flatten(feature, 1)
        if relu_feature:
            feature = torch.relu(feature)
        if exargs.model.norm_features:
            feature_std = feature.std(0, True)
            feature = (feature - feature.mean(0, True))
            feature = torch.where(feature_std < 1e-5, torch.zeros_like(feature), feature / (feature_std*2 + 1e-6) )
            
        if exargs.model.relu_features:
            feature = torch.relu(feature)
        relufdr= calc_fdr(torch.relu(feature), labels.squeeze(), args)
        fdr, _, _, inner_covnorm, intra_covnorm  = calc_fdr(feature, labels.squeeze(), args, verbose=True)
        if split_cos_type and labels is not None:
            data_container = calc_inner_intra_cossim(labels, feature, batch_size, data_container)
        else:
            cos_mat = calc_features_cos_sim_mat(feature)
            wo_diag_mask = torch.tril(torch.ones((batch_size, batch_size), dtype=torch.bool, device = feature.device), -1)
            cos_mat_wo_diag = cos_mat * wo_diag_mask
            cos_mean = (cos_mat_wo_diag.sum() / (batch_size ** 2 - batch_size) * 2).item()
            
            cos_2 = torch.tril((cos_mat_wo_diag - cos_mean)** 2, -1)
            cos_2_mean = (cos_2.sum() /  (batch_size ** 2 - batch_size) * 2).item()
            data_container.cossim[CEExpContainer.MEAN_KEY][""].append(cos_mean)
            data_container.cossim[CEExpContainer.VAR_KEY][""].append(cos_2_mean)
        
        data_container.features[CEExpContainer.MEAN_KEY].append(feature.mean().item())
        data_container.features[CEExpContainer.VAR_KEY].append(feature.var().item())
        
        sparsity_indices = calc_vec_sparsity(feature)
        data_container.sparsity[CEExpContainer.MEAN_KEY].append(sparsity_indices.mean().item())
        data_container.sparsity[CEExpContainer.VAR_KEY].append(sparsity_indices.var().item())
        
        data_container.relufdr.append(relufdr)
        data_container.fdr.append(fdr)
        data_container.inner_covnorm.append(inner_covnorm)
        data_container.intra_covnorm.append(intra_covnorm)
    
    return data_container


# plot cos_sim_graph and output data_statistics
def output_per_modules_res(exargs:ExBaseArgs, container_dict:CEExpContainerDict, get_color, get_style):
    cos_means_dict = container_dict.cossim[CEExpContainer.MEAN_KEY]
    cos_vars_dict = container_dict.cossim[CEExpContainer.VAR_KEY]
    f_means_dict = container_dict.features[CEExpContainer.MEAN_KEY]
    f_vars_dict = container_dict.features[CEExpContainer.VAR_KEY]
    s_means_dict = container_dict.sparsity[CEExpContainer.MEAN_KEY]
    s_vars_dict = container_dict.sparsity[CEExpContainer.VAR_KEY]
    
    for i, (means_dict, vars_dict, ylabel, metrics) \
        in enumerate(zip([cos_means_dict, f_means_dict, s_means_dict],
                         [cos_vars_dict, f_vars_dict, s_vars_dict],
                         ["Cosine similarity", "Magnitude", "Magnitude"],
                         ["cosine similarity", "features statistics", "sparsity"])):
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        for exp_name, means_list in means_dict.items():
            vars_list = vars_dict[exp_name]
            x = np.arange(len(means_list))
            upper = [mean + math.sqrt(var) for mean, var in zip(means_list, vars_list)]
            lower = [mean +- math.sqrt(var) for mean, var in zip(means_list, vars_list)]
            ax.plot(x, means_list, label = exp_name, color = get_color(exp_name), linestyle=get_style(exp_name))
            ax.fill_between(x, upper, lower, alpha=0.2, color = get_color(exp_name))
        ax.legend(loc="upper left")
        ax.set_xlabel("Modules N")
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.set_ylim(-0.2, 1.0)
        elif i == 1:
            ax.set_ylim(-5.0, 5.0)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_title("Per-module features {} ({})".format(metrics, "image" if exargs.data.use_image else "noise"))
        plt.savefig(os.path.join(get_work_dir_path(exargs), "models_{}.jpg".format(metrics.replace(" ", "_"))),
                    bbox_inches='tight', pad_inches=0)
    
    # fdr, relufdr, inner_intra_covnorm
    for solid_dict, dotted_dict, ylabel, metrics, solid_suffix, dotted_suffix, scale in \
        zip([container_dict.relufdr, container_dict.inner_covnorm],
            [container_dict.fdr, container_dict.intra_covnorm],
            ["FDR", "CovNorm"], ["fdr",  "covnorm"], ["(ReLUFDR)", "(inner)"],
            ["(FDR)", "(intra)"], ["linear", "log"]):

        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
        for exp_name, relues_list in solid_dict.items():
            fdrs_list = dotted_dict[exp_name]
            x = np.arange(len(relues_list))
            ax.plot(x, relues_list, label = exp_name+solid_suffix, color = get_color(exp_name), linestyle=get_style(exp_name))
            ax.plot(x, fdrs_list, label = exp_name+dotted_suffix, color = get_color(exp_name), linestyle=":")
            if metrics == "covnorm":
                divided = np.array(fdrs_list) / np.array(relues_list)
                ax.plot(x, divided,label = exp_name + "_ratio", color = get_color(exp_name), linestyle = "-.")
        ax.legend()
        ax.set_xlabel("Modules N")
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.set_title("Per-module features {} ({})".format(metrics, "image" if exargs.data.use_image else "noise"))
        plt.savefig(os.path.join(get_work_dir_path(exargs), "models_{}.jpg".format(metrics.replace(" ", "_"))),
                    bbox_inches='tight', pad_inches=0)    
    
    
    with open(os.path.join(get_work_dir_path(exargs), "models_per_layer_statistics.json"), "w") as f:
        json.dump({"cos_mean": cos_means_dict, "cos_var": cos_vars_dict, 
                    "f_mean": f_means_dict, "f_var": f_vars_dict,
                    "s_mean": s_means_dict, "s_vars": s_means_dict, 
                    "fdr": container_dict.fdr}, f, indent = 2)

def set_experimental_model(exargs: ExBaseArgs, args: BaseArgs, load_param: bool):
    args.model.load_last_linear = exargs.model.load_last_linear
    base_model, _, args = set_model(args, False, False, load_best = False, get_residual=exargs.get_residual)
    if load_param:
        best_param_path = os.path.join(get_work_dir_path(args), BEST_PARAM_FILE_NAME)
        state_dict = torch.load(best_param_path, map_location=args.device)
        if exargs.shuffle.shuffle_param: state_dict = shuffle_parameters(state_dict, base_model, exargs)
        if exargs.model.shuffle_last_linear:
            for key in base_model.get_last_layer_weights_names_list():
                # shuffled = state_dict[key].data
                # ind_tensor_list = [ torch.randperm(shuffled.shape[-1]) + shuffled.shape[-1]*i
                #         for i in range(shuffled.shape[0])]
                # state_dict[key].data = shuffled.view(-1)[torch.concat(ind_tensor_list)].reshape(shuffled.shape)
                # break
                state_dict[key].data = shuffle_tensor(state_dict[key].data)
            
        if(not args.model.load_last_linear):
            for key in base_model.get_last_layer_weights_names_list():
                state_dict[key] = base_model.state_dict()[key]

    
    if exargs.model.only_first_layer:
        model = ConvBnRelu(3, 64, 3)
        if load_param: model.load_resnet_weight(state_dict)
        model.to(args.device)
    else:
        model = base_model
        if load_param: model.load_state_dict(state_dict)
    
    model.eval()
    return model, args

#######################
## Main
#######################

def check_models_cone_effect_inner(exargs: ExBaseArgs, container_dict: CEExpContainerDict, image_size: int,
                                   cos_keys_list: list, relu_feature: bool = False):
    BATCH_SIZE = exargs.data.batch_size
    data_loder = None
    for i, (exp_name, arg_path, load_param) in enumerate(zip(exargs.model.exp_names_list, exargs.model.args_pathes_list, exargs.model.load_param_tf_list)):
        args = BaseArgs.load(arg_path)
        args = set_device(args)
        args.data.batch_size = BATCH_SIZE
        args.logger.info(" ")
        args.logger.info(arg_path)

        model, args = set_experimental_model(exargs, args, load_param)
        model.fineGrainedFeatures = exargs.model.fine_grained_features
        
        if data_loder is None:
            args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)
            data_loder = dataloaders_dict[exargs.data.data_phase]
        if exargs.data.use_image:
            data_loder_iter = data_loder.__iter__()
        
        data_container = container_dict.add_container(exp_name, cos_keys_list)
        
        if exargs.data.use_image:
            data = next(data_loder_iter)
            images = data[1].to(args.device)
            labels = data[2][:, :,0].to(args.device)
        else:
            images = torch.randn((BATCH_SIZE, 3, image_size, image_size)).to(args.device)
            labels = None
        with torch.no_grad():
            with OutFeatureChanger(model, True, exargs.per_modules):
                if exargs.per_modules:
                    features_list = model(images)
                else:
                    features_list = [model(images)]
                
                data_container = calc_cos_stsatistics(features_list, labels, BATCH_SIZE,
                                                      exargs.split_cos_type, data_container,
                                                      args, exargs, relu_feature)

        # process
        data_container.process()
        
        if not exargs.per_modules:           
            all_mean = torch.mean(data_container.cossim[CEExpContainer.MEAN_KEY][""])
            all_var = torch.mean(data_container.cossim[CEExpContainer.VAR_KEY][""])
            all_std = math.sqrt(all_var)
            print(exp_name)
            print("Mean: {:10.4f}, Std{:10.4f}".format(all_mean, all_std))
            with open(os.path.join(get_work_dir_path(exargs), "models_cone_effect.txt"), "a") as f:
                f.write("{:10} & ${:10.5f}\pm{:10.5f}$\n".format(exp_name, all_mean, all_std))
    return container_dict


def check_models_cone_effect(exargs: ExBaseArgs):

    
    assert not exargs.split_cos_type or exargs.data.use_image, "split_cos_type shuld be false if you use noise as data"
    
    cos_keys_list = COS_TYPES_LIST if exargs.split_cos_type else [""]
    get_color = make_color_getter(exargs.model.exp_names_list)
    get_style = make_style_getter(cos_keys_list)

    CIFAR_100_SIZE = 32
    container_dict = CEExpContainerDict()
    
    container_dict = check_models_cone_effect_inner(exargs, container_dict, CIFAR_100_SIZE, cos_keys_list)
                    
    if exargs.per_modules:
        container_dict.rebuild()
        output_per_modules_res(exargs, container_dict, get_color, get_style)


def calc_cos_mean(features, batch_size):
    cos_mat = calc_features_cos_sim_mat(features)
    cos_mat_wo_diag = torch.tril(cos_mat, -1)
    cos_mean = (cos_mat_wo_diag.sum() / (batch_size ** 2 - batch_size) * 2).item()
    return cos_mean

def make_toy_data(exargs: OneLayerExArgs, ind: int):
    data = None
    if exargs.data_dist[ind] == "gauss":
        data = torch.randn((exargs.batch_size, exargs.d_in), device = exargs.device)
        data =  data * exargs.data_sigma[ind] + exargs.data_mu[ind]
    elif exargs.data_dist[ind] == "spike&slab":
        data = torch.randn((exargs.batch_size, exargs.d_in), device = exargs.device)
        data =  data * exargs.data_sigma[ind] + exargs.data_mu[ind]
        
        mask = torch.bernoulli(torch.ones(
            (exargs.labels_n[ind] ,exargs.d_in),
            device = exargs.device, dtype = torch.float) * exargs.ber_p[ind]
                               ).tile(1, exargs.batch_size // exargs.labels_n[ind]
                                      ).reshape(exargs.batch_size, exargs.d_in)
        data =  data * mask
        
    if exargs.relu_data[ind]:
        data = torch.relu(data)
    return data
    

def make_result_text_for_one_layer(exargs: OneLayerExArgs, ind: int, before_data_container: CEExpContainer,
                                   after_data_container: CEExpContainer):
    
    res_text_formatters_list = ["{:<3}", "before -- ", "after  -- ", "inner ", " : intra ", "Mean: {:6.4f}, Std{:6.4f}"]
    ltx_text_formatters_list = ["{:<10} & ", "b ", "a ", "", "& ", " ${:10.5f}\pm{:10.5f}$ "]
    res_texts_list = []
    for i, formatters in enumerate([res_text_formatters_list, ltx_text_formatters_list]):
        now_res = ""
        for top_element, b_or_a, now_container in \
            zip([ind if i == 0 else exargs.name_list[ind], ""], formatters[1:3],
                [before_data_container, after_data_container]):
            now_res += formatters[0].format(top_element)
            now_res += b_or_a + formatters[3]
            mean_container = now_container.cossim[CEExpContainer.MEAN_KEY]
            var_container = now_container.cossim[CEExpContainer.VAR_KEY]
            now_res += formatters[5].format(mean_container[INNER_KEY],
                                            var_container[INNER_KEY])
            if exargs.labels_n[ind] > 1:
                now_res += formatters[4]
                now_res += formatters[5].format(mean_container[INTRA_KEY],
                                            var_container[INTRA_KEY])
            now_res += "\n"
        res_texts_list.append(now_res)
    return res_texts_list

def check_one_layer_cone_effect_iter(exargs: OneLayerExArgs, i: int, before_data_container: CEExpContainer,
                                     after_data_container: CEExpContainer, use_fdr = False):
    device = exargs.device
    D_IN = exargs.d_in
    D_OUT = exargs.d_out
    STEPS_N = exargs.steps_n
    for step in range(STEPS_N):
        assert (exargs.batch_size % exargs.labels_n[i] == 0), "batch_size should be a multiple of labels_num"
        linear = torch.randn((D_IN, D_OUT), device=device)*exargs.linear_std_list[i] + exargs.linear_mean_list[i]
        linear_ber = torch.bernoulli(torch.ones_like(linear) * exargs.linear_ber_p[i])
        linear *= linear_ber
        bn_weights = torch.randn((1, D_OUT), device=device)*exargs.bn_weights_std_list[i] + exargs.bn_weights_mean_list[i]
        bn_bias = torch.randn((1, D_OUT), device=device)*exargs.bn_bias_std_list[i] + exargs.bn_bias_mean_list[i]
        

        data = make_toy_data(exargs, i)
        labels = torch.arange(exargs.labels_n[i], device=device).unsqueeze(1)\
            .expand(-1,exargs.batch_size // exargs.labels_n[i]).flatten().unsqueeze(0)
        
        features = torch.mm(data, linear)
        if exargs.normalize_list[i]:
            features = (features - features.mean(0, True)) / (features.std(0, True) + 1e-5)
        features = torch.relu(features * bn_weights + bn_bias)
        if step == 0:
            print("zero_elements(%): {:.4}".format(((features.abs() < 1e-3).count_nonzero() / features.shape[0] / features.shape[1]).item()*100))
        # features = torch.mm(data, linear)
        if use_fdr:
            b_fdr_res = calc_fdr(data, labels.squeeze(), verbose=True)
            a_fdr_res = calc_fdr(features, labels.squeeze(), verbose=True)
            before_data_container.fdr.append(b_fdr_res[0])
            after_data_container.fdr.append(a_fdr_res[0])
            before_data_container.inner_covnorm.append(b_fdr_res[3])
            before_data_container.intra_covnorm.append(b_fdr_res[4])
            after_data_container.inner_covnorm.append(a_fdr_res[3])
            after_data_container.intra_covnorm.append(a_fdr_res[4])
        else:
            before_data_container = calc_inner_intra_cossim(labels, data, exargs.batch_size, before_data_container)
            after_data_container = calc_inner_intra_cossim(labels, features, exargs.batch_size, after_data_container)
    
    if use_fdr:
        for container in [before_data_container, after_data_container]:
           container.fdr = sum(container.fdr) / len(container.fdr)
    else:
        for container in [before_data_container, after_data_container]:
            container.calc_mean_std()
        
    return before_data_container, after_data_container


def check_one_layer_cone_effect(exargs: OneLayerExArgs):
    if len(exargs.name_list) == 0:
        exargs.name_list = ["{:2}".format(i) for i in range(len(exargs.linear_mean_list))]
   
    print(exargs.exp_name)
    for i in range(len(exargs.linear_mean_list)):
        before_data_container = CEExpContainer(COS_TYPES_LIST)
        after_data_container = CEExpContainer(COS_TYPES_LIST)
        before_data_container, after_data_container = check_one_layer_cone_effect_iter(
            exargs, i, before_data_container, after_data_container) 
        res_texts_list = make_result_text_for_one_layer(exargs, i, before_data_container, after_data_container)
        print(res_texts_list[0])
        with open(os.path.join(get_work_dir_path(exargs), "layers_cone_effect.txt"), "a") as f:
            f.write(res_texts_list[0]+"\n")
        with open(os.path.join(get_work_dir_path(exargs), "layers_cone_effect_ltx.txt"), "a") as f:
            f.write(res_texts_list[1]+"\n")

def calc_one_layer_loss(before: CEExpContainer, after: CEExpContainer):
    # before_mean = before.cossim[CEExpContainer.MEAN_KEY]
    # after_mean = after.cossim[CEExpContainer.MEAN_KEY]
    
    # inner_loss = before_mean[INNER_KEY] - after_mean[INNER_KEY]
    # inner_loss = inner_loss if inner_loss < 0 else inner_loss *10
    # intra_loss = after_mean[INTRA_KEY] - before_mean[INTRA_KEY]
    # intra_loss = intra_loss if intra_loss < 0 else intra_loss * 10
    # return inner_loss + intra_loss
    b_inner = torch.log(before.inner_covnorm[0])
    b_intra = torch.log(before.intra_covnorm[0])
    a_inner = torch.log(after.inner_covnorm[0])
    a_intra = torch.log(after.intra_covnorm[0])
    print("before: inner {:.4}, intra{:.4}".format(b_inner, b_intra))
    print("after : inner {:.4}, intra{:.4}".format(a_inner, a_intra))
    return a_inner- a_intra - b_inner + b_intra
    
    
# def check_one_layer_cone_effect_iter(device, exp_name, params):
#     D_IN = 512
#     D_OUT = D_IN
#     BATCH_SIZE = 1024
#     STEPS_N = 100
    
#     sigma = 0.01
#     # linear_mean = 0
#     # linear_std = sigma
#     # bn_weights_mean = 0.01
#     # bn_weights_std = sigma
#     # bn_bias_mean = bn_w_mean
#     # bn_bias_std = 0.0
#     linear_mean, linear_std, bn_weights_mean, bn_weights_std, bn_bias_mean, bn_bias_std, \
#         bn_running_mean_mean, bn_running_mean_std, bn_running_std_mean, bn_running_std_std= params
    
    
#     mean_list = []
#     for step in range(STEPS_N):
#         linear = torch.randn((D_IN, D_OUT), device=device)*linear_std + linear_mean
#         bn_weights = torch.randn((1, D_OUT), device=device)*bn_weights_std + bn_weights_mean
#         bn_bias = torch.randn((1, D_OUT), device=device)*bn_bias_std + bn_bias_mean
#         bn_running_mean = torch.randn((1, D_OUT), device=device)*bn_running_mean_std + bn_running_mean_mean
#         bn_running_std = torch.relu(torch.randn((1, D_OUT), device=device)*bn_running_std_std + bn_running_std_mean) + 1e-6
#         # bn_running_mean = 0
#         # bn_running_std = 1
        
#         data = torch.randn((BATCH_SIZE, D_IN), device = device)
#         features = (torch.mm(data, linear) - bn_running_mean) / bn_running_std
#         features = torch.relu(features * bn_weights + bn_bias)
#         # features = torch.mm(data, linear)
#         cos_mat = calc_features_cos_sim_mat(features)
#         cos_mat_wo_diag = torch.tril(cos_mat, -1)
#         cos_mean = (cos_mat_wo_diag.sum() / (BATCH_SIZE ** 2 - BATCH_SIZE) * 2).item()
#         mean_list.append(cos_mean)
        
#     mean_tensor = torch.tensor(mean_list)
#     mean_mean = mean_tensor.mean()
#     mean_std = mean_tensor.std()
#     print(exp_name)
#     print("Mean: {:10.4f}, Std{:10.4f}".format(mean_mean, mean_std))
#     with open(os.path.join(EXP_OUTPUT_DIR, "layers_cone_effect.txt"), "a") as f:
#         f.write("{:10} & ${:10.5f}\pm{:10.5f}$\n".format(exp_name, mean_mean, mean_std))


# def check_one_layer_cone_effect(device):
#     names_list = ["not_trained", "naive", "wd"]
#     params_list = [[0 , 0.02667, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                    [-0.00070, 0.03192, 1.00217, 0.07261, -0.13665, 0.08670, -0.76171, 1.91778, 2.47306, 1.24495],
#                    [-0.00014, 0.00240, 0.06812, 0.06418, -0.00907, 0.03306, -0.00931, 0.01942, 0.01271, 0.02255]]
#     # bn_weights_mean_list = [0.0, 0.01, 1.0]
#     for params, name in zip(params_list, names_list):
#         check_one_layer_cone_effect_iter(device, name, params)
        
def check_multi_layers_cone_effect_iter(device, exp_name, params, layer_n = 1):
    D_IN = D_OUT = 512
    
    BATCH_SIZE = 1024
    STEPS_N = 100
    
    linear_mean, linear_std, bn_weights_mean, bn_weights_std, bn_bias_mean, bn_bias_std= params
    
    
    means_list = []
    for step in range(STEPS_N):
        step_means_list = []
        features = torch.randn((BATCH_SIZE, D_IN), device = device)
        for layer_i in range(layer_n):
            linear = torch.randn((D_IN, D_OUT), device=device)*linear_std + linear_mean
            bn_weights = torch.randn((1, D_OUT), device=device)*bn_weights_std + bn_weights_mean
            bn_bias = torch.randn((1, D_OUT), device=device)*bn_bias_std + bn_bias_mean
            
            features = torch.mm(features, linear)
            features = (features - features.mean(1, keepdim  = True)) / features.std(1, keepdim = True)
            features = torch.relu(features * bn_weights + bn_bias)
            
            cos_mat = calc_features_cos_sim_mat(features)
            cos_mat_wo_diag = torch.tril(cos_mat, -1)
            cos_mean = (cos_mat_wo_diag.sum() / (BATCH_SIZE ** 2 - BATCH_SIZE) * 2).item()
            step_means_list.append(cos_mean)
        means_list.append(torch.tensor(step_means_list))
        
    means_tensor = torch.stack(means_list)
    mean_mean = means_tensor.mean(0).numpy().tolist()
    mean_std = means_tensor.std(0).numpy().tolist()
    print(exp_name)
    return mean_mean, mean_std

def check_multi_layers_cone_effect(device):
    output_dict = {}
    output_dict["description"] = "正規化ありの複数層でBNのweightの相対的な分散が大きくなるとどうなるか(bn_bias_mean=-0.01 bn_bias_std = 0.01)"
    names_list = ["std=0", "std=0.1", "std=0.5", "std=1"]
    params_list = [[0 , 0.01, 0.1, 0.0, 0, 0.01],
                   [0 , 0.01, 0.1, 0.1*0.1, 0, 0.01],
                   [0 , 0.01, 0.1, 0.1*0.5, 0, 0.01],
                   [0 , 0.01, 0.1, 0.1*1.0, 0, 0.01]]
    means_list = []
    stds_list = []
    for params, name in zip(params_list, names_list):
        mean, std = check_multi_layers_cone_effect_iter(device, name, params, 5)
        means_list.append(mean)
        stds_list.append(std)
    
    output_dict["params"] = params_list
    output_dict["names"] = names_list
    output_dict["means"] = means_list
    output_dict["stds"] = stds_list
    
    out_dir = os.path.join(EXP_OUTPUT_DIR, "multi_layers_results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    now_output_file_num = len(glob.glob(os.path.join(out_dir, "*")))
    with open(os.path.join(out_dir, str(now_output_file_num) + ".json"), "w") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

CHECK_GRAD_OUTPUT_DIR = os.path.join(EXP_OUTPUT_DIR, "check_grad")
# calc i'th softmax_outputs
def calc_softmax(features_mat: torch.Tensor, linear_weight: torch.Tensor, ind: int):
    numerator = torch.exp(torch.mm(features_mat, linear_weight[ind:ind+1, :].T))
    denominator = torch.exp(torch.mm(features_mat, linear_weight.T)).sum(1, keepdim=True)
    return numerator / denominator

def calc_unit_softmax_denominator_grad(softmax_features_tensor: torch.Tensor, beta_tensor: torch.Tensor, ind: int):
    beta_tensor = torch.cat((beta_tensor[:ind], beta_tensor[ind + 1:]))[:, None]
    softmax_features_tensor = torch.cat((softmax_features_tensor[:ind, ind, :],
                                      softmax_features_tensor[ind + 1:, ind, :]), 0)
    return torch.sum(softmax_features_tensor * beta_tensor, dim = 0)
    
# check the gradient of the Alshammari's methods
def check_wb_grad(load_finish: True):
    args_path = "./exp/Cifar100/ResNet34/second/MaxNorm_w/0/args.json"
    args: BaseArgs = BaseArgs.load(args_path)
    args.logger.info(sys.version)
    args.logger.info(torch.__version__)
    set_seeds(args.seeds)
    args = set_device(args)
    set_cuda()

    args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)

    model, pgd_func, args = set_model(args, True, train_img_num_per_class = img_num_per_cls, load_best = load_finish)
    
    ans_dict = process_all_data(dataloaders_dict["train"], args, model, (FEATURES_MAT_KEY, ), 
                     new_label_list, None, None)
    linear_weight = model.get_fc_weight() # [C, D]
    
    sorted_labels_list = list(range(args.data.n_classes))
    features_mat_list = [ans_dict[FEATURES_MAT_KEY][torch.tensor(new_label_list) == i].to(args.device)
                                for i in sorted_labels_list]
    beta_tensor = torch.tensor([(1-args.optim.cb_beta) / (1 - args.optim.cb_beta**n) for n in img_num_per_cls], device=args.device)
    
    # make the tensor of \sum_i (softmax_j * \phi x)
    # size is [C(sum_i), C(softmax_j), D]
    softmax_features_tensor = torch.stack([torch.stack(
            [torch.sum(calc_softmax(features_mat_list[i], linear_weight, j)*features_mat_list[i], 0)
             for j in sorted_labels_list]
        ) for i in sorted_labels_list])
    
    numerators_grad = beta_tensor[:, None] \
        * torch.stack([softmax_features_tensor[ind, ind, :] -features_mat.sum(dim=0)
           for ind, features_mat in enumerate(features_mat_list)])
    
    denominators_grad = torch.stack([
        calc_unit_softmax_denominator_grad(softmax_features_tensor, beta_tensor, i)
        for i in sorted_labels_list])
    
    # orthonormalize denominators with respect to the numerators vector
    # denominator = k numerator + l others
    # calc the rate of k/l
    
    unit_numerators_grad = numerators_grad / numerators_grad.norm(dim = 1, keepdim = True)
    denominators_num_component = torch.bmm(denominators_grad.unsqueeze(1), unit_numerators_grad.unsqueeze(2)).squeeze()
    others_denominators_grad = denominators_grad 
    - denominators_num_component[:, None] * unit_numerators_grad
    kl_rate = others_denominators_grad.norm(dim = 1) / denominators_num_component
    
    
    output_dict = {}
    output_dict["lambda"] = args.optim.weight_decay
    output_dict["numerator"] = numerators_grad.norm(dim=1).cpu().numpy().tolist()
    output_dict["denominator"] = denominators_grad.norm(dim=1).cpu().numpy().tolist()
    output_dict["kl_rate"] = kl_rate.cpu().numpy().tolist()

    if not os.path.exists(CHECK_GRAD_OUTPUT_DIR):
        os.makedirs(CHECK_GRAD_OUTPUT_DIR)
    with open(os.path.join(CHECK_GRAD_OUTPUT_DIR, "compare_grad.json"), "w") as f:
        json.dump(output_dict, f)


# only outputs some results of "process_all_data"
LOGIT_MAT_KEY = "logit_matrix"
FEATURES_MEAN_KEY = "features_mean"
def output_processed_data(key: str):
    # args_path = "./exp/Cifar100/ResNet34/second/MaxNorm_w/0/args.json"
    # args_path = "./exp/Cifar100/ResNet34/first/wd_5e-3/0/args.json"
    # args_path = "./exp/Cifar100/ResNet34/first/wd_5e-3_leaky/0/args.json"
    args_path = "./exp/MNIST_Balanced/BNMLP_3_1024/first/wd/0/args.json"
    args: BaseArgs = BaseArgs.load(args_path)
    args.logger.info(sys.version)
    args.logger.info(torch.__version__)
    set_seeds(args.seeds)
    args = set_device(args)
    set_cuda()

    args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)

    model, pgd_func, args = set_model(args, True, train_img_num_per_class = img_num_per_cls, load_best = True)
    model.fineGrainedFeatures = True
    ans_dict = process_all_data(dataloaders_dict["train"], args, model, (FEATURES_MAT_KEY,), 
                     new_label_list, None, None)
    linear_weight = model.get_fc_weight()
    
    res = None
    if key == LOGIT_MAT_KEY:
        res = torch.mm(ans_dict[FEATURES_MAT_KEY].to(args.device), linear_weight.T)
    elif key == FEATURES_MEAN_KEY:
        new_label_tensor = torch.tensor(new_label_list)
        features_mean = []
        for i in range(args.data.n_classes):
            features = ans_dict[FEATURES_MAT_KEY][new_label_tensor == i, :].mean(dim = 0)
            features_mean.append(features)
        res = torch.stack(features_mean)
        
    
    torch.save(res.to("cpu"), os.path.join(EXP_OUTPUT_DIR, "{}.ckpt".format(key)))
    
# calculate the mean of per-class features' mean
def calc_mu_norm():
    args_path = "./exp/Cifar100/ResNet34/first/wd_5e-3/0/args.json"
    args: BaseArgs = BaseArgs.load(args_path)
    args.logger.info(sys.version)
    args.logger.info(torch.__version__)
    set_seeds(args.seeds)
    args = set_device(args)
    set_cuda()
    
    args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)

    model, pgd_func, args = set_model(args, True, train_img_num_per_class = img_num_per_cls, load_best = True)
    ans_dict = process_all_data(dataloaders_dict["train"], args, model, (FEATURES_MAT_KEY,), 
                     new_label_list, None, None)
    
    mu_list = []
    ind = 0
    for i, img_num in enumerate(img_num_per_cls):
        mu_list.append(ans_dict[FEATURES_MAT_KEY][ind: ind+img_num, :].mean(0))
        ind += img_num
    mu_tensor = torch.stack(mu_list, 0)
    mu_mean = mu_tensor.mean(0)
    
    print(mu_mean.norm())
    print(mu_mean)
    
# calculate the gradation of the Weight Balancing (2nd Stage)
def calc_2nd_grad():
    args_path = "./exp/Cifar100/ResNet34/second/MaxNorm_w/0/args.json"
    args: BaseArgs = BaseArgs.load(args_path)
    args.logger.info(sys.version)
    args.logger.info(torch.__version__)
    set_seeds(args.seeds)
    args = set_device(args)
    set_cuda()
    
    args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)

    model, pgd_func, args = set_model(args, True, train_img_num_per_class = img_num_per_cls, load_best = True)
    model.train()
    
    active_layers = model.get_trained_params_list(args)
    if args.project_name == "second":
        for name, param in model.named_parameters(): #freez all model paramters except the classifier layer
            if model.belongs_to_fc_layers(name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.project_name != "first":
        assert False, "Invalid project name"

    optimizer = optim.SGD([{'params': active_layers, 'lr': args.optim.base_lr}],
        lr=args.optim.base_lr, momentum=0.9)
    loss_function, use_features = set_loss(img_num_per_cls, args, reduction="none")
    
    norm_list = []
    for batch_index, sample in enumerate(dataloaders_dict["train"]):                
        data_indices, image_list, label_list = sample
        image_list = image_list.to(args.device)
        label_list = label_list.type(torch.long).view(-1).to(args.device)

        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            with OutFeatureChanger(model, True):
                features = model(image_list)
                logits = model.forward_last_layer(features)

            error = loss_function(logits, label_list, model, features)
            error = error.mean()
            error.backward()
            
            norm_list.append(model.encoder.fc.weight.grad.norm(dim=1).detach())
    print(torch.stack(norm_list, dim = 0).mean())


def iterate_solve(A, B):
    ans = torch.zeros_like(A)
    next_B = B
    for i in range(10):
        solved = torch.linalg.solve(A, next_B)
        ans = ans + solved
        
        residual = B - torch.matmul(A, ans)   
        next_B = residual
        print(torch.norm(residual))
         
def check_iterate_solve():
    matrices = torch.load(os.path.join(EXP_OUTPUT_DIR, "ill_inverse.ckpt"))
    print("cpu")
    iterate_solve(matrices[0].to(torch.float64), matrices[1].to(torch.float64))
    print("gpu")
    iterate_solve(matrices[0].to("cuda", torch.float64), matrices[1].to("cuda", torch.float64))
    

def check_ill_inverse():
    matrices = torch.load(os.path.join(EXP_OUTPUT_DIR, "ill_inverse.ckpt"))
    print("cpu: {}".format(torch.trace(torch.linalg.solve(matrices[0], matrices[1])).item()))
    print("gpu: {}".format(torch.trace(torch.linalg.solve(matrices[0].to("cuda"), matrices[1].to("cuda"))).item()))




def output_mu_feature_per_layers(exargs: ExBaseArgs, file_name: str):

    
    assert exargs.data.use_image, "Use image data"

    BATCH_SIZE = exargs.data.batch_size

    data_loder = None
    data_dict = {}

    for i, (exp_name, arg_path, load_param) in enumerate(zip(exargs.model.exp_names_list, exargs.model.args_pathes_list, exargs.model.load_param_tf_list)):
        args = BaseArgs.load(arg_path)
        args = set_device(args)
        args.data.batch_size = BATCH_SIZE
        args.logger.info(" ")
        args.logger.info(arg_path)

        model, args = set_experimental_model(exargs, args, load_param)
        model.fineGrainedFeatures = exargs.model.fine_grained_features
        
        if data_loder is None:
            args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(args)
            data_loder = dataloaders_dict[exargs.data.data_phase]
        data_loder_iter = data_loder.__iter__()        
        data = next(data_loder_iter)
        images = data[1].to(args.device)
        labels = data[2][:, :,0].to(args.device)[:,0]

        with torch.no_grad():
            with OutFeatureChanger(model, True, exargs.per_modules):
                if exargs.per_modules:
                    features_list = model(images)
                else:
                    features_list = [model(images)]
        features_mean_list = [torch.stack([features[labels == i].mean(0) for i in range(args.data.n_classes)]).cpu() for features in features_list]
        data_dict[args.exp_name] = features_mean_list
    torch.save(data_dict, os.path.join(EXP_OUTPUT_DIR, file_name + ".ckpt"))

def check_dim_increment_effect(device, d_in: int, d_out: int, use_relu: bool = True, N: int = 1024, iter_n: int = 5):
    
    in_cos_sim_list = []
    out_dif_list = []
    for i in range(iter_n):
        in_tensor = torch.relu(torch.randn((N, d_in), device=device))
        linear_tensor = nn.Linear(d_in, d_out, False, device = device)
        out_tensor = linear_tensor(in_tensor)
        if use_relu:
            out_tensor = torch.relu(out_tensor)
        
        in_cos_sim = calc_features_cos_sim_mat(in_tensor)
        out_cos_sim = calc_features_cos_sim_mat(out_tensor)
        out_dif = out_cos_sim # - in_cos_sim
        mask = torch.tril(torch.ones_like(in_cos_sim), -1)
        
        in_cos_sim = in_cos_sim[mask == 1]
        out_dif = out_dif[mask == 1]
        in_cos_sim_list.append(in_cos_sim.detach().cpu().numpy())
        out_dif_list.append(out_dif.detach().cpu().numpy())
    
    plt.scatter(np.concatenate(in_cos_sim_list), np.concatenate(out_dif_list), s = 10)
    relued_text = "_relued" if use_relu else ""
    plt.title(f"D_IN:{d_in}, D_OUT: {d_out} {relued_text}")
    plt.savefig(f"./exp_theory/dim_inc_imgs/cos_sim_change_{d_in}_{d_out}{relued_text}.png")
    
        
    
    
    
    
