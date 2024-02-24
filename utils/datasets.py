#!/usr/bin/env python
# coding: utf-8
import scipy.io
from lib2to3.pytree import Base
import os, pickle, math
from utils.data_units import *
from utils.utils import *
from utils.other_datasets.dataloader import *
from utils.mini_imagenet.create_mini_imagenet import create_images_labels_list, \
                    load_img_num_per_cls, check_and_make_split_text
from utils.conf import *
from matplotlib import pyplot as plt
from os import path
from scipy.io import arff

VALID_SAMPLE_N = 20
CIFAR100_MANY_THRE = 100
CIFAR10_MANY_THRE = 1000
MNIST_MANY_THRE = 1000
HELENA_MANY_THRE = 500
CIFAR100_FEW_THRE =20
CIFAR10_FEW_THRE =200
MNIST_FEW_THRE = 200
HELENA_FEW_THRE = 200

MINI_IMAGENET_LABEL_N = 100
HELENA_LABEL_N = 100
MNIST_LABEL_N = 10
MNIST_VALID_SAMPLE_N = 1000 * MNIST_LABEL_N

TRAIN_DATA_KEY = "train"
VALID_DATA_KEY = "valid"
TEST_DATA_KEY = "test"

ALL_DATA_KEYS_LIST = [TRAIN_DATA_KEY, VALID_DATA_KEY, TEST_DATA_KEY]

PATH_TO_DB_DICT = {CIFAR_100_DATA: 'cifar-100-python', CIFAR_10_DATA: 'cifar-10-batches-py'}
META_FILE_NAME_DICT = {CIFAR_100_DATA: "meta", CIFAR_10_DATA: "batches.meta"}
TEST_FILE_NAME_DICT = {CIFAR_100_DATA: "test", CIFAR_10_DATA: "test_batch"}
LABEL_NAME_KEY_DICT = {CIFAR_100_DATA: b'fine_label_names', CIFAR_10_DATA: b'label_names'}
LABEL_KEY_DICT = {CIFAR_100_DATA: b'fine_labels', CIFAR_10_DATA: b'labels'}
CIFAR_10_TRAIN_DATA_FORMAT = "data_batch_{}"

PATH_TO_DB = "datasets"

def get_many_threshold(args: BaseArgs):
    if args.data.name.startswith(CIFAR_100_DATA):
        return CIFAR100_MANY_THRE
    elif args.data.name.startswith(CIFAR_10_DATA):
        return CIFAR10_MANY_THRE
    elif args.data.name.startswith(MNIST_DATA):
        return MNIST_MANY_THRE
    # elif args.data.name.startswith("ImageNet"):
    elif args.data.name.startswith(MINI_IMAGENET_DATA):
        return CIFAR100_MANY_THRE
    elif args.data.name.startswith(HELENA_DETA):
        return HELENA_MANY_THRE
    elif args.data.name.startswith(IMAGENET_DATA):
        return CIFAR100_MANY_THRE
    else:
        assert False, "Invalid dataset name."

def get_few_threshold(args: BaseArgs):
    if args.data.name.startswith(CIFAR_100_DATA):
        return CIFAR100_FEW_THRE
    elif args.data.name.startswith(CIFAR_10_DATA):
        return CIFAR10_FEW_THRE
    elif args.data.name.startswith(MNIST_DATA):
        return MNIST_FEW_THRE
    elif args.data.name.startswith(MINI_IMAGENET_DATA):
        return CIFAR100_FEW_THRE
    elif args.data.name.startswith(HELENA_DETA):
        return HELENA_FEW_THRE
    elif args.data.name.startswith(IMAGENET_DATA):
        return CIFAR100_FEW_THRE
    else:
        assert False, "Invalid dataset name."

def get_dataloaders_dict(img_arrays_list: list, label_arrays_list: list,
                    label_names_list: list, args: BaseArgs, dataset_class, transpose=True):
    datasets_dict = {}
    for setname, img_list, label_list in zip(ALL_DATA_KEYS_LIST, img_arrays_list,
                                            label_arrays_list):
    
        datasets_dict[setname] = dataset_class(
            image_list=img_list, label_list=label_list, label_names=label_names_list,
            set_name=setname, use_augment=setname==TRAIN_DATA_KEY, transpose = transpose)
        args.logger.info('#examples in {}-set:'.format(setname) +" " + str(datasets_dict[setname].current_set_len))


    dataloaders_dict = {set_name: DataLoader(datasets_dict[set_name],
                                        batch_size=args.data.batch_size,
                                        shuffle=set_name==TRAIN_DATA_KEY, 
                                        num_workers=4) # num_work can be set to batch_size
                for set_name in ALL_DATA_KEYS_LIST} # TRAIN_DATA_KEY,

    args.logger.info('#train batch:' + str(len(dataloaders_dict[TRAIN_DATA_KEY])) + '\t#test batch:' +  str(len(dataloaders_dict[TEST_DATA_KEY])))
    return dataloaders_dict
    
def set_mnist_dataset(args: BaseArgs):
    # place mnist-original.mat here
    assert args.data.imb_factor == 1, "No support for imbalanced MNIST"
    mnist_data_dict = scipy.io.loadmat(os.path.join(".", PATH_TO_DB, "mnist-original.mat"))
    label_names_list = [str(i) for i in range(10)]
    
    from sklearn.model_selection import train_test_split
    data = mnist_data_dict["data"].transpose(1,0).reshape(70000, 28, 28)
    labels = mnist_data_dict["label"].reshape(-1)
    train_images, test_images, train_labels, test_labels =  train_test_split(data, labels,
                        test_size=MNIST_VALID_SAMPLE_N, random_state=0, stratify=labels)
    train_images, valid_images, train_labels, valid_labels =  train_test_split(train_images,
                        train_labels, test_size=MNIST_VALID_SAMPLE_N, random_state=0, stratify=train_labels)
    plt.imshow(train_images[0])
    img_num_per_cls = [(train_labels == i).sum().item() for i in range(MNIST_LABEL_N)]
    
    dataloaders_dict = get_dataloaders_dict([train_images, valid_images, test_images], 
                                            [train_labels, valid_labels, test_labels],
                                            label_names_list, args, MNIST, transpose = False)
    
    return args, dataloaders_dict, train_labels.tolist(), label_names_list, img_num_per_cls



def set_cifar_dataset(args:BaseArgs, show_fig= False, data_key: str = CIFAR_100_DATA, path_to_DB = './datasets'):

    if not os.path.exists(path_to_DB): os.makedirs(path_to_DB)
    _ = torchvision.datasets.CIFAR100(root=path_to_DB, train=True, download=True)
    _ = torchvision.datasets.CIFAR10(root=path_to_DB, train=True, download=True)

    path_to_DB = path.join(path_to_DB,  PATH_TO_DB_DICT[data_key] )


    setname = META_FILE_NAME_DICT[data_key]
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        label_names_list = pickle.load(obj, encoding='bytes')
        label_names_list = label_names_list[LABEL_NAME_KEY_DICT[data_key]]
    for i in range(len(label_names_list)):
        label_names_list[i] = label_names_list[i].decode("utf-8") 
    
        
    if data_key == CIFAR_100_DATA:
        with open(os.path.join(path_to_DB, TRAIN_DATA_KEY), 'rb') as obj:
            database = pickle.load(obj, encoding='bytes') # 50000*3072 array
        img_list = database[b'data'].reshape((database[b'data'].shape[0],3, 32,32))
        label_list = database[LABEL_KEY_DICT[data_key]] # list
        total_num = len(label_list)
    else:
        img_list = np.empty((0,3, 32, 32))
        label_list = []
        for i in range(1, 6):
            with open(os.path.join(path_to_DB, CIFAR_10_TRAIN_DATA_FORMAT.format(i)), 'rb') as obj:
                database = pickle.load(obj, encoding='bytes') # 50000*3072 array
                now_img_list = database[b'data']
                img_list = np.concatenate([img_list, now_img_list.reshape(now_img_list.shape[0], 3, 32, 32)])
                label_list = label_list + database[LABEL_KEY_DICT[data_key]]
        total_num = len(label_list)
    img_num_per_cls = get_img_num_per_cls(args.data.n_classes,
         total_num, args.data.imb_type, args.data.imb_factor, VALID_SAMPLE_N)
    new_img_list_train, new_label_list_train, new_img_list_valid, new_label_list_valid = gen_imbalanced_data(img_num_per_cls, img_list, label_list, VALID_SAMPLE_N)

    setname = TEST_DATA_KEY
    with open(os.path.join(path_to_DB, TEST_FILE_NAME_DICT[data_key]), 'rb') as obj:
        database = pickle.load(obj, encoding='bytes')
    new_img_list_test = database[b'data'].reshape((database[b'data'].shape[0],3, 32,32))
    new_label_list_test = database[LABEL_KEY_DICT[data_key]]


    # ↑done
    dataset_class = CIFAR100LT if data_key == CIFAR_100_DATA else CIFAR10LT
    dataloaders_dict = get_dataloaders_dict( [new_img_list_train, new_img_list_valid, new_img_list_test],
                                               [new_label_list_train, new_label_list_valid, new_label_list_test], 
                                               label_names_list, args, dataset_class)

    if show_fig:
        #preview training data distribution

        plt.plot(img_num_per_cls)
        plt.xlabel('class ID sorted by cardinality')
        plt.ylabel('#training examples')
        plt.show()
        # plt.savefig("test.png")
        # In[8]:


        data_sampler = iter(dataloaders_dict[TRAIN_DATA_KEY])
        data = next(data_sampler)
        _, image_list, label_list = data

        image_list = image_list.to(args.device)
        label_list = label_list.type(torch.long).view(-1).to(args.device)

        # args.logger.info(image_list.shape)

        im_list = image_list.permute(0,2,3,1).cpu().numpy()
        im_list -= im_list.min()
        im_list /= im_list.max()+0.0001
        im_list = createMontage(im_list, (32, 32, 64))

        fig = plt.figure(figsize=(5,5), dpi=95) # better display with larger figure
        plt.imshow(im_list)


        # In[9]:


        data_sampler = iter(dataloaders_dict[TEST_DATA_KEY])
        data = next(data_sampler)
        _, image_list, label_list = data

        image_list = image_list.to(args.device)
        label_list = label_list.type(torch.long).view(-1).to(args.device)

        # args.logger.info(image_list.shape)

        im_list = image_list.permute(0,2,3,1).cpu().numpy()
        im_list -= im_list.min()
        im_list /= im_list.max()+0.0001
        im_list = createMontage(im_list, (32, 32, 64))

        fig = plt.figure(figsize=(5,5), dpi=95) # better display with larger figure
        plt.imshow(im_list)
    return args, dataloaders_dict, new_label_list_train, label_names_list, img_num_per_cls

    
def set_imagenet_dataset(args: BaseArgs):
    phase_names_list = ["train", "val", "test"]
    dataloaders_dict = {phase if phase != "val" else "valid": load_data(data_root=IMAGE_NET_DIR, dataset="ImageNet_LT", phase=phase, 
                                batch_size=args.data.batch_size,
                                num_workers=4) for phase in phase_names_list}
    new_label_list, label_names_list, img_num_per_cls = load_meta_data(args.data.name)
    return args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls

def set_mini_imagenet_dataset(args: BaseArgs):
    check_and_make_split_text(MINI_IMAGENET_SPLIT_DIR, MINI_IMAGENET_META_DIR, args.data.imb_factor,
                              IMAGE_NET_TRAIN)
    
    phase_names_list = ["train", "valid", "test"]
    image_arrays_list = []
    label_arrays_list = []
    train_labels = None
    for phase in phase_names_list:
        imgs_list, labels_list = create_images_labels_list(IMAGE_NET_TRAIN, MINI_IMAGENET_SPLIT_DIR, 
                                                           phase, args.data.imb_factor, about_num = 50000 if args.data.imb_factor == 1.0 else 20000)
        image_arrays_list.append(imgs_list)
        if phase == "train":
            train_labels = labels_list
        label_arrays_list.append(labels_list)
    label_names_list = [str(i) for i in range(MINI_IMAGENET_LABEL_N)]
    dataloaders_dict =  get_dataloaders_dict(image_arrays_list, label_arrays_list,
                                label_names_list, args, MINI_IMAGENET, transpose=False)
    img_num_per_cls = load_img_num_per_cls(MINI_IMAGENET_SPLIT_DIR, args.data.imb_factor)
    return args, dataloaders_dict, train_labels, label_names_list, img_num_per_cls

def set_table_dataset(args: BaseArgs):
    if args.data.name.startswith(HELENA_DETA):
        return set_helena_dataset(args)
    else:
        assert False, "Invalid dataset name."
        
    
def set_helena_dataset(args: BaseArgs):
    dataset, meta = arff.loadarff(os.path.join(PATH_TO_DB, "helena.arff"))
    old_labels_array = np.array([int(data[0]) for data in dataset])
    data_array = np.stack([list(data)[1:] for data in dataset])

    # sort by counts in descending order
    u, counts = np.unique(old_labels_array, return_counts = True)
    sorted_counts = counts[np.argsort(-counts)]
    sorted_labels = u[np.argsort(-counts)]

    # exchange the index and value of np.array
    inv_sorted_labels = np.empty_like(sorted_labels)
    inv_sorted_labels[sorted_labels] = np.arange(len(sorted_labels))

    # soted label ... 0 Most frequent label -> 99 Least frequent label
    labels_array = inv_sorted_labels[old_labels_array]

    # dictionary of label to indices_array
    label_ind_dict = {label: np.where(labels_array == label)[0] for label in range(HELENA_LABEL_N)}
    
    # load indices of validation and test set for HELENA
    with open(os.path.join(PATH_TO_DB, 'helena_valid_indices.json'), 'r') as f:
        valid_indices_dict = json.load(f)
        valid_indices_dict = {int(k): v for k, v in valid_indices_dict.items()}
        valid_labels_array = np.concatenate([labels_array[np.array(valid_indices_dict[label]).astype(int)] for label in range(HELENA_LABEL_N)])
        valid_data_array = data_array[np.concatenate([valid_indices_dict[label] for label in range(HELENA_LABEL_N)]).astype(int)]
    with open(os.path.join(PATH_TO_DB, 'helena_test_indices.json'), 'r') as f:
        test_indices_dict = json.load(f)
        test_indices_dict = {int(k): v for k, v in test_indices_dict.items()}
        test_labels_array = np.concatenate([labels_array[np.array(test_indices_dict[label]).astype(int)] for label in range(HELENA_LABEL_N)])
        test_data_array = data_array[np.concatenate([test_indices_dict[label] for label in range(HELENA_LABEL_N)]).astype(int)]
        
    # data without test and valid
    train_indices_dict = {label: np.setdiff1d(label_ind_dict[label], np.concatenate([valid_indices_dict[label], test_indices_dict[label]])).astype(int).tolist() for label in range(HELENA_LABEL_N)}

    train_labels_array = np.concatenate([labels_array[train_indices_dict[label]] for label in range(HELENA_LABEL_N)])
    train_data_array = np.concatenate([data_array[train_indices_dict[label]] for label in range(HELENA_LABEL_N)])
    
    labels_names_list = [str(i) for i in range(HELENA_LABEL_N)]
    dataloaders_dict = get_dataloaders_dict([train_data_array, valid_data_array, test_data_array],
                                             [train_labels_array, valid_labels_array, test_labels_array],
                                             labels_names_list, args, HELENA)
    
    return args, dataloaders_dict, train_labels_array.tolist(), labels_names_list, sorted_counts.tolist()



# output json file for indices of validation and test set for HELENA
def make_helena_valid_test_data(label_ind_dict):
    # take 40 samples randomly for each label
    valid_test_indices_dict = {label: np.random.choice(label_ind_dict[label], 40, replace=False) for label in range(HELENA_LABEL_N)}
    # split
    valid_indices_dict = {label: valid_test_indices_dict[label][:20].astype(int).tolist() for label in range(HELENA_LABEL_N)}
    test_indices_dict = {label: valid_test_indices_dict[label][20:].astype(int).tolist() for label in range(HELENA_LABEL_N)}
    PATH_TO_DB = "datasets"
    with open(os.path.join(PATH_TO_DB, 'helena_valid_indices.json'), 'w') as f:
        json.dump(valid_indices_dict, f)
    with open(os.path.join(PATH_TO_DB, 'helena_test_indices.json'), 'w') as f:
        json.dump(test_indices_dict, f)
    
def set_dataset(args:BaseArgs, show_fig= False):
    if args.data.name.startswith(CIFAR_100_DATA):
        return set_cifar_dataset(args, show_fig, CIFAR_100_DATA)
    elif args.data.name.startswith(CIFAR_10_DATA):
        return set_cifar_dataset(args, show_fig, CIFAR_10_DATA)
    elif args.data.name.startswith(IMAGENET_DATA):
        return set_imagenet_dataset(args)
    elif args.data.name.startswith(MNIST_DATA):
        return set_mnist_dataset(args)
    elif args.data.name.startswith(MINI_IMAGENET_DATA):
        return set_mini_imagenet_dataset(args)
    else:
        return set_table_dataset(args)


def createMontage(imList, dims, times2rot90=0):
    '''
    imList isi N x HxWx3
    making a montage function to assemble a set of images as a single image to display
    '''
    imy, imx, k = dims
    rows = round(math.sqrt(k))
    cols = math.ceil(k/rows)
    imMontage = np.zeros((imy*rows, imx*cols, 3))
    idx = 0
    
    y = 0
    x = 0
    for idx in range(k):
        imMontage[y*imy:(y+1)*imy, x*imx:(x+1)*imx, :] = imList[idx, :,:,:] #np.rot90(imList[:,:,idx],times2rot90)
        if (x+1)*imx >= imMontage.shape[1]:
            x = 0
            y += 1
        else:
            x+=1
    return imMontage
