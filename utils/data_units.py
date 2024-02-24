import torchvision, torch
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod, abstractclassmethod
import numpy as np
import PIL.Image
from torchvision import datasets, models, transforms
from utils.utils import BaseArgs
CIFAR_10_DATA = "Cifar10"
CIFAR_100_DATA = "Cifar100"
MINI_IMAGENET_DATA = "MiniImagenet"
IMAGENET_DATA = "ImageNet"
MNIST_DATA = "MNIST"
HELENA_DETA = "Helena"


def get_input_dimension(args: BaseArgs):
    name = args.data.name
    if name.startswith(HELENA_DETA):
        return 27
    else:
        assert False, "Invalid dataset name."


DATA_MEAN_DICT = {CIFAR_10_DATA: (0.4914, 0.4822, 0.4465),
                  CIFAR_100_DATA: (0.4914, 0.4822, 0.4465),
                  MINI_IMAGENET_DATA: (0.485, 0.456, 0.406),
                  MNIST_DATA: (0.1301,), 
                  HELENA_DETA: ( 1.58288167e-01,  5.57037048e-01,  3.99761081e-01,  5.00160372e-01,
                        5.10247896e-01,  5.29567537e-01,  3.46977703e-01,  5.36485030e-02,
                        3.62584491e-01,  1.25675640e+02,  1.19772758e+02,  1.12860018e+02,
                        3.66884493e+01,  3.49288424e+01,  3.34216862e+01, -8.20831699e-02,
                    -5.04515470e-02, -3.33100116e-02,  7.04400200e+01,  1.03401015e+00,
                        4.69232939e+00,  1.02238266e+01,  3.69231674e+00,  5.61279166e+00,
                    -7.95518359e-01,  4.93099725e-01,  4.55378998e-01)}
DATA_STD_DICT = {CIFAR_10_DATA: (0.2470, 0.2435, 0.2616),
                 CIFAR_100_DATA: (0.2023, 0.1994, 0.2010),
                 MINI_IMAGENET_DATA: (0.229, 0.224, 0.225), 
                 MNIST_DATA: (0.3069,), 
                 HELENA_DETA: (1.68466041e-01, 3.54006602e-01, 2.42181709e-01, 2.17568431e-01,
                        2.61706411e-01, 3.70982956e-01, 2.23766462e-01, 3.67327475e-02,
                        1.64758695e-01, 5.33314783e+01, 5.47079150e+01, 6.17488870e+01,
                        1.87199115e+01, 1.85985040e+01, 1.89926139e+01, 2.66167363e+00,
                        2.95894681e+00, 3.23702490e+00, 1.52503935e+01, 6.27665329e+00,
                        1.08637107e+01, 5.91986458e+00, 3.69953085e+00, 4.02897586e+00,
                        3.10204107e+00, 1.97985517e+00, 1.89782894e+00)}
class MyDataset(Dataset):
    def __init__(self, set_name='train', image_list=[], label_list=[], label_names=[], use_augment=True, transpose=True):
        self.set_name: str = set_name
        self.use_augment: bool = use_augment
        self.labelNames: list = label_names
        
        self.imageList: list = image_list
        self.labelList: list = label_list
        self.current_set_len: int = len(self.labelList)
        self.transpose: bool = transpose
    
    @property
    def use_augment(self):
        return self.__use_augment
    
    @use_augment.setter
    def use_augment(self, value: bool):
        self.__use_augment = value
        if self.set_name=='train' and self.__use_augment:            
            self.transform = self.get_aug_transform()
        else:
            self.transform = self.get_std_transform()
    
    @abstractclassmethod
    def get_aug_transform(cls):
        raise NotImplementedError()
    
    @abstractclassmethod
    def get_std_transform(cls):
        raise NotImplementedError()
    
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx].astype(np.uint8)
        # print(curImage.shape)
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.fromarray(
            curImage.transpose(1,2,0) if self.transpose else curImage)
        
        curImage = self.transform(curImage)     
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return torch.tensor(idx), curImage, curLabel
    
    # change transform with "with dataset.set_augment():"
    def set_augment(self, value: bool):
        return AugmentChanger(self, self.use_augment, value)

class MyTableDataset(MyDataset):
    def __init__(self, set_name='train', image_list=[], label_list=[], label_names=[], use_augment=True, transpose=True):
        self.set_name: str = set_name
        self.use_augment: bool = use_augment
        self.labelNames: list = label_names
        
        self.imageList: list = image_list
        self.labelList: list = label_list
        self.current_set_len: int = len(self.labelList)
        self.transpose: bool = transpose
    
    @property
    def use_augment(self):
        return self.__use_augment
    
    @use_augment.setter
    def use_augment(self, value: bool):
        self.__use_augment = value
        if self.set_name=='train' and self.__use_augment:            
            self.transform = self.get_aug_transform()
        else:
            self.transform = self.get_std_transform()
    
    @abstractclassmethod
    def get_aug_transform(cls):
        raise NotImplementedError()
    
    @abstractclassmethod
    def get_std_transform(cls):
        raise NotImplementedError()
    
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curData = torch.Tensor(self.imageList[idx])

        curData = self.transform(curData)     
        curLabel =  np.asarray(self.labelList[idx])
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0)
        return torch.tensor(idx), curData, curLabel
    
    # change transform with "with dataset.set_augment():"
    def set_augment(self, value: bool):
        return AugmentChanger(self, self.use_augment, value)



# Class for changing dataset's transform using "with"
class AugmentChanger():
    def __init__(self, dataset: MyDataset, before_use_augment: bool, set_use_augment: bool):
        self.dataset = dataset
        self.before_use_augment = before_use_augment
        self.set_use_augment = set_use_augment
        
    def __enter__(self):
        self.dataset.use_augment = self.set_use_augment
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.dataset.use_augment = self.before_use_augment
        
def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor, valid_sample_n):
    # This function is excerpted and modified from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num -valid_sample_n
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(round(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(round(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(round(img_max * imb_factor))
    else:
        img_num_per_cls.extend([round(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data(img_num_per_cls, img_list, label_list, valid_sample_n):
    # This function is excerpted and modified from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    new_data_train = []
    new_targets_train = []
    new_data_valid = []
    new_targets_valid = []
    targets_np = np.array(label_list, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
        selec_idx_train = idx[:the_img_num]
        selec_idx_val = idx[the_img_num:the_img_num + valid_sample_n]
        new_data_train.append(img_list[selec_idx_train, ...])
        new_data_valid.append(img_list[selec_idx_val, ...])
        new_targets_train.extend([the_class, ] * the_img_num)
        new_targets_valid.extend([the_class, ] * valid_sample_n)
    new_data_train = np.vstack(new_data_train)
    new_data_valid = np.vstack(new_data_valid)
    return (new_data_train, new_targets_train, new_data_valid, new_targets_valid)



class Normalize1d():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def __call__(self, x):
        return (x - self.mean) / self.std

def get_std_transform(key: str):
    if key == MINI_IMAGENET_DATA:
        return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(DATA_MEAN_DICT[key], DATA_STD_DICT[key])
                ])
    else:
        if key == HELENA_DETA:
            return  Normalize1d(DATA_MEAN_DICT[key], DATA_STD_DICT[key])

        else:
            return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(DATA_MEAN_DICT[key], DATA_STD_DICT[key])
                    ])

def get_aug_transform(key: str):
    if key == MINI_IMAGENET_DATA:
        return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEAN_DICT[key], DATA_STD_DICT[key]),
            ])   
    elif key == HELENA_DETA:
        return get_std_transform(key)
    else:
        crop_size = 28 if key == MNIST_DATA else 32
        return transforms.Compose([
                    transforms.RandomCrop(crop_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    get_std_transform(key)
                ])

class CIFAR100LT(MyDataset):
    @classmethod
    def get_aug_transform(cls):
        return get_aug_transform(CIFAR_100_DATA)
    @classmethod
    def get_std_transform(cls):
        return get_std_transform(CIFAR_100_DATA)



class CIFAR10LT(MyDataset):
    @classmethod
    def get_aug_transform(cls):
        return get_aug_transform(CIFAR_10_DATA)
    @classmethod
    def get_std_transform(cls):
        return get_std_transform(CIFAR_10_DATA)
    
class MNIST(MyDataset):
    @classmethod
    def get_aug_transform(cls):
        return get_aug_transform(MNIST_DATA)
    @classmethod
    def get_std_transform(cls):
        return get_std_transform(MNIST_DATA)
    
    
class MINI_IMAGENET(MyDataset):
    @classmethod
    def get_aug_transform(cls):
        return get_aug_transform(MINI_IMAGENET_DATA)
    @classmethod
    def get_std_transform(cls):
        return get_std_transform(MINI_IMAGENET_DATA)
    
class HELENA(MyTableDataset):
    @classmethod
    def get_aug_transform(cls):
        return get_aug_transform(HELENA_DETA)
    @classmethod
    def get_std_transform(cls):
        return get_std_transform(HELENA_DETA)