# This file is copied from a publicly available code and modified [commit f50e375, BSD 3-Clause License]:
# # https://github.com/zhmiao/OpenLongTailRecognition-OLTR

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
IMAGENET_METAFILES_DIR = os.path.join(".", "utils", "other_datasets")


# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class AugmentChanger():
    def __init__(self, dataset: Dataset, before_use_augment: bool, set_use_augment: bool):
        self.dataset = dataset
        self.before_use_augment = before_use_augment
        self.set_use_augment = set_use_augment
        
    def __enter__(self):
        self.dataset.use_augment = self.set_use_augment
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.dataset.use_augment = self.before_use_augment

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform_dict=None, get_path=False, use_augment = False):
        self.img_path = []
        self.labelList = []
        self.transform_dict = transform_dict
        self.get_path = get_path
        self.use_augment = use_augment
        txt = os.path.realpath(txt)
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labelList.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labelList)
        
    def __getitem__(self, index):
        # print(index)
        path = self.img_path[index]
        label = self.labelList[index]
        label = torch.tensor([label]).unsqueeze(0)
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        

        if self.use_augment:
            sample = self.transform_dict['train'](sample)
        else:
            sample = self.transform_dict['test'](sample)
            

        if self.get_path:
            return torch.tensor(index), sample, label, path
        else:
            return torch.tensor(index), sample, label
    def set_augment(self, value: bool):
        return AugmentChanger(self, self.use_augment, value)
# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):
    
    txt = IMAGENET_METAFILES_DIR + '/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[phase]

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, data_transforms, use_augment = phase == 'train')
    set_.pin_memory = True

    if phase == 'test' and test_open:
        open_txt = IMAGENET_METAFILES_DIR + '/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset(IMAGENET_METAFILES_DIR + '/%s/%s_open'%(dataset, dataset), open_txt, data_transforms,  use_augment = phase == 'train')
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

def load_meta_data(dataset: str):
    dataset_metafiles_dir = os.path.join(IMAGENET_METAFILES_DIR, dataset+"_LT")
    
    # label_names_list
    labels_txt_path = os.path.join(dataset_metafiles_dir, dataset+"_LT_class_labels.txt")
    use_all_name = False
    if use_all_name:
        def process_line(line: str):
            return line.rstrip(os.linesep)
    else:
        def process_line(line: str):
            return line.split(",")[0].rstrip(os.linesep)
    with open(labels_txt_path, "r") as f:
        label_names_list = [process_line(line) for line in f]
        
    # new_label_list        
    train_txt_path = os.path.join(dataset_metafiles_dir, dataset+"_LT_train.txt")
    with open(train_txt_path, "r") as f:
        new_label_list = [int(line.split(" ")[-1].rstrip(os.linesep)) for line in f ]
    
    # img_num_per_cls
    new_label_array = np.array(new_label_list)
    cls_n = new_label_array.max() + 1
    img_num_per_cls = [np.count_nonzero(new_label_array == i) for i in range(cls_n)]
    
    return new_label_list, label_names_list, img_num_per_cls
    