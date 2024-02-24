# This file is copied from a publicly available code and modified:
# # https://github.com/rahulvigneswaran/TailCalibX

# Imports
import glob, os, json
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Split ratios 
TRAIN_SPLIT_RATIO = 0.8
TEST_SPLIT_RATIO = 1 - TRAIN_SPLIT_RATIO
VAL_SPLIT_RATIO = 0.2
SPLIT_FILE_FORMAT = "{}/{}_{}.txt"
IMG_NUM_FILE_NAME_FORMAT = "{}/{}_num_list.json"

PHASE_LIST = ["train", "val", "test"]

def get_split_file_path(split_dir, imb_ratio, phase):
    return f'{split_dir}/{imb_ratio}_{phase}.txt'
    
# Limiting train, val, test datapoints to 500, 100, 100 per class (Not a very clean code but gets the job done)
def select_random_data(train_x, train_y, count=500):
    train_classwise_dict = {}
    for i, j in zip(train_y, train_x):
        if i in train_classwise_dict.keys():
            train_classwise_dict[i].extend([j])
        else:
            train_classwise_dict[i] = []
            train_classwise_dict[i].extend([j])

    new_train_x = []
    new_train_y = []
    for i in train_classwise_dict.keys():
        ind1 = np.random.permutation(len(train_classwise_dict[i]))[:count]
        new_train_x.append(list(np.array(train_classwise_dict[i])[ind1]))
        new_train_y.append([i]*count)
    return sum(new_train_x, []), sum(new_train_y, [])

# Making Imbalanced train data
def get_img_num_per_cls(cls_num, imb_type, imb_factor, data_length):
    img_max = data_length / cls_num
    img_num_per_cls = []
    if imb_type == "exp":
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == "step":
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(img_num_per_cls, data, targets):
        new_data = []
        new_targets = []
        targets_np = np.array(targets, dtype=np.int64)
        data = np.array(data)
        classes = np.unique(targets_np)

        num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend(data[selec_idx, ...])
            new_targets.extend([the_class,]* the_img_num)
        
        # print(len(new_data[-1]))
        # new_data = np.stack(new_data)
        return new_data, new_targets

def create_mini_imagenet(output, meta_data, imb_ratio, root):
    final = []
    labels = []
    final_1 = []
    labels_1 = []
    final_2 = []
    labels_2 = []

    all_classes = []
    equal_classes = []
    unequal_classes = []

    mini_keys = ['n02110341', 'n01930112', 'n04509417', 'n04067472', 'n04515003', 'n02120079', 'n03924679', 'n02687172', 'n03075370', 'n07747607', 'n09246464', 'n02457408', 'n04418357', 'n03535780', 'n04435653', 'n03207743', 'n04251144', 'n03062245', 'n02174001', 'n07613480', 'n03998194', 'n02074367', 'n04146614', 'n04243546', 'n03854065', 'n03838899', 'n02871525', 'n03544143', 'n02108089', 'n13133613', 'n03676483', 'n03337140', 'n03272010', 'n01770081', 'n09256479', 'n02091244', 'n02116738', 'n04275548', 'n03773504', 'n02606052', 'n03146219', 'n04149813', 'n07697537', 'n02823428', 'n02089867', 'n03017168', 'n01704323', 'n01532829', 'n03047690', 'n03775546', 'n01843383', 'n02971356', 'n13054560', 'n02108551', 'n02101006', 'n03417042', 'n04612504', 'n01558993', 'n04522168', 'n02795169', 'n06794110', 'n01855672', 'n04258138', 'n02110063', 'n07584110', 'n02091831', 'n03584254', 'n03888605', 'n02113712', 'n03980874', 'n02219486', 'n02138441', 'n02165456', 'n02108915', 'n03770439', 'n01981276', 'n03220513', 'n02099601', 'n02747177', 'n01749939', 'n03476684', 'n02105505', 'n02950826', 'n04389033', 'n03347037', 'n02966193', 'n03127925', 'n03400231', 'n04296562', 'n03527444', 'n04443257', 'n02443484', 'n02114548', 'n04604644', 'n01910747', 'n04596742', 'n02111277', 'n03908618', 'n02129165', 'n02981792']

    with open(meta_data + "/all_classes.txt", "r") as f:
        for line in f:
            all_classes.append(str(line.strip()))

    with open(meta_data + "/equal_classes.txt", "r") as f:
        for line in f:
            equal_classes.append(str(line.strip()))

    with open(meta_data + "/unequal_classes.txt", "r") as f:
        for line in f:
            unequal_classes.append(str(line.strip()))

    # filenames = next(walk(root), (None, None, []))[2]  
    img_path_list = glob.glob(root+"/*/*")

    for img_path in img_path_list:
        name = os.path.split(img_path)[-1]
        label_temp = name.split("_")[0]
        if label_temp in all_classes:
            final.append(name)
            labels.append(label_temp)
            if label_temp in equal_classes:
                final_1.append(name)
                labels_1.append(label_temp)
            else:
                final_2.append(name)
                labels_2.append(label_temp)
                


    actual_label = np.unique(labels)
    pseudo_label = np.arange(len(np.unique(labels)))

    # Converts the labels to range of 0 to max ints
    label_dict = {}
    inverse_label_dict = {}

    for i,j in zip(actual_label, pseudo_label):
        label_dict[i] = j
        inverse_label_dict[j] = i
        
    # Re-splitting the mini-imagenet which was made for few-shot into proper train, val, test sets.
    train_x, test_x, train_y, test_y = train_test_split(final_1, labels_1, train_size=TRAIN_SPLIT_RATIO, test_size=TEST_SPLIT_RATIO, stratify=labels_1)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=1-VAL_SPLIT_RATIO, test_size=VAL_SPLIT_RATIO, stratify=train_y)



    train_x, train_y = select_random_data(train_x, train_y, 500)
    val_x, val_y = select_random_data(val_x, val_y, 100)
    test_x, test_y = select_random_data(test_x, test_y, 100)

    # Randomly select and limit datapoints per class from unequal_classes, divide them into train, val, test and append it to the already limited and divided train, val, test of equal_classes
    train_classwise_dict = {}
    for i, j in zip(labels_2, final_2):
        if i in train_classwise_dict.keys():
            train_classwise_dict[i].extend([j])
        else:
            train_classwise_dict[i] = []
            train_classwise_dict[i].extend([j])

    new_train_x = []
    new_train_y = []
    new_val_x = []
    new_val_y = []
    new_test_x = []
    new_test_y = []
    for i in train_classwise_dict.keys():
        ind1 = np.random.permutation(len(train_classwise_dict[i]))[:500]
        ind2 = np.random.permutation(len(train_classwise_dict[i]))[500:600]
        ind3 = np.random.permutation(len(train_classwise_dict[i]))[600:700]
        new_train_x.append(list(np.array(train_classwise_dict[i])[ind1]))
        new_val_x.append(list(np.array(train_classwise_dict[i])[ind2]))
        new_test_x.append(list(np.array(train_classwise_dict[i])[ind3]))
        new_train_y.append([i]*500)
        new_val_y.append([i]*100)
        new_test_y.append([i]*100)

    train_x.extend(sum(new_train_x, []))
    train_y.extend(sum(new_train_y, []))
    val_x.extend(sum(new_val_x, []))
    val_y.extend(sum(new_val_y, []))
    test_x.extend(sum(new_test_x, []))
    test_y.extend(sum(new_test_y, []))


    # print(np.unique(train_y, return_counts=True)[1], len(np.unique(train_y, return_counts=True)[1]))
    # print(np.unique(val_y, return_counts=True)[1], len(np.unique(val_y, return_counts=True)[1]))
    # print(np.unique(test_y, return_counts=True)[1], len(np.unique(test_y, return_counts=True)[1]))



    # Convert WordNetID labels to a range from 0 to 100
    train_y = [label_dict[i] for i in train_y]
    val_y = [label_dict[i] for i in val_y]
    test_y = [label_dict[i] for i in test_y]

    img_num_per_cls = get_img_num_per_cls(100, "exp", imb_ratio, len(train_x))
    with open(IMG_NUM_FILE_NAME_FORMAT.format(output, imb_ratio), "w")as f:
        json.dump(img_num_per_cls, f)
    train_x, train_y = gen_imbalanced_data(img_num_per_cls, train_x, train_y)

    # Writing as txt into "output" dir in inits
    dataxy = [(train_x, train_y), (val_x, val_y), (test_x, test_y)]
    for i, j in enumerate(PHASE_LIST):
        with open(get_split_file_path(output, imb_ratio, j), 'w') as f:
            for line, lab in zip(dataxy[i][0], dataxy[i][1]):
                f.write(line + " " + str(lab))
                f.write('\n')

    for phase in PHASE_LIST:
        finals = []
        labels = []
        input = get_split_file_path(output, imb_ratio, phase)
        with open(input) as f:
            for line in f:
                finals.append(line.split()[0])
                labels.append(line.split()[-1])


        print(np.unique(labels, return_counts=True)[1])
        max_val = max(np.unique(labels, return_counts=True)[1])
        min_val = min(np.unique(labels, return_counts=True)[1])
        sum_val = sum(np.unique(labels, return_counts=True)[1])
        cls_count = len(np.unique(labels, return_counts=True)[1])
        print(f"{phase} -> Max: {max_val} | Min: {min_val} | Sum: {sum_val} | Imb: {max_val/min_val} | Class count: {cls_count}")


def open_image_file(train_dir, file_name):
    label = file_name.split("_")[0]
    image_path = train_dir + "/" + label + "/" + file_name
    with open(image_path, "rb") as f:
        image = np.array(Image.open(f).convert("RGB"))
    return image

def create_images_labels_list(train_dir, split_dir, phase, imb_ratio, about_num = 20000):
    split_text_path = SPLIT_FILE_FORMAT.format(split_dir, imb_ratio, phase)
    images_list = [None] * about_num
    labels_list = [None] * about_num
    count = 0
    with open(split_text_path, "r") as f:
        for line in f:
            if line:
                file_name, label = line.split(" ")
                images_list[count] = open_image_file(train_dir, file_name)
                labels_list[count] = int(label)
                count += 1
    images_list = images_list[:count]
    labels_list = labels_list[:count]
    return images_list, labels_list
    
def load_img_num_per_cls(split_dir, imb_ratio):
    with open(IMG_NUM_FILE_NAME_FORMAT.format(split_dir, imb_ratio), "r")as f:
        ans = json.load(f)
    return ans

def check_and_make_split_text(split_dir, meta_data_dir, imb_ratio, imagenet_root):
    if os.path.isfile(IMG_NUM_FILE_NAME_FORMAT.format(split_dir, imb_ratio)):
        flag = True
        for phase in PHASE_LIST:
            if not os.path.isfile(get_split_file_path(split_dir, imb_ratio, phase)):
                flag = False
                break
        if flag: return
    create_mini_imagenet(split_dir, meta_data_dir, imb_ratio, imagenet_root)