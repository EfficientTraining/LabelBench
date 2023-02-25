import os
import json
import subprocess
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import pickle
from torchvision import transforms
from src.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType

urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}


def download_coco2014(root, phase):
    work_dir = os.getcwd()
    tmpdir = os.path.join(root, 'tmp/')
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(root, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        # subprocess.Popen('wget ' + urls['annotations'], shell=True)
        subprocess.call('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(root, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    print('[annotation] Done!')

    annotations_data = os.path.join(root, 'annotations')
    anno = os.path.join(root, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(root, 'category.json')):
            json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
    os.chdir(work_dir)


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(self.root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        print('[dataset] COCO2014 classification phase={} number of classes={}  number of images={}'.format(phase,
                                                                                                            self.num_classes,
                                                                                                            len(
                                                                                                                self.img_list)))

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[labels] = 1
        return img, target

@register_dataset("coco", LabelType.MULTI_LABEL)
def get_coco_dataset(n_class,*args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.66, 1.5), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = COCO2014("./data/coco2014", phase="train", transform=train_transform)
    test_dataset = COCO2014("./data/coco2014", phase="val", transform=test_transform)
    valid_dataset,test_dataset = random_split(test_dataset,
                                       [len(test_dataset) // 4, len(test_dataset) - len(test_dataset) // 4],
                                       generator=torch.Generator().manual_seed(42))

    n_class=train_dataset.num_classes

    return train_dataset, valid_dataset, test_dataset, None, None, None, n_class


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, _,_, _, _ = get_coco_dataset(None)
    print(len(train), len(val), len(test))
    loader = DataLoader(train, batch_size=1)
    img, target = next(iter(loader))
    print(img.size())
    print(target)
    print(target.size())
    

#[TESTED]
