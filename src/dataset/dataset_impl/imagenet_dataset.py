import os
import requests

import torch
import torch.nn.functional as F
from tqdm import tqdm
from urllib.request import urlretrieve
from torchvision import transforms
from torchvision.datasets import ImageNet
from src.skeleton.dataset_skeleton import DatasetOnMemory, LabelType, register_dataset
from torch.utils.data import Dataset, random_split

def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.
    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.
    Returns
    -------
    filename : str
        The location of the downloaded file.
    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)




urls = {'train': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
        'val': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
        'test': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar',
        'dev_kit': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz'}


def download_imagenet2012(root,phase):

    url = urls[phase]
    filename = os.path.basename(url)

    if os.path.exists(os.path.join(root, filename)):
        print(f'{filename} dataset already exists!')
    else:
        download_url(url, destination=os.path.join(root, filename), progress_bar=True)
        print(f'{filename} dataset downloaded!')


@register_dataset("imagenet", LabelType.MULTI_CLASS)
def get_imagenet_dataset(data_dir,*args):

    download_imagenet2012(data_dir,"train") 
    download_imagenet2012(data_dir,"val")
    download_imagenet2012(data_dir,"dev_kit")

    n_class=1000
    train_transform =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(x, n_class))])

    train_dataset = ImageNet(data_dir, split = "train", transform=train_transform, target_transform=target_transform ) 
    #[TODO] I use validation as test because pytorchvision does not has test split, shall I add myself?
    test_dataset = ImageNet(data_dir, split = "val", transform=test_transform, target_transform=target_transform)

    print("spliting test dataset into validation and testing")
    valid_dataset,test_dataset = random_split(test_dataset,
                                       [len(test_dataset) // 4, len(test_dataset) - len(test_dataset) // 4],
                                       generator=torch.Generator().manual_seed(42))
    

    return train_dataset, valid_dataset, test_dataset, None, None, None, n_class

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, _,_, _, n_class =  get_imagenet_dataset("./data")
    print(len(train), len(val), len(test),n_class)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)