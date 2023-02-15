import os
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.skeleton.dataset_skeleton import register_dataset, LabelType
import pandas as pd
import subprocess


urls = {'classes': "https://storage.googleapis.com/openimages/2018_04/class-descriptions.csv"}

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


#[TODO] some labels are not valid?
# def generate_classes(root,n_class):
#     if not os.path.exists(root):
#         os.mkdir(root)

#     if os.path.exists(os.path.join(root, "class-descriptions.csv")):
#         print(f'class-descriptions already exists!')
#     else:
#         myfile = requests.get(urls['classes'], allow_redirects=True)
#         with open(os.path.join(root, "class-descriptions.csv"), 'wb') as f:
#             f.write(myfile.content)
#         print(f'class-descriptions downloaded!')

#     df = pd.read_csv(os.path.join(root, "class-descriptions.csv"))
#     return df.iloc[10:10+n_class, 1] 

#[TEST]
def generate_classes(root,n_class):
    if not os.path.exists(root):
        os.mkdir(root)

    classes = ["Nut","Apple","Orange","Team"]
    return classes[:n_class]
    
class OID_v4(Dataset):
    def __init__(self, root, classes, transform=None, target_transform=None, phase='train', loadToolDir = "/home/yifang/OIDv4_ToolKit"):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.img_list = []
        self.load_data(loadToolDir)

    def load_data(self,loadToolDir):
        for c in self.classes:

            assert os.path.exists(loadToolDir), "need to download OIDv4_ToolKit first (https://github.com/EscVM/OIDv4_ToolKit)"

            if os.path.exists(os.path.join(self.root, self.phase, c)):
                print("dataset already downloaded ！")
            else: 
                #[TEST] add l00 limit for testing
                print(f"python3 {loadToolDir}/main.py downloader_ill --sub m -y --classes {c} --type_csv {self.phase} --Dataset {self.root} --limit 30")
                runcmd(f"python3 {loadToolDir}/main.py downloader_ill --sub m -y --classes {c} --type_csv {self.phase} --Dataset {self.root} --limit 30", verbose = True)
                print("dataset downloaded ！")

        self.img_list =  ImageFolder(os.path.join(self.root,self.phase), self.transform, self.target_transform)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, target = self.img_list[index]
        return img, target

@register_dataset("OID_v4", LabelType.MULTI_CLASS)
def get_openimage_v4_dataset(n_class,*args):
    classes =generate_classes("./data/OID_v4",n_class)

    train_transform =  transforms.Compose([
        transforms.RandomResizedCrop((224, 224), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5]),
    ]) 

    test_transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5]),
    ]) 

    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])

    train_dataset = OID_v4("./data/OID_v4", classes, phase = "train", transform = train_transform,target_transform=target_transform)
    val_dataset = OID_v4("./data/OID_v4", classes, phase = "validation", transform = test_transform,target_transform=target_transform)
    test_dataset = OID_v4("./data/OID_v4", classes, phase = "test", transform = test_transform,target_transform=target_transform)

    return train_dataset,val_dataset,test_dataset,None,None,None, len(classes)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _ = get_openimage_v4_dataset(2)
    print(len(train), len(val), len(test))
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)



    