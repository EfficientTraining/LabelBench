from torchvision.datasets import CelebA
from torchvision import transforms
from LabelBench.skeleton.dataset_skeleton import register_dataset, LabelType, TransformDataset


@register_dataset("celeb", LabelType.MULTI_LABEL)
def get_celeb_dataset(data_dir, *args):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    train_dataset = CelebA(root=data_dir, split="train", target_type="attr", download=True)
    val_dataset = CelebA(root=data_dir, split="valid", target_type="attr", download=True)
    test_dataset = CelebA(root=data_dir, split="test", target_type="attr", download=True)
    return TransformDataset(train_dataset, transform=transform), TransformDataset(val_dataset, transform=transform), \
           TransformDataset(test_dataset, transform=transform), None, None, None, 40, None
    #TODO: add class names


if __name__ == "__main__":
    train, val, test, _, _, _, _, _ = get_celeb_dataset()
    print(len(train), len(val), len(test))
