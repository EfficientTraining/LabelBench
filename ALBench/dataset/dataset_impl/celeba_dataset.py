from torchvision.datasets import CelebA
from torchvision import transforms
from ALBench.skeleton.dataset_skeleton import register_dataset, LabelType


@register_dataset("celeb", LabelType.MULTI_LABEL)
def get_celeb_dataset(data_dir, *args):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    train_dataset = CelebA(root=data_dir, split="train", target_type="attr", transform=transform, download=True)
    val_dataset = CelebA(root=data_dir, split="valid", target_type="attr", transform=transform, download=True)
    test_dataset = CelebA(root=data_dir, split="test", target_type="attr", transform=transform, download=True)
    return train_dataset, val_dataset, test_dataset, None, None, None, 40


if __name__ == "__main__":
    train, val, test, _, _, _, _ = get_celeb_dataset()
    print(len(train), len(val), len(test))
