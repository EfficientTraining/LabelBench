import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import random
from torchvision import transforms

from LabelBench.dataset.rand_augment import RandAugment


class FeatureExtractor:
    def __init__(self, model_fn, file_name, embed_model_config):

        self.model = None
        self.model_fn = model_fn
        self.embed_model_config = embed_model_config
        self.file_name = file_name
        self.batch_size = \
            embed_model_config["inference_batch_size"] if "inference_batch_size" in embed_model_config else 128
        self.num_workers = embed_model_config["num_workers"] if "num_workers" in embed_model_config else 4

        self.precomputed_features = {"train": None, "val": None, "test": None}
        self.num_transform_seeds = \
            embed_model_config["num_transform_seeds"] if "num_transform_seeds" in embed_model_config else 1

    def get_feature(self, dataset, dataset_split, epoch, use_strong):
        use_features_on_memory = (dataset_split != "train") or (self.num_transform_seeds == 1)

        # Only use strong transform on train set.
        use_strong = use_strong and (dataset_split == "train")

        if use_features_on_memory and self.precomputed_features[dataset_split] is not None:
            return self.precomputed_features[dataset_split]
        else:
            seed = epoch % self.num_transform_seeds
            if use_strong:
                filename = f'{self.file_name}_{seed}_features_strong.pt'
            else:
                filename = f'{self.file_name}_{seed}_features_{dataset_split}.pt'
            if os.path.exists(filename):
                print(f"Loading features from {filename}")
                features = torch.load(filename)
            else:
                print(f"Start extracting features... and save to {filename}")
                features = self.get_feature_helper(dataset, use_strong, dataset_split, seed=seed)
                torch.save(features, filename, pickle_protocol=4)

            if use_features_on_memory:
                self.precomputed_features[dataset_split] = features
            return features

    def get_feature_helper(self, dataset, use_strong, dataset_split, seed=None):
        if self.model is None:
            self.model = self.model_fn(self.embed_model_config).cuda()

        # Get model specific transform of dataset.
        print("Update the transform of dataset to model's special preprocess.")
        transform = self.model.get_preprocess(split=dataset_split)
        dataset.set_transform(transform)
        dataset.set_strong_transform(None)

        if use_strong:
            transform_weak, transform_strong = make_semi_transforms(transform)
            assert transform_strong is not None, "Strong transform cannot be None"
            dataset.set_transform(transform_weak)
            dataset.set_strong_transform(transform_strong)

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.model.eval()
        features = np.zeros((len(dataset), self.model.get_embedding_dim()), dtype=float)
        if use_strong:
            features_strong = np.zeros((len(dataset), self.model.get_embedding_dim()), dtype=float)

        counter = 0
        if seed is not None:
            random.seed(random.RandomState(seed).randint(1000000000))
        for img, _, *other in tqdm(loader):
            if use_strong:
                img, img_strong = img
                img_strong = img_strong.float().cuda()
            img = img.float().cuda()
            with torch.cuda.amp.autocast(), torch.no_grad():
                _, feature = self.model(img)
                if use_strong:
                    _, feature_strong = self.model(img_strong)
            features[counter: (counter + len(feature))] = feature.data.cpu().numpy()
            if use_strong:
                features_strong[counter: (counter + len(feature))] = feature_strong.data.cpu().numpy()
            counter += len(feature)

        if use_strong:
            features = (features, features_strong)

        return features

def make_semi_transforms(transform):
    transform_weak = transform

    # instantiate transform_strong as a copy of transform
    transform_strong = copy.deepcopy(transform)

    for i, t in enumerate(transform_strong.transforms):
        if isinstance(t, transforms.ToTensor):
            # based on strong augmentation used in https://github.com/microsoft/Semi-supervised-learning
            transform_strong.transforms.insert(i, RandAugment(3, 5))
            break

    return transform_weak, transform_strong
