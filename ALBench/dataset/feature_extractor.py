import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import random
from torchvision import transforms

from ALBench.dataset.rand_augment import RandAugment


def make_semi_transforms(transform):
    transform_weak = transform

    # instantiate transform_strong as a copy of transform
    transform_strong = copy.deepcopy(transform)

    for i, t in enumerate(transform_strong.transforms):
        if isinstance(t, transforms.ToTensor):
            # based on strong augmentation used in https://github.com/microsoft/Semi-supervised-learning
            transform_strong.transforms.insert(i, RandAugment(3, 5))

    return transform_weak, transform_strong

# TODO(Greg): handle dataset_split checking


def get_feature_helper(model_fn, embed_model_config, dataset, seed, batch_size, num_workers, file_name, dataset_split,
                       use_strong):

    # Override use_strong if dataset_split is not train.
    use_strong = use_strong and dataset_split == "train"

    if use_strong and os.path.exists(f'{file_name}_features_strong.pt'):
        print(f"Loading features from {file_name}_features_strong.pt")
        features = torch.load(f'{file_name}_features_strong.pt')
    elif (not use_strong) and os.path.exists(f'{file_name}_features.pt'):
        print(f"Loading features from {file_name}_features.pt")
        features = torch.load(f'{file_name}_features.pt')
    else:
        # Load the model.
        model = model_fn(embed_model_config).cuda()

        # Get model specific transform of dataset.
        if "use_customized_transform" in embed_model_config and embed_model_config["use_customized_transform"]:
            print("Update the transform of dataset to model's special preprocess.")
            transform = model.get_preprocess(split=dataset_split)
            dataset.set_transform(transform)
        else:
            transform = dataset.get_transform()

        if use_strong:
            transform_weak, transform_strong = make_semi_transforms(transform)
            assert transform_strong is not None, "Strong transform cannot be None"
            dataset.set_transform(transform_weak)
            dataset.set_strong_transform(transform_strong)

        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)
        model.eval()
        features = np.zeros(
            (len(dataset), model.get_embedding_dim()), dtype=float)
        if use_strong:
            features_strong = np.zeros(
                (len(dataset), model.get_embedding_dim()), dtype=float)
        counter = 0

        strong_str = "_strong" if use_strong else ""
        print(
            f"Start extracting features... and save to {file_name}_features{strong_str}.pt")
        if seed is not None:
            random.seed(random.RandomState(seed).randint(1000000000))
        for img, _, *other in tqdm(loader):
            if use_strong:
                img, img_strong = img
                img_strong = img_strong.float().cuda()
            img = img.float().cuda()
            with torch.cuda.amp.autocast(), torch.no_grad():
                _, feature = model(img)
                if use_strong:
                    _, feature_strong = model(img_strong)
            features[counter: (counter + len(feature))
                     ] = feature.data.cpu().numpy()
            if use_strong:
                features_strong[counter: (
                    counter + len(feature))] = feature_strong.data.cpu().numpy()
            counter += len(feature)

        if use_strong:
            torch.save((features, features_strong),
                       f'{file_name}_features_strong.pt', pickle_protocol=4)
            features = (features, features_strong)
        else:
            torch.save(features, f'{file_name}_features.pt', pickle_protocol=4)
    return features


def get_feature(model_fn, dataset, dataset_split, file_name, embed_model_config, epoch, use_strong):
    if "use_customized_transform" in embed_model_config and embed_model_config["use_customized_transform"]:
        num_transform_seeds = \
            embed_model_config["num_transform_seeds"] if "num_transform_seeds" in embed_model_config else 1

        # Compute the embedding of dataset using customized transformer.
        seed = epoch % num_transform_seeds
        feat_emb = get_feature_helper(model_fn, embed_model_config, dataset, seed,
                                      batch_size=embed_model_config["inference_batch_size"],
                                      num_workers=embed_model_config["num_workers"],
                                      file_name=f"{file_name}_{dataset_split}_{seed}",
                                      dataset_split=dataset_split,
                                      use_strong=use_strong)
    else:
        # Compute the embedding of dataset without model specific data augmentation.
        feat_emb = get_feature_helper(model_fn, embed_model_config, dataset, None,
                                      batch_size=embed_model_config["inference_batch_size"],
                                      num_workers=embed_model_config["num_workers"],
                                      file_name=f"{file_name}_{dataset_split}",
                                      dataset_split=dataset_split,
                                      use_strong=use_strong)
    return feat_emb
