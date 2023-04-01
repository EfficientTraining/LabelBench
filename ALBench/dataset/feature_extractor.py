import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_feature_helper(model, dataset, batch_size, num_workers, file_name):
    if os.path.exists(f'{file_name}_features.pt'):
        print(f"Loading features from {file_name}_features.pt")
        features = torch.load(f'{file_name}_features.pt')
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        model.eval()
        features = np.zeros((len(dataset), model.get_embedding_dim()), dtype=float)
        counter = 0

        print(f"Start extracting features... and save to {file_name}_features.pt")
        for img, _, *other in tqdm(loader):
            img = img.float().cuda()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    _, feature = model(img)
            features[counter: (counter + len(feature))] = feature.data.cpu().numpy()
            counter += len(feature)

        torch.save(features, f'{file_name}_features.pt', pickle_protocol=4)
    return features


def get_feature(model_fn, dataset, dataset_split, file_name, embed_model_config, epoch):
    # Load the model.
    model = model_fn(embed_model_config, input_dim=None).cuda()

    if "use_customized_transform" in embed_model_config and embed_model_config["use_customized_transform"]:
        # Get model specific transform of dataset.
        print("Update the transform of dataset to model's special preprocess.")
        transform = model.get_preprocess(split=dataset_split)
        num_transform_seeds = \
            embed_model_config["num_transform_seeds"] if "num_transform_seeds" in embed_model_config else 1

        # Compute the embedding of dataset.
        seed = epoch % num_transform_seeds
        dataset.set_transform(transform, seed=np.random.RandomState(seed).randint(1000000000))
        feat_emb = get_feature_helper(model, dataset, batch_size=embed_model_config["inference_batch_size"],
                                      num_workers=embed_model_config["num_workers"],
                                      file_name=f"{file_name}_{dataset_split}_{seed}")
    else:
        # Compute the embedding of dataset.
        feat_emb = get_feature_helper(model, dataset, batch_size=embed_model_config["inference_batch_size"],
                                      num_workers=embed_model_config["num_workers"],
                                      file_name=f"{file_name}_{dataset_split}")

    return feat_emb
