import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_feature(model, dataset, batch_size, num_workers, file_name, **kwargs):

    if os.path.exists(f'{file_name}_features.pt'):
        print(f"Loading features from {file_name}_features.pt")
        features = torch.load(f'{file_name}_features.pt')
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)
        model.eval()
        features = np.zeros(
            (len(dataset), model.get_embedding_dim()), dtype=float)
        counter = 0

        print(
            f"Start extracting features... and save to {file_name}_features.pt")
        for img, _, *other in tqdm(loader):
            img = img.float().cuda()
            with torch.cuda.amp.autocast():
                _, feature = model(img)
            features[counter: (counter + len(feature))
                     ] = feature.data.cpu().numpy()
            counter += len(feature)

        torch.save(features, f'{file_name}_features.pt')

    return features


def update_embed_dataset(model_fn, dataset, file_name, embed_model_config, **kwargs):

    # Load the model.
    model = model_fn(embed_model_config, input_dim=None).cuda()

    for i, dataset_split in enumerate(["train", "val", "test"]):
        cur_dataset = dataset.get_input_datasets()[i]

        # If clip model, we need to update the transform of dataset.
        if embed_model_config["use_customized_transform"]:
            print("update the transform of dataset for clip special preprocess")
            transform = model.get_preprocess()
            cur_dataset.set_transform(transform)

        # Compute the embedding of dataset and add to dataset.
        feat_emb = get_feature(model, cur_dataset, batch_size=embed_model_config["inference_batch_size"], num_workers=embed_model_config["num_workers"], file_name=f"{file_name}_{dataset_split}")
        dataset.update_emb(feat_emb, dataset_split=dataset_split)

    # Update the input dim of the classifer model as the embedding dim.
    input_dim = feat_emb.shape[1]

    return input_dim
