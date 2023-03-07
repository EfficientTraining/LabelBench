import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from ALBench.skeleton.model_skeleton import register_model


class CLIP_VisionOnly(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain, model_name):

        super(CLIP_VisionOnly, self).__init__()
        assert pretrain, "CLIP_VisionOnly only support pretrain model"

        model, preprocess = clip.load(model_name)
        self.image_encoder_model = model
        self.embed_dim = model.state_dict()["text_projection"].shape[1]
        self.preprocess_transform = preprocess
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifer = nn.Linear(self.embed_dim, num_output)
        else:
            self.classifer = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=True):
        if freeze:
            with torch.no_grad():
                features = self.image_encoder_model.encode_image(imgs)
        else:
            features = self.image_encoder_model.encode_image(imgs)

        if ret_features:
            return self.classifer(features), features.data
        elif self.ret_emb:
            return self.classifer(features), features
        else:
            return self.classifer(features)

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self):
        return self.preprocess_transform


@register_model("clip_vitb16")
def init_clip_viTB16(model_config, input_dim=None):
    return CLIP_VisionOnly(model_config["num_output"],
                           ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                           pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                           model_name="ViT-B/16")


@register_model("clip_vitb32")
def init_clip_viTB32(model_config, input_dim=None):
    return CLIP_VisionOnly(model_config["num_output"],
                           ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                           pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                           model_name="ViT-B/32")
