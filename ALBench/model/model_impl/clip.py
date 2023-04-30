# Code related to zeroshot classifier is mainly ported from
# https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py
import clip
import torch
import torch.nn as nn

from ALBench.skeleton.model_skeleton import register_model
from ALBench.model.model_impl.zero_shot_head import get_zeroshot_classifier


class CLIPVisionOnly(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain, model_name):

        super(CLIPVisionOnly, self).__init__()
        assert pretrain, "CLIPVisionOnly only support pretrain model"

        model, preprocess = clip.load(model_name)
        self.image_encoder_model = model.float() # Convert to float to avoid NAN loss when using AdamW.
        self.embed_dim = model.state_dict()["text_projection"].shape[1]
        self.preprocess_transform = preprocess
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output)
            print("Initialize CLIP_VisionOnly with default linear head. "
                  "Recommend to initialize the head using zero-shot classifier params.")
        else:
            self.classifier = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.image_encoder_model.encode_image(imgs)
        else:
            features = self.image_encoder_model.encode_image(imgs)

        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)

    def init_head_withzeroshot(self, classnames, template):
        self.classifier = get_zeroshot_classifier(self.image_encoder_model, classnames, template)
        print("Update the head initialization using zero-shot classifier params.")

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self, split):
        return self.preprocess_transform


@register_model("clip_vitb16")
def init_clip_viTB16(model_config):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/16")


@register_model("clip_vitb32")
def init_clip_viTB32(model_config):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/32")
