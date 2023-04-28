# Code related to zeroshot classifier is mainly ported from
# https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py
import open_clip
import torch
import torch.nn as nn

from ALBench.skeleton.model_skeleton import register_model
from ALBench.model.model_impl.zero_shot_head import get_zeroshot_classifier


class COCAVisionOnly(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain):

        super(COCAVisionOnly, self).__init__()
        assert pretrain, "COCAVisionOnly only support pretrain model"
        model, self.train_transform, self.test_transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32",
            pretrained="laion2b_s13b_b90k"
        )
        self.image_encoder_model = model
        self.embed_dim = 512
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output)
        else:
            self.classifier = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.image_encoder_model.encode_image(imgs, normalize=False)
        else:
            features = self.image_encoder_model.encode_image(imgs, normalize=False)

        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)

    def init_head_withzeroshot(self, classnames, template):
        self.classifier = get_zeroshot_classifier(self.image_encoder_model, open_clip.tokenize, classnames, template)
        print("Update the head initialization using zero-shot classifier params.")
        del self.image_encoder_model.text, self.image_encoder_model.text_decoder

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self, split):
        if split == "train":
            return self.train_transform
        else:
            return self.test_transform


@register_model("coca_vitb32")
def init_coca_viTB32(model_config):
    return COCAVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True)
