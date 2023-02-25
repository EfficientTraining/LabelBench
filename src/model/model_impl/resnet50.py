import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from src.skeleton.model_skeleton import register_model


class Resnet50(nn.Module):
    model_name = "resnet50"

    def __init__(self, num_output, ret_emb, pretrain):
        super(Resnet50, self).__init__()
        if pretrain:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet50()
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(2048, num_output)
        self.ret_emb = ret_emb
        self.num_output = num_output

    def forward(self, imgs, ret_features=False):
        features = self.model(imgs)
        features = torch.flatten(features, 1)
        if ret_features:
            return self.linear(features), features.data
        elif self.ret_emb:
            return self.linear(features), features
        else:
            return self.linear(features)

    @staticmethod
    def get_embedding_dim():
        return 2048


@register_model("resnet50")
def init_resnet50(model_config):
    return Resnet50(model_config["num_output"],
                    ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                    pretrain=model_config["pretrain"] if "pretrain" in model_config else True)
