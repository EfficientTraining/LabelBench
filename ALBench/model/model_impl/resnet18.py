import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from ALBench.skeleton.model_skeleton import register_model


class Resnet18(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain):
        super(Resnet18, self).__init__()
        if pretrain:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18()
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(512, num_output)
        self.ret_emb = ret_emb
        self.num_output = num_output

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.model(imgs)
        else:
            features = self.model(imgs)
        if ret_features:
            return self.linear(features), features.data
        elif self.ret_emb:
            return self.linear(features), features
        else:
            return self.linear(features)

    @staticmethod
    def get_embedding_dim():
        return 512


@register_model("resnet18")
def init_resnet18(model_config):
    return Resnet18(model_config["num_output"],
                    ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                    pretrain=model_config["pretrain"] if "pretrain" in model_config else True)
