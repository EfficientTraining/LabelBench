import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from LabelBench.skeleton.model_skeleton import register_model


class ModifiedShallow(nn.Module):

    def __init__(self, num_input, num_output, num_hidden, ret_emb):
        super(ModifiedShallow, self).__init__()
        assert num_hidden >= 2, "Shallow network must have at least 2 hidden layers"
        hidden_channels = [num_input] * (num_hidden - 1)
        self.shallow_model = MLP(in_channels=num_input, hidden_channels=hidden_channels, norm_layer=None, dropout=0.0)
        self.classifier = nn.Linear(num_input, num_output)
        self.num_output = num_output
        self.ret_emb = ret_emb

    def forward(self, features, ret_features=False, **kwargs):
        features = F.relu(self.shallow_model(features))
        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)


@register_model("shallow")
def init_MLP(model_config):
    return ModifiedShallow(model_config["input_dim"], model_config["num_output"],
                           model_config["num_hidden"] if "num_hidden" in model_config else 2,
                           ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False)
