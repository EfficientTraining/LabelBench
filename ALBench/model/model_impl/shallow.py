import torch
import torch.nn as nn
from torchvision.ops import MLP
from ALBench.skeleton.model_skeleton import register_model


class ModifiedShallow(nn.Module):

    def __init__(self, num_input, num_output):
        super(ModifiedShallow, self).__init__()
        self.model = MLP(in_channels=num_input, hidden_channels=[
                         num_input, num_output], norm_layer=None, dropout=0.0)
        self.num_output = num_output

    def forward(self, features, ret_features=False, **kwargs):
        assert not ret_features, "shallow network does not support ret_features"
        return self.model(features)


@register_model("shallow")
def init_MLP(model_config, input_dim=None):
    return ModifiedShallow(input_dim, model_config["num_output"])
