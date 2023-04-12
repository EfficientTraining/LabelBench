import torch
import torch.nn as nn
from ALBench.skeleton.model_skeleton import register_model


class ModifiedLinear(nn.Module):

    def __init__(self, num_input, num_output):
        super(ModifiedLinear, self).__init__()
        self.model = nn.Linear(num_input, num_output)
        self.num_output = num_output

    def forward(self, features, ret_features=False, **kwargs):
        assert not ret_features, "linear network does not support ret_features"
        return self.model(features)


@register_model("linear")
def init_MLP(model_config):
    return ModifiedLinear(model_config["input_dim"], model_config["num_output"])
