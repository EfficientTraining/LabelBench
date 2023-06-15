import torch
import torch.nn as nn
from LabelBench.skeleton.model_skeleton import register_model


class ModifiedLinear(nn.Module):

    def __init__(self, num_input, num_output, ret_emb):
        super(ModifiedLinear, self).__init__()
        self.model = nn.Linear(num_input, num_output)
        self.num_output = num_output
        self.ret_emb = ret_emb

    def forward(self, features, ret_features=False, freeze=False):
        if freeze:
            features = features.data
        if ret_features:
            return self.model(features), features.data
        elif self.ret_emb:
            return self.model(features), features
        else:
            return self.model(features)


@register_model("linear")
def init_Linear(model_config):
    return ModifiedLinear(model_config["input_dim"], model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False)
