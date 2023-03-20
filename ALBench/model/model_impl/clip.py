import ALBench.templates as templates
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ALBench.skeleton.model_skeleton import register_model
from tqdm import tqdm

#Code related to zeroshot classifier is mainly ported from https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py

def get_zeroshot_classifier(clip_model, classnames, template):

    assert template is not None, 'template is required for zeroshot classifier.'
    assert classnames is not None, 'classnames is required for zeroshot classifier.'
    template = getattr(templates, template)
    logit_scale = clip_model.logit_scale

    clip_model.eval()
    clip_model.cuda()

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).cuda() # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head 

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


    # TODO: save and load
    # def save(self, filename):
    #     print(f'Saving classification head to {filename}')
    #     utils.torch_save(self, filename)

class CLIPVisionOnly(nn.Module):
    def __init__(self, num_output, ret_emb, pretrain, model_name):

        super(CLIPVisionOnly, self).__init__()
        assert pretrain, "CLIPVisionOnly only support pretrain model"

        model, preprocess = clip.load(model_name)
        self.image_encoder_model = model
        self.embed_dim = model.state_dict()["text_projection"].shape[1]
        self.preprocess_transform = preprocess
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output)
            print("Initialize CLIP_VisionOnly with default linear head. Recommend to initialize the head using zero-shot classifier params.")
        else:
            self.classifier = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=True):
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

    def get_preprocess(self):
        return self.preprocess_transform


@register_model("clip_vitb16")
def init_clip_viTB16(model_config, input_dim=None):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/16")


@register_model("clip_vitb32")
def init_clip_viTB32(model_config, input_dim=None):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/32")
