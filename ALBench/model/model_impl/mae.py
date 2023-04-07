import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
import os
import hashlib
import urllib
import warnings
import math
from tqdm import tqdm
from functools import partial
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from ALBench.skeleton.model_skeleton import register_model


def _download(url: str, root: str, expected_md5: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.md5(open(download_target, "rb").read()).hexdigest().startswith(expected_md5):
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the MD5 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.md5(open(download_target, "rb").read()).hexdigest().startswith(expected_md5):
        raise RuntimeError("Model has been downloaded but the MD5 checksum does not not match")

    return download_target


class MAEVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, ret_emb=False, pretrain=True, **kwargs):
        assert pretrain, "MAEVisionTransformer only support pretrain model"
        super(MAEVisionTransformer, self).__init__(**kwargs)
        self.global_pool = (self.num_classes != 0)

        # Setup model.
        if self.num_classes != 0:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # Remove the original normalization layer.

        # Load pretrained checkpoint.
        download_url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        download_root = os.path.expanduser("~/.cache/clip")
        expected_md5 = "8cad7c"
        model_path = _download(download_url, download_root, expected_md5)

        checkpoint = torch.load(model_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % model_path)
        checkpoint_model = checkpoint['model']
        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        self.interpolate_pos_embed(checkpoint_model)

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if self.num_classes == 0:
            assert len(msg.missing_keys) == 0
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            trunc_normal_(self.head.weight, std=2e-5)

        self.ret_emb = ret_emb

    def interpolate_pos_embed(self, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            # Height (== width) for the checkpoint position embedding.
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # Height (== width) for the new position embedding.
            new_size = int(num_patches ** 0.5)
            # Class_token and dist_token are kept unchanged.
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # Only the position tokens are interpolated.
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.forward_features(imgs)
        else:
            features = self.forward_features(imgs)

        if ret_features:
            return self.head(features), features.data
        elif self.ret_emb:
            return self.head(features), features
        else:
            return self.head(features)

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self, split):
        if self.num_classes == 0:
            # Linear probing or shallow net.
            if split == "train":
                return transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                return transforms.Compose([
                    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            # End-to-end finetuning.
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
            if split == "train":
                transform = create_transform(
                    input_size=224,
                    is_training=True,
                    color_jitter=None,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    interpolation='bicubic',
                    re_prob=0.25,
                    re_mode='pixel',
                    re_count=1,
                    mean=mean,
                    std=std,
                )
                return transform
            else:
                return transforms.Compose([
                    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])


@register_model("mae_vitb16")
def init_mae_viTB16(model_config, input_dim=None):
    model = MAEVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0 if model_config["num_output"] == 0 else .1,
        num_classes=model_config["num_output"], ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
        pretrain=model_config["pretrain"] if "pretrain" in model_config else True)
    return model
