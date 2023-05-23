import torch
from tqdm import tqdm

import ALBench.templates as templates


def get_zeroshot_classifier(clip_model, tokenizer, classnames, template):
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
            texts = tokenizer(texts).cuda()  # Tokenize.
            embeddings = clip_model.encode_text(texts)  # Embed with text encoder.
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


