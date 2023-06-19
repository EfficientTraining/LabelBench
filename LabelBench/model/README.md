# Models
To implement a new model, implement the following function for initializing an instance of the model under `model_impl`.
```
@register_model("Your Model Name")
def init_your_model(model_config):
    ...
    return model
```
`"Your Model Name"` should be specified as the name of the model to be used in configuration files.
`model_config` is a dictionary of all the configuration arguments passed in from configuration files and
upstream python scripts (e.g. `main.py`).
The returned `model` should be an instance of `torch.nn.Module`. Additionally, if this is a pretrained model,
the model class of the returned `model` should also implement the following functions:
- `init_head_withzeroshot(self, classnames, templates)` that returns a zero-shot linear classifier head.
Return a randomly initialized linear model if the pretrained model does not have zero-shot capabilities.
- `get_embedding_dim(self)` that returns an integer of the number of dimensions of the penultimate layer feature
embeddings.
- `get_preprocess(self, split)` that returns a model specific transformation for data augmentation on dataset.

For a detailed example, see `model_impl/clip.py`.