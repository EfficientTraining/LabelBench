# Datasets
To implement a new dataset, simply implement a new function under `./dataset_impl`.
The function should be declared using the `@register_dataset` decorator. Roughly, the function should look
like the following:
```
@register_dataset("Your Dataset Name", LabelType.MULTI_CLASS)
def get_your_dataset(data_dir, *args):
    ...
    return train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, n_class, classnames
```
where `LabelType.MULTI_CLASS` can be specified as another LabelType depending on what the task of the dataset.
`"Your Dataset Name"` can be set to the name of the dataset and will be the name for specifying `--dataset` arguments
when launching experiments.
Additionally, `data_dir` specifies the directory to save and load the dataset. The returned values should follow the following spec:
- `train_dataset`, `val_dataset` and `test_dataset` are all instances `LabelBench.skeleton.dataset_skeleton.TransformDataset`.
- `train_labels`, `val_labels`, `test_labels` are optional numpy arrays containing all labels as one-hot vectors. If specified to `None`, additional computation will be performed to loop over the entire dataset to obtain the labels. The numpy arrays should always be of shape N by K, where N is the number of exampels in the dataset and K is the number of classes.
- `n_class` is the number of classes of the dataset.
- `classnames` is the list of strings containing the meaning of each word in natural language words. Its length should equal to the number of classes.
