# Trainers
Trainer takes a partially labeled dataset and trains a model. Trainers are more complicated to implement.
In general, a trainer should be implemented in `trainer_impl` and
inherit from `LabelBench.skeleton.trainer_skeleton.Trainer`. A class attribute `trainer_name` should be
specified as the trainer's name and will be used in configuration files.

For supervised learning trainers, please see the implementation of `PyTorchPassiveTrainer` in
`trainer_impl/pytorch_passive_trainer.py` as an example.

For semi-supervised learning trainers, one can inherit from the `PyTorchSemiTrainer` class in
`trainer_impl/pytorch_semi_trainer.py` and simply implement a `train_step` function.
See `trainer_impl/flexmatch.py` for an example implementation.
