# Active Learning Strategies

To implement a new active learning strategy in `strategy_impl`, simply inherit the
`LabelBench.skeleton.active_learning_skeleton.Strategy` class and start with the following structure.

```
class YourSampling(Strategy):
    strategy_name = "Your Strategy Name"

    def __init__(self, strategy_config, dataset):
        super(YourSampling, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_PRED]

    def select(self, trainer, budget):
        input1, input2, ... = trainer.retrieve_inputs(self.input_types)
        ...
        return selected_idxs
```

`strategy_name` should be specified as the name of the new strategy. It is also used to specify
the name of the strategy in configuration files.
Moreover, `self.input_types` should be specified as a list of information the active learning strategy
needs to make its selections. Each type in the list should be an instance of
`LabelBench.skeleton.active_learning_skeleton.ALInput`. The input_types can then be passed to `trainer`
to retrieve the necessary inputs to the active learning strategy in order.
`trainer` is passed in as an instance of `LabelBench.skeleton.trainer_skeleton.Trainer` that contains
the latest predictions of the model.

The returned `selected_idxs` should have length of `budget`, and are indices to the
`train_dataset` (active learning pool) as part of the `dataset`, which is an instance
of `LabelBench.skeleton.dataset_skeleton.ALDataset`.
