# All trainers.
trainers = {}


class Trainer:
    """
    Trainer class for training model on given (partially) datasets.
    """

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric):
        """
        :param Dict trainer_config: Dictionary of hyper-parameters of the trainer.
        :param ALDataset dataset: An initial ALDataset.
        :param model_fn: Function for instantiating a model.
        :param Dict model_config: Dictionary of hyper-parameters for instantiating a model.
        :param Metric metric: Metric object for tracking performances.
        """
        self.trainer_config = trainer_config
        self.dataset = dataset
        self.model_fn = model_fn
        self.metric = metric

    def __init_subclass__(cls, **kwargs):
        """Register trainer subclasses."""
        super().__init_subclass__(**kwargs)
        trainers[cls.trainer_name] = cls

    def train(self, log=False, finetune_model=None, finetune_config=None):
        """
        Train a model.

        :param bool log: Flag indicating whether to log training metrics.
        :param Optional[torch.nn.Module] finetune_model: Warm start model if indicated.
        :param Optional[Dict] finetune_config: Warm start model hyper-parameters.
        """
        pass
