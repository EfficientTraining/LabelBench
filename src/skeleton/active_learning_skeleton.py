from enum import Enum


class ALInput(Enum):
    TRAIN_PRED = 1
    VAL_PRED = 2
    TEST_PRED = 3

    TRAIN_LABEL = 4
    VAL_LABEL = 5
    TEST_LABEL = 6

    TRAIN_LOSS = 7
    VAL_LOSS = 8
    TEST_LOSS = 9

    TRAIN_EMBEDDING = 10
    VAL_EMBEDDING = 11
    TEST_EMBEDDING = 12


def has_train_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the training set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any training input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 1:
            return True
    return False


def has_val_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the validation set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any validation input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 2:
            return True
    return False


def has_test_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the test set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any testing input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 0:
            return True
    return False


# List of available active learning strategies.
strategies = {}


class Strategy:
    """
    Abstract class for active learning strategies.
    """
    def __init__(self, strategy_config, dataset):
        """
        :param Dict strategy_config: dictionary mapping from parameter name to value.
        :param ALDataset dataset: ALDataset object containing all of training/validation/testing data and embeddings.
        """
        self.strategy_config = strategy_config
        self.dataset = dataset
        self.input_types = None

    def __init_subclass__(cls, **kwargs):
        """
        Register strategy by its strategy_name.
        """
        super().__init_subclass__(**kwargs)
        strategies[cls.strategy_name] = cls

    def select(self, trainer, budget):
        """
        Selecting a batch of examples based on output from the trainer.

        :param Trainer trainer: trainer object with most up-to-date model.
        :param int budget: total batch size budget for the selection.
        :return: A list of indexes of the original dataset indexing the batch for annotation.
        """
        pass
