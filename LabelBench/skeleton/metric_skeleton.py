metrics = {}


class Metric:
    """
    Parent class for metric logging.
    """

    def __init_subclass__(cls, **kwargs):
        """Tracks all of the metric classes."""
        super().__init_subclass__(**kwargs)
        metrics[cls.metric_name] = cls

    def compute(self, epoch, preds, labels, losses, val_preds, val_labels, val_losses, test_preds, test_labels,
                test_losses, num_labeled=None, labeled=None):
        """
        Compute training metrics.

        :param int epoch: epoch number.
        :param numpy.ndarray preds: Model predictions on the entire pool.
        :param numpy.ndarray labels: Ground truth labels of the entire pool.
        :param numpy.ndarray losses: Training loss values of the entire pool.
        :param numpy.ndarray val_preds: Model predictions on the validation set.
        :param numpy.ndarray val_labels: Ground truth labels of the validation set.
        :param numpy.ndarray val_losses: Loss values of the validation set. One loss per example.
        :param numpy.ndarray test_preds: Model predictions on the test set.
        :param numpy.ndarray test_labels: Ground truth labels of the test set.
        :param numpy.ndarray test_losses: Loss values of the test set. One loss per example.
        :param Optional[int] num_labeled: Number of labeled examples in the pool.
        :param Optional[numpy.ndarray] labeled: A binary array of the same size as the pool. Indicating whether an
            example is labeled or not.
        :return: A dictionary containing all logged metrics.
        """
        pass
