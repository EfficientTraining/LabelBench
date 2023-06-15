import LabelBench.metric.metric_impl
from LabelBench.skeleton.metric_skeleton import metrics


def get_metric(metric_name):
    return metrics[metric_name]()
