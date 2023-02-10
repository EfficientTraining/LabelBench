import src.metric.metric_impl
from src.skeleton.metric_skeleton import metrics


def get_metric(metric_name):
    return metrics[metric_name]
