import ALBench.metric.metric_impl
from ALBench.skeleton.metric_skeleton import metrics


def get_metric(metric_name):
    return metrics[metric_name]()
