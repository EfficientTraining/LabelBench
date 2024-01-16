import LabelBench.corrupter.corrupter_impl
from LabelBench.skeleton.corrupter_skeleton import corrupter_fns


def get_corrupter_fn(name, corrupter_config):
    fn = corrupter_fns[name]
    return lambda labels: fn(corrupter_config, labels)
