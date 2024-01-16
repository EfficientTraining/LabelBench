from LabelBench.skeleton.corrupter_skeleton import register_corrupter


@register_corrupter("noiseless")
def noiseless_corrupter(corrupter_config, labels):
    return labels
