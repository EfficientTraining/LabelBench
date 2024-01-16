corrupter_fns = {}


def register_corrupter(name: str):
    """
    Register corrupter with corrupter_name.

    :param name: Corrupter's name.
    :return: Corrupter function.
    """
    def model_decor(corrupter_fn):
        corrupter_fns[name] = corrupter_fn
        return corrupter_fn
    return model_decor
