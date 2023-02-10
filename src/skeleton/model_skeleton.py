model_fns = {}


def register_model(name: str):
    """
    Register model with model_name.

    :param name: Model's name.
    :return: Model decorator.
    """
    def model_decor(model_fn):
        model_fns[name] = model_fn
        return model_fn
    return model_decor
