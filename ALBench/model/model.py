import ALBench.model.model_impl
from ALBench.skeleton.model_skeleton import model_fns


def get_model_fn(name):
    fn = model_fns[name]
    return fn


if __name__ == "__main__":
    print(model_fns)
    print(get_model_fn("resnet18"))
