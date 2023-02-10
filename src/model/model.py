import src.model.model_impl
from src.skeleton.model_skeleton import model_fns


def get_model(name, model_config, model=None):
    fn = model_fns[name]
    return fn(model_config, model=model)


if __name__ == "__main__":
    print(model_fns)
    print(get_model("resnet18", {"num_output": 10}))
