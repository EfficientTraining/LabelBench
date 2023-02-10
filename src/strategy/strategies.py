import src.strategy.strategy_impl
from src.skeleton.active_learning_skeleton import strategies


def get_strategy(name, strategy_config, dataset):
    return strategies[name](strategy_config, dataset)
