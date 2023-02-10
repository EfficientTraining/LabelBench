import wandb
from torch.utils.data import Subset

from src.dataset import get_dataset
from hyperparam import *
from model import get_model_class
from src.strategy import get_strategies
from trainer import PassiveTrainer, get_fns

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, multi_label_flag, n_class = get_dataset(data_name, batch_size)
    run_name = "%s, model=%s, batch_size=%d" % (alg_str, model_name, batch_size)
    data_str = "%s_%d_classes" % (data_name, n_class)
    wandb.init(project="Active Learning, %s" % data_str, entity=wandb_name, name=run_name, config=vars(args))
    model_class = get_model_class(model_name)
    loss_fn, pred_fn, metric = get_fns(multi_label_flag)
    trainer = PassiveTrainer(model_class, n_class, n_epoch, loss_fn, metric, pred_fn, multi_label_flag=multi_label_flag)

    sampling_alg, strategy = get_strategies(alg_name, sub_procedures)
    labeled = np.random.choice(np.arange(len(train_dataset)), size=batch_size, replace=False)
    model = trainer.train(Subset(train_dataset, labeled), None)
    train_preds, train_labels, train_losses, embs = trainer.test(train_dataset, model, ret_emb=True)
    val_preds, val_labels, val_losses = trainer.test(val_dataset, model)
    test_preds, test_labels, test_losses = trainer.test(test_dataset, model)
    wandb.log(metric.compute(1, train_preds, train_labels, train_losses, val_preds, val_labels, val_losses, test_preds,
                             test_labels, test_losses, num_labeled=len(labeled), labeled=labeled))
    for idx in range(2, num_batch + 1):
        new_label = sampling_alg(trainer, model, embs, train_preds, train_labels, labeled, train_dataset, batch_size)
        labeled = np.concatenate([labeled, new_label], axis=0)
        model = trainer.train(Subset(train_dataset, labeled), None)
        train_preds, train_labels, train_losses, embs = trainer.test(train_dataset, model, ret_emb=True)
        val_preds, val_labels, val_losses = trainer.test(val_dataset, model)
        test_preds, test_labels, test_losses = trainer.test(test_dataset, model)
        wandb.log(
            metric.compute(idx, train_preds, train_labels, train_losses, val_preds, val_labels, val_losses, test_preds,
                           test_labels, test_losses, num_labeled=len(labeled), labeled=labeled))

    for i, idx in enumerate(labeled):
        wandb.log({
            "i": i,
            "Label index": idx,
        })
    wandb.finish()
