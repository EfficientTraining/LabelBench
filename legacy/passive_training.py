if __name__ == "__main__":
    from hyperparam import *
    from src.dataset import get_dataset
    from model import get_model_class

    lr = 1e-4
    batch_size = 50
    n_epoch = 50
    run_name = "model=%s, lr=%f, batch_size=%d, n_epoch=%d" % (model_name, lr, batch_size, n_epoch)
    wandb.init = wandb.init(project="Model Training, %s" % data_name, entity=wandb_name, name=run_name,
                            config=vars(args))
    train_dataset, val_dataset, test_dataset, multi_label_flag, n_class = get_dataset(data_name, batch_size)
    train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=(
        len(train_dataset) // 10, len(train_dataset) - len(train_dataset) // 10))
    model_class = get_model_class(model_name)
    loss_fn, pred_fn, metric = get_fns(multi_label_flag)
    trainer = PassiveTrainer(model_class, n_class, n_epoch, loss_fn, metric, pred_fn, batch_size=batch_size)
    trainer.train(train_dataset, test_dataset, log=True, lr=lr)
