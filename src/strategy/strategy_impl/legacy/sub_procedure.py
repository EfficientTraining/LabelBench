class SamplingSubProcedure:
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        self.trainer = trainer
        self.model = model
        self.embs = embs
        self.preds = preds
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size

    def sample(self, labeled_set):
        raise NotImplementedError()

    def update(self, idx):
        return
