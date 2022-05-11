import abc


class TrainerImplementor(metaclass=abc.ABCMeta):
    """

    """
    @abc.abstractmethod
    def fit(self, model, dataset,
            chkpt_path,exp_name,results_path, logs_path,  version,
            batch_size=64, epochs=20, use_optuna=False, learning_rate=1e-4, optimizer='adam', patience_optim=5,
            train_idx=None, validation_idx=None):
        pass

    @abc.abstractmethod
    def get_dataset(self,  main, submain, seq_type,
                    in_size, out_size, point_position,
                    target_norm, quantiles= None,  loader= None, hparams=None):
        pass

    @abc.abstractmethod
    def train_test_split(self, dataset, train_idx, val_idx):
        pass

    @abc.abstractmethod
    def predict(self, model, mains):
        pass

    @abc.abstractmethod
    def load_model(self,model, path):
        pass