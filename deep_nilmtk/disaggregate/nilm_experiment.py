from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
import pytorch_lightning as pl
from nilmtk.disaggregate import Disaggregator
import mlflow
from functools import partial
from sklearn.model_selection import TimeSeriesSplit
import optuna 
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pickle

from ..config import get_exp_parameters
from ..preprocessing.pre_processing import data_preprocessing
from ..utils.utils import get_latest_checkpoint
from ..model.model_pil import pilModel
from ..config import NILM_MODELS
from ..utils import DictLogger



class NILMExperiment(Disaggregator):
    """
    This class defines a NILM experiment. It is compatibale with both
    single and multi-appliance models and offers different advanced features 
    like cross-validation and hyper-parametrs optimization during the 
    training phase. The class is independent of the deep model used for 
    load disaggregation. 

    .. note:: For a PyTorch model to be compatible with this class, an entry should be added for this model in the config module. 
    """

    def __init__(self, params): 
        """
        Initialise the environmental parametrs for the NILM experiment. For the 
        hyper-parameters, it takes the default values defined in the config module 
        and updates only the subset of values specified in params.

        :param params:  Dictionnary with different values of hyper-parameters.
        :type params: dictionnary
        """
        super().__init__()
        
        hparams = get_exp_parameters()
        hparams = vars(hparams.parse_args())
        hparams.update(params)
        pl.seed_everything(hparams['seed'])
        
        self._data = None
        self.models = OrderedDict()
        self.data_loaders = OrderedDict()
        self.MODEL_NAME = hparams['model_name']
        self.hparams = hparams
        # Paramter for MLflow
        self.run_id = OrderedDict()
        # Parameters for Optuna
        self.optuna_params = OrderedDict()
        self.best_trials ={}
        # Parameter for appliances
        self.appliance_params ={}
        
    def _prepare_data(self, mains, sub_main):
        """
        Performs data pre-processing and formating. By default, the default pre-processing 
        method in used. Neverthless, custom pre-processing methdos are also possible
        to use and need only to be specified in the corresponding entry of the model
        within the config module within the extra_params. For example:
        
        NILM_MODELS = {
            ...
            'NILMmodel': {
                'model': modelClass,
                'loader': dataLoaderClass,
                'extra_params':{
                    'pre-process': preprocessingFunction
                }
            },
            ...
        }

        :param mains: aggregtae power consumption
        :type mains: List of pd.DataFrame
        :param sub_main: sub metered energy consumption
        :type sub_main: List of pd.DataFrame
        """
        # Check if the data is not already loaded
        
        if self._data is None:
            # Check if the specified model require a custom preprocessing function
            preprocess_func = NILM_MODELS[self.hparams['model_name']]['extra_params']['pre-process'] if 'pre-process' in NILM_MODELS[self.hparams['model_name']]['extra_params'] else data_preprocessing
            # Pre-processing
            mains, multi_appliance_meters, single_appliance_meters = preprocess_func(mains, 
                                                            sub_main, 
                                                            self.hparams['feature_type'],
                                                            self.hparams['alpha'],
                                                            self.hparams['input_norm'], 
                                                            self.hparams['main_mu'], 
                                                            self.hparams['main_std'], 
                                                            self.hparams['q_filter'])
            # Data fromating according to the type of the model ['single-appliance', 'multi-appliance']
            self._data = {
                    "features":mains, 
                    "targets":multi_appliance_meters 
            } if self.hparams["multi_appliance"] else {
                    "features":mains, 
                    "targets":single_appliance_meters
            }
        
    def  partial_fit(self,  mains, sub_main,do_preprocessing=True, **load_kwargs):
        """ Trains the model for appliances according to the model name specified 
        in the experiment's definition. It starts with the data pre-processing and 
        formatting and then train the model based on the type of the model(single 
        or multi-task).


        :param mains: Aggregate power measurements.
        :type mains: Liste of pd.DataFrame
        :param sub_main: Appliances power measurements.
        :type sub_main: Liste of pd.DataFrame
        :param do_preprocessing: Performs pre-processing or not. Defaults to True., defaults to True
        :type do_preprocessing: bool, optional
        """
    
        # Check if the fitting was not already done
        if self._data is None:
            # Data pre-processing
            if do_preprocessing:
                self._prepare_data(mains, sub_main)
            
            # Preparing folders for saving results
            logs = Path(self.hparams['logs_path']) # logs/ log files containing the details of the execution
            results = Path(self.hparams['results_path']) # results/ csv and pickle files recording the testing results
            figures = Path(self.hparams['figure_path']) # figures/ folder to save figures 
            
            logs.mkdir(parents=True, exist_ok=True)
            logs.mkdir(parents=True, exist_ok=True)
            results.mkdir(parents=True, exist_ok=True)
            figures.mkdir(parents=True, exist_ok=True)

            # Select a fitting strategy according to the model type
            if not self.hparams['multi_appliance']:
                self.single_appliance_fit()
            else:
                print("""
                
                This is partial fit with multi-applaince
                
                """)
                self.multi_appliance_fit()

    def disaggregate_chunk(self,test_main_list,do_preprocessing=True):
        """
        Uses trained models to disaggregate the test_main_list. It is compatible with both single and multi-appliance models. 

        :param test_main_list: Aggregate power measurements.
        :type test_main_list: Liste of pd.DataFrame
        :param do_preprocessing: Specify if pre-processing need to be done or not, defaults to True
        :type do_preprocessing: bool, optional
        :return: Appliances power measurements.
        :rtype: list of pd.DataFrame
        """
        
        if not self.hparams['multi_appliance']:
            test_predictions = self.single_appliance_disaggregate(test_main_list, do_preprocessing = do_preprocessing)
            return test_predictions
        else:
            test_predictions = self.multi_appliance_disaggregate(test_main_list, do_preprocessing = do_preprocessing)
            return test_predictions

    def single_appliance_disaggregate(self, test_main_list, model=None,do_preprocessing=True):
        """
        Perfroms load disaggregtaion for single appliance models. If Optuna was used during the 
        training phase, it disaggregtaes the test_main_list using only the best trial. 
        If cross-validation is used during training, it returns the average of predictions 
        cross all folds for each applaince. In this later case, the predictions for each fold 
        are also logged in the results folder under the name 
        ['model_name']_[appliance_name]_all_folds_predictions.p. 
        Alternatively, when both Optuna and cross-validation are used, it returns the average predictions 
        of all folds for only the best trial.

        :param test_main_list: Aggregate power measurements
        :type test_main_list: liste of pd.DataFrame
        :param model: Pre-trained appliance's models. Defaults to None.
        :type model: dict, optional
        :param do_preprocessing: Specify if pre-processing need to be done or not, defaults to True
        :type do_preprocessing: bool, optional
        :return: estimated power consumption of the considered appliances.
        :rtype: liste of dict
        """    
          
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = data_preprocessing(test_main_list, None,
                                            self.hparams['feature_type'],
                                            self.hparams['alpha'],
                                            self.hparams['input_norm'], 
                                            self.hparams['main_mu'], 
                                            self.hparams['main_std'], 
                                            self.hparams['q_filter'])
        
        test_predictions = []
        test_results = []
        for test_main in test_main_list:
            test_main = test_main.values
            disggregation_dict = {}
            result_dict = {}

            
            
            for appliance in self.models:
            
                dataloader = self.data_loaders[appliance]
                model = self.models[appliance]
                
                
            
                data = dataloader(inputs=test_main, 
                                    targets=None,
                                    params = self.hparams )

                test_loader = torch.utils.data.DataLoader(data, 
                                self.hparams['batch_size'], 
                                collate_fn= 
                                NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                    self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                shuffle=False, 
                                num_workers=self.hparams['num_workers'])
                
                exp_name = self.hparams['checkpoints_path']+f"{self.exp_name}_{appliance}"
                
                if self.hparams['use_optuna']:
                    exp_name += f'/trial_{self.best_trials[appliance]}/'

                if self.hparams['kfolds'] > 1:
                    # TODO: check if average cross all folds or only the best model
                    app_result_cross_fold =[]
                    dump_results ={}
                    for fold in  model:
                        #load checkpoints
                        checkpoint_path = get_latest_checkpoint(exp_name+f'/{fold}')
                        chechpoint=torch.load(checkpoint_path)

                        model_fold = model[fold]
                        model_fold.load_state_dict(chechpoint['state_dict'])
                        
                        model_fold.eval()
                        
                        network = model_fold.model.eval()
                        
                        if self.hparams['target_norm'] == 'z-norm' :
                            network.mean = self.appliance_params[appliance]['mean']
                            network.std = self.appliance_params[appliance]['std']
                        elif self.hparams['target_norm'] == 'min-max' :
                            network.min = self.appliance_params[appliance]['min']
                            network.max = self.appliance_params[appliance]['max']

                        results = network.predict(model_fold, test_loader) 
                        df = results['pred'].cpu().numpy().flatten()

                        app_result_cross_fold.append(df)
                        dump_results [fold] = df

                    dump_results ['mean_preditions'] = pd.Series(np.mean(np.array(app_result_cross_fold), axis=0))
                    dump_results ['std_predictions'] = pd.Series(np.std(np.array(app_result_cross_fold), axis=0))
                    dump_results ['min_predictions'] = pd.Series(np.min(np.array(app_result_cross_fold), axis=0))
                    dump_results ['max_predictions'] = pd.Series(np.max(np.array(app_result_cross_fold), axis=0))
                    
                    pickle.dump(dump_results, open(f"{self.hparams['results_path']}/{self.hparams['model_name']}_{appliance}_all_folds_predictions.p", "wb"))

                    df = pd.Series(np.mean(np.array(app_result_cross_fold), axis=0))
                    
                else:
                    #load checkpoints
                    
                    checkpoint_path = get_latest_checkpoint(exp_name)
                    chechpoint=torch.load(checkpoint_path)

                    model.load_state_dict(chechpoint['state_dict'])
                    model.eval()
                    
                    network = model.model.eval()
                    
                    if self.hparams['target_norm'] == 'z-norm' :
                            network.mean = self.appliance_params[appliance]['mean']
                            network.std = self.appliance_params[appliance]['std']
                    elif self.hparams['target_norm'] == 'min-max' :
                        network.min = self.appliance_params[appliance]['min']
                        network.max = self.appliance_params[appliance]['max']
                        
                    results = network.predict(model, test_loader) 
                    df = pd.Series(results['pred'].cpu().numpy().flatten())

                disggregation_dict[appliance] = df
                result_dict[appliance]=results
                
                # Tracking results of the current applaince
                appliance_results = {}
                for key in results:
                    appliance_results[key] = pd.Series(results[key].cpu().numpy().flatten())
                # Saving files in the disk
                appliance_results = pd.DataFrame(appliance_results)
                
                
                # TODO:should be logged in a subdirectory
                # Create couple of artifact files under the directory "data"
                os.makedirs(self.hparams['results_path']+f'/{appliance}', exist_ok=True)
                appliance_results.to_csv(self.hparams['results_path']+f'/{appliance}/{self.exp_name}.csv', index = False)
                # Logging results relative the current appliance
                mlflow.set_experiment(appliance)

                if self.hparams['log_artificat']:
                    with mlflow.start_run(self.run_id[appliance]):
                        mlflow.log_artifacts(self.hparams['results_path']+f'/{appliance}', artifact_path="test_results")

            results = pd.DataFrame(disggregation_dict, dtype='float32')
            
            test_predictions.append(results)
            test_results.append(result_dict)



        np.save(self.hparams['results_path']+f"{self.exp_name}.npy", test_results)   
         
        return test_predictions
           
    def objective(self, trial, train_loader=None, val_loader=None, fold_idx= None):
        """The objective function to be used with optuna. This function requires the model under study to 
        implement a static function called suggest_hparams() [see the model documentation for more informations]

        :param trial: Optuna.trial
        :param train_loader: training dataLoader for the current experiment. Defaults to None.
        :type train_loader: DataLoader, optional
        :param val_loader: validation dataLoader for the current experiment. Defaults to None.
        :type val_loader: DataLoader, optional
        :param fold_idx: Number of the fold of cross-validation is used. Defaults to None.
        :type fold_idx: int, optional
        :raises Exception: In case the model does not suggest any parameters.
        :return: The best validation loss aschieved
        :rtype: float
        """
        #
        #  Initialize the best_val_loss value 
        best_val_loss = float('Inf')
        
        mlflow.set_experiment(f'{self.optuna_params["appliance_name"]}')
        # Start a new mlflow run
        with mlflow.start_run():
            
            # Get hyperparameter suggestions created by Optuna 
            # and log them as params using mlflow
            
            suggested_params_func = NILM_MODELS[self.hparams['model_name']]['model'].suggest_hparams
            if callable(suggested_params_func):
                suggested_params = NILM_MODELS[self.hparams['model_name']]['model'].suggest_hparams(None,trial)
                self.hparams.update(suggested_params)
                mlflow.log_params(suggested_params)

            else:
                raise Exception('''
                                No params to optimise by optuna
                                A static function inside the NILM model should provide
                                a dictionnary of params suggested by optuna
                                see documentation for more details
                                ''')
            
            # Check if the appliance was already trained. If not then create a new model for it
            print("First model training for", self.optuna_params["appliance_name"])
            
            net, dataloader = self.get_net_and_loaders()
                
            # To use only if the Cross validation is not used    
            if (train_loader is None) or (val_loader is None):                
                self.data_loaders[self.optuna_params["appliance_name"]]=dataloader
                
                dataloader  = self.data_loaders[self.optuna_params["appliance_name"]]
                data = dataloader(inputs=self._data['features'], targets=self.optuna_params["power"])
                
                train_data, val_data = torch.utils.data.random_split(data, 
                                                     [int(data.len*(1-0.15)), data.len - int(data.len*(1-0.15))], 
                                                     generator=torch.Generator().manual_seed(42))
                
                train_loader = torch.utils.data.DataLoader(train_data, self.hparams['batch_size'], shuffle=True, 
                                                           collate_fn= 
                                                            NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                           num_workers=self.hparams['num_workers'])
                
                val_loader = torch.utils.data.DataLoader(val_data, 
                                                    self.hparams['batch_size'], 
                                                    shuffle=False, 
                                                    collate_fn= 
                                                            NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                    num_workers=self.hparams['num_workers'])

            # Auto log all MLflow from lightening
            mlflow.pytorch.autolog()  
            
            # Model Training
            if fold_idx is None:
                self.models[self.optuna_params["appliance_name"]] = pilModel(net, self.hparams)
                best_val_loss, path = self.train_model(
                    self.optuna_params["appliance_name"], 
                    train_loader, val_loader, 
                    self.optuna_params['exp_name'],
                    data.mean if  self.hparams['target_norm'] == 'z-norm' else None,
                    data.std if self.hparams['target_norm'] == 'z-norm' else None,
                    trial_idx = trial.number)
            else:
                
                self.models[self.optuna_params["appliance_name"]][f'fold_{fold_idx}'] = pilModel(net, self.hparams)
                best_val_loss, path = self.train_model(
                    self.optuna_params["appliance_name"], 
                    train_loader, val_loader, 
                    self.optuna_params['exp_name'],
                    data.mean if  self.hparams['target_norm'] == 'z-norm' else None,
                    data.std if self.hparams['target_norm'] == 'z-norm' else None,
                    trial_idx = trial.number,
                    fold_idx= fold_idx,
                    model = self.models[self.optuna_params["appliance_name"]][f'fold_{fold_idx}'])

                
            # saving the trained model 
            trial.set_user_attr(key='best_run_id', value = mlflow.active_run().info.run_id)
            trial.set_user_attr(key="trial_ID", value= trial.number )
            trial.set_user_attr(key="path", value= path )
            
        return best_val_loss
    
    def objective_cv(self, trial):
        """The objective function for Optuna when cross-validation is also used

        :param trial: An optuna trial
        :type trial: Optuna.Trial
        :return: average of best loss validations for considered folds
        :rtype: float
        """
        fold = TimeSeriesSplit(n_splits=self.hparams['kfolds'], test_size=self.hparams['test_size'], gap = self.hparams['gap'])
        scores = []
        
        
        #select model and data loaders to use
        _, dataloader = self.get_net_and_loaders()
        self.models[self.optuna_params["appliance_name"]] = {}
        self.data_loaders[self.optuna_params["appliance_name"]]=dataloader
        
        dataloader  = self.data_loaders[self.optuna_params["appliance_name"]]
        dataset = dataloader(inputs=self._data['features'], targets=self.optuna_params["power"])
        
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(dataset)))):
            print(f'started training for the fold {fold_idx}.')
            
            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, valid_idx)
            
            train_loader = torch.utils.data.DataLoader(train_data, self.hparams['batch_size'], shuffle=True, 
                                                       collate_fn= 
                                                           NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                            self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                       num_workers=self.hparams['num_workers'])
            
            val_loader = torch.utils.data.DataLoader(val_data, 
                                                self.hparams['batch_size'], 
                                                shuffle=False, 
                                                collate_fn= 
                                                        NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                            self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                num_workers=self.hparams['num_workers'])
        
            mae_loss = self.objective(trial, train_loader, val_loader, fold_idx)

        
            scores.append(mae_loss)
        
        return np.mean(scores)
    
    def get_net_and_loaders(self):
        """Returns an instance of the specified model and the correspanding dataloader

        :return: (model , dataloader)
        :rtype: tuple(nn.Module, torch.utils.data.Dataset)
        """
        # Get the class of the required model from the config file
        net = NILM_MODELS[self.hparams['model_name']]['model'](self.hparams)
        
        # Get the class of the related dataloader from the config file
        data = partial(
                NILM_MODELS[self.hparams['model_name']]['loader'],
                params = self.hparams)

        return net, data 

    def save_best_model(self, study, trial):
        """Keeps track of the trial giving best results

        :param study: Optuna study
        :param trial: Optuna trial
        """

        if study.best_trial.number == trial.number:
            study.set_user_attr(key="trial_ID", value=trial.number)
            study.set_user_attr(key="best_run_id", value=trial.user_attrs["best_run_id"])
            study.set_user_attr(key="path", value=trial.user_attrs["path"])
    
    def single_appliance_fit(self):
        """
        Train the specified models for each appliance separately taking into consideration
        the use of cross-validation and hyper-parameters optimisation. The checkpoints for 
        each model are saved in the correspondng path.
        """        
        self.exp_name = f"{self.hparams['model_name']}_{self.hparams['data']}_single_appliance_{self.hparams['experiment_label']}"
        original_checkpoint = self.hparams['checkpoints_path']
        
        for appliance_name, power in self._data['targets']:
            exp_name = f"{self.exp_name}_{appliance_name}"
            checkpoints = Path(original_checkpoint +f"{exp_name}")
            checkpoints.mkdir(parents=True, exist_ok=True)
            #update checkpoint path
            new_params = {"checkpoints_path": original_checkpoint +f"{exp_name}",
                          "appliances":[appliance_name]
                        }
            self.hparams.update(new_params)
            
            print(f"fit model for {exp_name}")
            
            if self.hparams['use_optuna']:
                # Use Optuna fot parameter optimisation of the model
                study = optuna.create_study(study_name=exp_name, direction="minimize")
                self.optuna_params ={
                    'power' :power,
                    'appliance_name':appliance_name,
                    'exp_name': exp_name
                    }
                
                if self.hparams['kfolds'] <= 1:
                    study.optimize(self.objective, n_trials=self.hparams['n_trials'], callbacks=[self.save_best_model])
                    # Load weights of the model
                    app_model, _ = self.get_net_and_loaders()
                    #TODO: load checkpoints

                    chechpoint=torch.load(study.user_attrs["path"])
                    
                    model = pilModel(app_model, self.hparams)
                    model.hparams['checkpoint_path'] = study.user_attrs["path"]
                    model.load_state_dict(chechpoint['state_dict'])
                    model.eval()
            
                    # Save best model for testing time
                    self.models[appliance_name] = model
                else:
                    study.optimize(self.objective_cv, n_trials=self.hparams['n_trials'], callbacks=[self.save_best_model])
                
                # Save figures
                try:
                    fig1 = optuna.visualization.plot_param_importances(study)
                    fig2 = optuna.visualization.plot_parallel_coordinate(study)
                    
                    fig2.write_image(self.hparams['checkpoints_path'] +'/_parallel_coordinate.pdf')
                    fig1.write_image(self.hparams['checkpoints_path'] +'/_param_importance.pdf')
                except:
                    pass

                results_df = study.trials_dataframe()
                results_df.to_csv( f'{self.hparams["checkpoints_path"]}/Seq2Point_Study_{exp_name}_{appliance_name}.csv')
                joblib.dump(study, f'{self.hparams["checkpoints_path"]}/Seq2Point_Study_{exp_name}_{appliance_name}.pkl')
                
                # Restoring the best model and use it for the testing
                self.best_trials[appliance_name] = study.best_trial.number 
                app_model, _ = self.get_net_and_loaders()
                #TODO: load checkpoints
                self.run_id[appliance_name] =  study.user_attrs["best_run_id"]
                
            else:
                # Check if the appliance was already trained. If not then create a new model for it
                
                if self.hparams['kfolds'] > 1:
                    self.models[appliance_name] ={}
                    # Getting the required data for the appliance
                    _, dataloader = self.get_net_and_loaders()
                    self.data_loaders[appliance_name]=dataloader
                    dataset = dataloader(inputs=self._data['features'], targets=power)
                    
                    # Splitting the data into several folds
                    fold = TimeSeriesSplit(n_splits=self.hparams['kfolds'], test_size=self.hparams['test_size'], gap = self.hparams['gap'])

                    scores = []
                    
                    # Using each fold as a validation data
                    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(dataset)))):
                        print(f'started training for the fold {fold_idx}.')

                        app_model, _ = self.get_net_and_loaders()

                        self.models[appliance_name][f'fold_{fold_idx}'] = pilModel(app_model, self.hparams)

                        train_data = torch.utils.data.Subset(dataset, train_idx)
                        val_data = torch.utils.data.Subset(dataset, valid_idx)
                        
                        train_loader = torch.utils.data.DataLoader(train_data, 
                                                                   self.hparams['batch_size'], 
                                                                   shuffle=True, 
                                                                   collate_fn= 
                                                                    NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                        self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                                   num_workers=self.hparams['num_workers'])
                        
                        val_loader = torch.utils.data.DataLoader(val_data, 
                                                            self.hparams['batch_size'], 
                                                            shuffle=False, 
                                                            collate_fn= 
                                                                    NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                        self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                            num_workers=self.hparams['num_workers'])
                    
                        # select experiment if does not exist create it 
                        # an experiment is created for each appliance
                        mlflow.set_experiment(f'{appliance_name}')
                        
                        
                
                        # Start a new for the current appliance
                        with mlflow.start_run(run_name=self.hparams['model_name']): 
                            # Auto log all MLflow from lightening
                            mlflow.pytorch.autolog()  
                            # Save the run ID to use in testing phase
                            self.run_id[appliance_name] =  mlflow.active_run().info.run_id
                            # Log parameters of current run 
                            mlflow.log_params(self.hparams)
                            # Model Training
                            mae_loss = self.train_model(appliance_name, 
                                                        train_loader, 
                                                        val_loader,
                                                        exp_name,
                                                        dataset.mean if self.hparams['target_norm'] == 'z-norm' else None,
                                                        dataset.std  if self.hparams['target_norm'] == 'z-norm' else None,
                                                        fold_idx = fold_idx,
                                                        model=self.models[appliance_name][f'fold_{fold_idx}'])
                        
                        scores.append(mae_loss)
                                            
                    
            
                else:

                    if appliance_name not in self.models:
                        print("First model training for", appliance_name)
                        #select model and data loaders to use
                        net, dataloader = self.get_net_and_loaders()
                        self.models[appliance_name] = pilModel(net, self.hparams)
                        self.data_loaders[appliance_name]=dataloader
                    # Retrain the particular appliance
                    else:
                        print("Started Retraining model for", appliance_name)
                    
                    

                    dataloader  = self.data_loaders[appliance_name]
            
                    data = dataloader(inputs=self._data['features'], targets=power)
                    
                    train_data, val_data=torch.utils.data.random_split(data, 
                                                         [int(data.len*(1-0.15)), data.len - int(data.len*(1-0.15))], 
                                                         generator=torch.Generator().manual_seed(42))
                    train_loader = torch.utils.data.DataLoader(train_data, self.hparams['batch_size'], shuffle=True, 
                                                               collate_fn= 
                                                                NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                    self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                               num_workers=self.hparams['num_workers'])
                    val_loader = torch.utils.data.DataLoader(val_data, 
                                                        self.hparams['batch_size'], 
                                                        shuffle=False, 
                                                        collate_fn= 
                                                                NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                    self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                        num_workers=self.hparams['num_workers'])
                    
                    # select experiment if does not exist create it 
                    # an experiment is created for each appliance
                    mlflow.set_experiment(f'{appliance_name}')
                    
                    # Start a new for the current appliance
                    with mlflow.start_run(): 
                        # Auto log all MLflow from lightening
                        mlflow.pytorch.autolog()  
                        # Save the run ID to use in testing phase
                        self.run_id[appliance_name] =  mlflow.active_run().info.run_id
                        # Log parameters of current run 
                        mlflow.log_params(self.hparams)
                        # Model Training
                        self.train_model(
                            appliance_name, 
                            train_loader, 
                            val_loader,
                            exp_name,
                            data.mean if self.hparams['target_norm'] == 'z-norm' else 0,
                            data.std if self.hparams['target_norm'] == 'z-norm' else 1)

        new_params = {"checkpoints_path": original_checkpoint }
        self.hparams.update(new_params)
        
    def train_model(self,
                    appliance_name,
                    train_loader, 
                    val_loader, 
                    exp_name,
                    mean = None,
                    std = None,
                    trial_idx= None,
                    fold_idx= None,
                    model =None):
        """Trains a single PyTorch model.

        :param appliance_name: Name of teh appliance to be modeled
        :type appliance_name: str
        :param train_loader: training dataLoader for the current appliance
        :type train_loader: DataLoader
        :param val_loader: validation dataLoader for the current appliance
        :type val_loader: DataLoader
        :param exp_name: the name of the experiment
        :type exp_name: str
        :param mean: mean value of the target appliance power. Defaults to None.
        :type mean: float, optional
        :param std: std value of the target applaince power. Defaults to None.
        :type std: float, optional
        :param trial_idx: ID of the current optuna trial if optuna is used. Defaults to None.
        :type trial_idx: int, optional
        :param fold_idx:  the number of the fold if CV is used. Defaults to None.
        :type fold_idx: int, optional
        :param model: Lightning model of the current appliance. Defaults to None.
        :return: in the case of using Optuna, it return the best validation loss and the path to the best checkpoint.
        :rtype:  tuple(int, str)
        """
        
        chkpt_path = self.hparams['checkpoints_path']
        version = ''

        if trial_idx is not None:
            chkpt_path += f"/trial_{trial_idx}" 
            version += f"/trial_{trial_idx}" 

        if fold_idx is not None:
            chkpt_path += f"/fold_{fold_idx}" 
            version += f"/fold_{fold_idx}" 
        
        

        best_checkpoint=get_latest_checkpoint(chkpt_path) 
        model = model if model is not None else self.models[appliance_name] 
        
        if self.hparams['target_norm'] == 'z-norm':
            self.appliance_params[appliance_name] = {
                'mean': mean,
                'std': std
                }
    
        
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath = chkpt_path, 
                                               monitor = 'val_mae', 
                                               mode="min", 
                                               save_top_k = 1)
        
        early_stop_callback = pl.callbacks.EarlyStopping(monitor ='val_mae',
                                             min_delta=1e-4,
                                             patience = self.hparams['patience_check'], 
                                             mode="min")
                                             
        logger  = DictLogger(self.hparams['logs_path'],
                        name = exp_name,
                        version = "single_appliance_experiment" + version if version !='' else "single_appliance_experiment" )
        
        trainer = pl.Trainer(logger = logger,
                gradient_clip_val=self.hparams['clip_value'],
                # checkpoint_callback=checkpoint_callback,
                max_epochs = self.hparams['max_nb_epochs'],
                callbacks=[early_stop_callback, checkpoint_callback],
                gpus=-1 if torch.cuda.is_available() else None,
                resume_from_checkpoint=best_checkpoint if not self.hparams['use_optuna'] else None) 
        
        if self.hparams['train']:
            trainer.fit(model, train_loader, val_loader)  
        
        if len(logger.metrics) >= 2:
            if self.hparams['use_optuna']:
                # TODO: Get the minimal validation loss and not the last one
                return logger.metrics[-2]["val_loss"], checkpoint_callback.best_model_path

    def multi_appliance_disaggregate(self, test_main_list, model=None,do_preprocessing=True):
        return None

    def multi_appliance_fit(self):
        """
        Train the specified models for each appliance separately taking into consideration
        the use of cross-validation and hyper-parameters optimisation. The checkpoints for 
        each model are saved in the correspondng path.
        """        
        self.exp_name = f"{self.hparams['model_name']}_{self.hparams['data']}_{self.hparams['experiment_label']}"
        original_checkpoint = self.hparams['checkpoints_path']
        
        
        exp_name = f"{self.exp_name}_multi_app"
        checkpoints = Path(original_checkpoint +f"{exp_name}")
        checkpoints.mkdir(parents=True, exist_ok=True)
        #update checkpoint path
        new_params = {"checkpoints_path": original_checkpoint +f"{exp_name}"}
        self.hparams.update(new_params)
        
        print(f"fit model for {exp_name}")
        
        if self.hparams['use_optuna']:
            # Use Optuna fot parameter optimisation of the model
            study = optuna.create_study(study_name=exp_name, direction="minimize")
            self.optuna_params ={
                'power' :self._data['targets'],
                'appliance_name':'Multi-appliance',
                'exp_name': exp_name
                }
            
            if self.hparams['kfolds'] <= 1:
                study.optimize(self.objective, n_trials=self.hparams['n_trials'], callbacks=[self.save_best_model])
                # Load weights of the model
                app_model, _ = self.get_net_and_loaders()
                #TODO: load checkpoints
                chechpoint=torch.load(study.user_attrs["path"])
                
                model = pilModel(app_model, self.hparams)
                model.hparams['checkpoint_path'] = study.user_attrs["path"]
                model.load_state_dict(chechpoint['state_dict'])
                model.eval()
        
                # Save best model for testing time
                self.models['Multi-appliance'] = model
            else:
                study.optimize(self.objective_cv, n_trials=self.hparams['n_trials'], callbacks=[self.save_best_model])
            
            # Save figures
            try:
                fig1 = optuna.visualization.plot_param_importances(study)
                fig2 = optuna.visualization.plot_parallel_coordinate(study)
                
                fig2.write_image(self.hparams['checkpoints_path'] +'/_parallel_coordinate.pdf')
                fig1.write_image(self.hparams['checkpoints_path'] +'/_param_importance.pdf')
            except:
                pass
            results_df = study.trials_dataframe()
            results_df.to_csv( f'{self.hparams["checkpoints_path"]}/Seq2Point_Study_{exp_name}_Multi-appliance.csv')
            joblib.dump(study, f'{self.hparams["checkpoints_path"]}/Seq2Point_Study_{exp_name}_Multi-appliance.pkl')
            
            # Restoring the best model and use it for the testing
            self.best_trials['Multi-appliance'] = study.best_trial.number 
            app_model, _ = self.get_net_and_loaders()
            #TODO: load checkpoints
            self.run_id['Multi-appliance'] =  study.user_attrs["best_run_id"]
            
        else:
            # Check if the appliance was already trained. If not then create a new model for it
            
            if self.hparams['kfolds'] > 1:
                self.models['Multi-appliance'] ={}
                # Getting the required data for the appliance
                _, dataloader = self.get_net_and_loaders()
                self.data_loaders['Multi-appliance']=dataloader
                dataset = dataloader(inputs=self._data['features'], targets=self._data['targets'])
                
                # Splitting the data into several folds
                fold = TimeSeriesSplit(n_splits=self.hparams['kfolds'], test_size=self.hparams['test_size'], gap = self.hparams['gap'])
                scores = []
                
                # Using each fold as a validation data
                for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(dataset)))):
                    print(f'started training for the fold {fold_idx}.')
                    app_model, _ = self.get_net_and_loaders()
                    self.models['Multi-appliance'][f'fold_{fold_idx}'] = pilModel(app_model, self.hparams)
                    train_data = torch.utils.data.Subset(dataset, train_idx)
                    val_data = torch.utils.data.Subset(dataset, valid_idx)
                    
                    train_loader = torch.utils.data.DataLoader(train_data, 
                                                               self.hparams['batch_size'], 
                                                               shuffle=True, 
                                                               collate_fn= 
                                                                NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                    self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                               num_workers=self.hparams['num_workers'])
                    
                    val_loader = torch.utils.data.DataLoader(val_data, 
                                                        self.hparams['batch_size'], 
                                                        shuffle=False, 
                                                        collate_fn= 
                                                                NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                    self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                        num_workers=self.hparams['num_workers'])
                
                    # select experiment if does not exist create it 
                    # an experiment is created for each appliance
                    mlflow.set_experiment(f'Multi-appliance')
                    
                    
            
                    # Start a new for the current appliance
                    with mlflow.start_run(run_name=self.hparams['model_name']): 
                        # Auto log all MLflow from lightening
                        mlflow.pytorch.autolog()  
                        # Save the run ID to use in testing phase
                        self.run_id['Multi-appliance'] =  mlflow.active_run().info.run_id
                        # Log parameters of current run 
                        mlflow.log_params(self.hparams)
                        # Model Training
                        mae_loss = self.train_model('Multi-appliance', 
                                                    train_loader, 
                                                    val_loader,
                                                    exp_name,
                                                    dataset.mean if self.hparams['target_norm'] == 'z-norm' else None,
                                                    dataset.std  if self.hparams['target_norm'] == 'z-norm' else None,
                                                    fold_idx = fold_idx,
                                                    model=self.models['Multi-appliance'][f'fold_{fold_idx}'])
                    
                    scores.append(mae_loss)
                                        
                
        
            else:
                if 'Multi-appliance' not in self.models:
                    print("First model training for Multi-appliance model")
                    #select model and data loaders to use
                    net, dataloader = self.get_net_and_loaders()
                    self.models['Multi-appliance'] = pilModel(net, self.hparams)
                    self.data_loaders['Multi-appliance'] = dataloader
                # Retrain the particular appliance
                else:
                    print("Started Retraining Muti-appliance model")
                
                
                dataloader  = self.data_loaders['Multi-appliance']
        
                data = dataloader(inputs=self._data['features'], targets=self._data['targets'])
                
                train_data, val_data=torch.utils.data.random_split(data, 
                                                     [int(data.len*(1-0.15)), data.len - int(data.len*(1-0.15))], 
                                                     generator=torch.Generator().manual_seed(42))
                train_loader = torch.utils.data.DataLoader(train_data, self.hparams['batch_size'], shuffle=True, 
                                                           collate_fn= 
                                                            NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                self.hparams, sample=True)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                           num_workers=self.hparams['num_workers'])
                val_loader = torch.utils.data.DataLoader(val_data, 
                                                    self.hparams['batch_size'], 
                                                    shuffle=False, 
                                                    collate_fn= 
                                                            NILM_MODELS[self.hparams['model_name']]['extra_params']['collate_fns'](
                                                                self.hparams, sample=False)  if 'collate_fns' in  NILM_MODELS[self.hparams['model_name']]['extra_params'] else None,
                                                    num_workers=self.hparams['num_workers'])
                
                # select experiment if does not exist create it 
                # an experiment is created for each appliance
                mlflow.set_experiment(f'Multi-appliance')
                
                # Start a new for the current appliance
                with mlflow.start_run(): 
                    # Auto log all MLflow from lightening
                    mlflow.pytorch.autolog()  
                    # Save the run ID to use in testing phase
                    self.run_id['Multi-appliance'] =  mlflow.active_run().info.run_id
                    # Log parameters of current run 
                    mlflow.log_params(self.hparams)
                    # Model Training
                    self.train_model(
                        'Multi-appliance', 
                        train_loader, 
                        val_loader,
                        exp_name,
                        data.mean if self.hparams['target_norm'] == 'z-norm' else 0,
                        data.std if self.hparams['target_norm'] == 'z-norm' else 1)
        new_params = {"checkpoints_path": original_checkpoint }
        self.hparams.update(new_params)   

     
        
            
                   
