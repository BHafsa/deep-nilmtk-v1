from argparse import ArgumentParser


def get_exp_parameters():
    """

    Defines the default values for the hyper-parameters of the experiment.

    :return: A dictionnary with values of the hyper-parameters
    :rtype: dict
    """

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--use_optuna', type=bool, default=False)
    parser.add_argument('--log_artificat', type=bool, default=False)
    parser.add_argument('--max_nb_epochs', type=int, default=20)

    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip_value', type=int, default=10)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    # parser.add_argument('--eps', default=1e-8, type=float)
    # parser.add_argument('--dropout', default=0.25, type=float)
    # parser.add_argument('--pool_filter', default=16, type=int)
    # parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    # parser.add_argument('--features_start', default=16, type=int)
    # parser.add_argument('--latent_size', default=64, type=int)
    # parser.add_argument('--num_gauss', default=5, type=int)
    # parser.add_argument('--min_std', default=0.1, type=float)
    # parser.add_argument('--n_layers', default=4, type=int)
    parser.add_argument('--out_size', default=1, type=int)
    parser.add_argument('--in_size', default=99, type=int)
    parser.add_argument('--border', default=0, type=int)
    parser.add_argument('--multi_appliance', default=False, type=bool)

    parser.add_argument('--custom_preprocess', default=None, type=int)
    parser.add_argument('--custom_postprocess', default=None, type=int)


    parser.add_argument('--patience_optim', default=5, type=int)
    parser.add_argument('--patience_check', default=5, type=int)

    # parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--experiment_label', default='', type=str)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--feature_type', default="combined", type=str)
    # parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--seed', default=3407, type=float)
    parser.add_argument('--main_mu', default=150.0, type=float)
    parser.add_argument('--main_std', default=350.0, type=float)
    parser.add_argument('--input_norm', default="z-norm", type=str)
    parser.add_argument('--q_filter', default=None, type=dict)
    # parser.add_argument('--sample_second', default=6, type=int)
    parser.add_argument('--seq_type', default="seq2point", type=str)
    parser.add_argument('--point_position', default="last_position", type=str)
    parser.add_argument('--target_norm', default="lognorm", type=str)

    parser.add_argument('--threshold_method', default="at", type=str)

    parser.add_argument('--model_class', default=None, type=object)
    parser.add_argument('--loader_class', default=None, type=object)

    parser.add_argument('--train', default=1, type=int)

    parser.add_argument('--kfolds', default=1, type=int)
    parser.add_argument('--gap', default=0, type=int)
    parser.add_argument('--test_size', default=None, type=int)

    parser.add_argument('--z_dim', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--mdn_dist_type', default="normal", type=str)
    parser.add_argument('--data', default="UKDALE", type=str)
    parser.add_argument('--quantiles', default=[0.1, 0.25, 0.5, 0.75, 0.9], type=list)
    parser.add_argument('--logs_path', default="logs", type=str)
    # parser.add_argument('--results_path', default="results\\", type=str)
    parser.add_argument('--figure_path', default="figures", type=str)
    parser.add_argument('--checkpoints_path', default="checkpoints", type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--version', default=0, type=int)
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

    return parser
