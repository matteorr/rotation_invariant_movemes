from train_model.train_model import train_model
from utilities.utilities import utilities
import os

## DATASET_PATH: path to the dataset with the annotations
DATASET_PATH = '../inputs/CO_LSP_train2016.json'

## IMAGES_PATH: path of the actual images of LSP
IMAGES_PATH = '/Users/matteorr/Desktop/lowrank_upload/inputs/lsp_dataset/images/'

## SAVE_PATH: path of the folder to save the resulting model data
SAVE_PATH       = '../data'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

save_dataset        = True
save_partial_models = False

# number of latent factors to learn
num_factors = 10

# activities to include in the analysis
#   - 'athletics'
#   - 'badminton'
#   - 'baseball'
#   - 'gymnastics'
#   - 'parkour'
#   - 'soccer'
#   - 'tennis'
#   - 'volleyball'
#   - 'other'
activities = ['athletics', 'badminton', 'soccer', 'tennis', 'volleyball', 'baseball']

# model that should be used to learn the basis pose factorization
#   - 'svd':
#   - 'bucketed_svd_2d':
#   - 'bucketed_svd_3d':
#   - 'lfa_2d':
#   - 'lfa_3d':
model_type = 'lfa_2d'

# initialization for the U and V matrices in the lfa3d model
#   - 'random':
#   - 'svd':
init_type = 'random'

# type of angle annotations for initializing the angle of view of each pose
#   - 'random':
#   - 'heuristic':
#   - 'coarse':
#   - 'gt':
bucketing_metric = 'gt'

# number of pose clusters to use for discretizing the angles of view
num_buckets = 8

# objective function for the stochastic gradient descent
#   - 'l1_reg':
#   - 'l2_reg':
#   - 'l2_l1_ista_reg':
objective_f_type = 'l2_reg'

hyper_params = dict()
hyper_params['l_rate_U']           = 1e-4
hyper_params['l_rate_V']           = 1e-4
hyper_params['m_rate_U']           = 1e-5
hyper_params['m_rate_V']           = 1e-5
hyper_params['l_rate_theta']       = 1e-5
hyper_params['m_rate_theta']       = 1e-6
hyper_params['positive_V_flag']    = True
hyper_params['rmse_tolerance']     = 1e-5
hyper_params['lr_decay']           = 0.5
hyper_params['obj_func_tolerance'] = 1e4
hyper_params['UV_batch_step']      = 1e3
hyper_params['theta_batch_step']   = 1e3
hyper_params['lr_bound']           = 1e-6
hyper_params['max_iter']           = int(1e7)
hyper_params['error_window']       = 5

# Model training
train_model_obj = train_model(
                    # path to the dataset json file and images
                    dataset_path=DATASET_PATH, images_path=IMAGES_PATH,
                    # path at which models are saved and flags
                    save_path=SAVE_PATH, save_dataset=save_dataset,
                    save_partial_models=save_partial_models,
                    # number of latent factors
                    num_factors=num_factors,
                    # actions to exclude from the analysis
                    activity_list=activities,
                    # model type trained
                    model_type=model_type,
                    # initialization for lfa3d model
                    init_type=init_type,
                    # type of angle bucketing (heuristic, random or gt based)
                    bucketing_metric=bucketing_metric,
                    # number of buckets to cluster poses
                    num_buckets=num_buckets,
                    # objective function type for the SGD procedures
                    objective_f_type=objective_f_type,
                    # a dictionary containing all the optimization hyperparams
                    hyper_params=hyper_params,
                    # provide an input matrix for U
                    U_test=None
                )

train_model_obj.train()

colors = ['g',
           'g',
           'y',
           'y',
           'r',
           'b',
           'r',
           'b',
           'y',
           'y',
           'm',
           'c',
           'm',
           'c']
utility_obj = utilities(train_model_obj, colors, basis_coeff=10)
utility_obj.plot_movemes()
