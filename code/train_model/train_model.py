from data_loading.data_loading import data_loader
from bucketing.bucketing import bucketer
from factorization.factorization import factorization

import json, pickle
import numpy as np
import sys, os
import datetime, time

"""
Train a model using the given parameters
"""
class train_model:
    def __init__(self,
                 dataset_path, images_path,
                 save_path, save_dataset, save_partial_models,
                 num_factors,
                 activity_list,
                 model_type,
                 init_type,
                 bucketing_metric,
                 num_buckets,
                 objective_f_type,
                 hyper_params,
                 U_test=None):
        print "[TRAIN_MODEL]: Initializing..."
        tic = time.time()

        self.dataset_path = dataset_path
        self.images_path  = images_path

        self.save_dataset        = save_dataset
        self.save_partial_models = save_partial_models

        self.num_factors   = num_factors
        self.activity_list = activity_list
        self.model_type    = model_type
        self.init_type     = init_type

        self.bucketing_metric = bucketing_metric
        self.num_buckets      = num_buckets

        self.objective_f_type = objective_f_type

        self.hyper_params = hyper_params

        self.dataset_obj = None

        # create a directory for storing results
        self.save_dir_name = \
          'model[%s]_numfact[%d]_angleanns[%s]_numbuckets[%d]_init[%s]_objfunc[%s]'\
          %(model_type,num_factors,bucketing_metric,num_buckets,init_type,objective_f_type)

        if not os.path.exists(save_path + '/' + self.save_dir_name):
            os.system('mkdir %s'%(save_path + '/' + self.save_dir_name))

        self.save_path    = save_path + '/' + self.save_dir_name
        self.time_str     = datetime.datetime.strftime(\
                             datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_plots_path  = self.save_path + '/' + self.time_str + '/plots'
        self.save_models_path = self.save_path + '/' + self.time_str + '/models'

        self.U_test             = U_test

        print "[TRAIN_MODEL]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def save_init(self):
        os.system('mkdir %s'%(self.save_path + '/' + self.time_str))
        os.system('mkdir %s'%(self.save_plots_path))
        os.system('mkdir %s'%(self.save_models_path))

        # save json file with dictionary of all parameters
        params = {}
        params['dataset_path']         = self.dataset_path
        params['images_path']          = self.images_path
        params['model_type']           = self.model_type
        params['init_type']            = self.init_type
        params['objective_f_type']     = self.objective_f_type
        params['num_factors']          = self.num_factors
        params['num_buckets']          = self.num_buckets
        params['bucketing_metric']     = self.bucketing_metric
        params['save_path']            = self.save_path
        params['time_str']             = self.time_str
        params['save_plots_path']      = self.save_plots_path
        params['save_models_path']     = self.save_models_path
        params['activity_list']        = self.activity_list
        params['hyper_params']         = self.hyper_params

        with open(self.save_path + '/' + self.time_str + '/params.json','wb') as f:
            json.dump(params,f)

    def train(self):
        self.save_init()

        print "[TRAIN_MODEL]: Training..."
        tic = time.time()
        ########################################################################
        ## LOAD DATA
        ########################################################################
        data_loading_obj = data_loader(self.dataset_path,
                                       self.images_path,
                                       self.activity_list)

        if self.model_type in ["svd", "bucketed_svd_2d", "lfa_2d"]:
            dataset_obj = data_loading_obj.load2D()

        elif self.model_type in ["lfa_3d", "bucketed_svd_3d"]:
            dataset_obj  = data_loading_obj.load3D()

        else:
            raise ValueError("[TRAIN_MODEL]: Uknown model_type [%s]!"%self.model_type)

        if self.save_dataset:
            with open(self.save_path + '/' + self.time_str + '/dataset_obj.pkl','wb') as fp:
                pickle.dump(dataset_obj,fp)

        self.dataset_obj = dataset_obj

        ########################################################################
        ## FACTORIZATION MODEL IMPLEMENTING THE DIFFERENT MODELS
        ########################################################################
        fact_obj = factorization(dataset_obj, self.save_partial_models)

        ########################################################################
        ## TRAIN MODEL
        ########################################################################
        if self.model_type == 'svd':
            U, s, V, data_2d_mean = fact_obj.svd()

            U_ = np.zeros((1, U.shape[0], U.shape[1]))
            U_[0,:,:] = U

            self.U = U[:,:self.num_factors]
            self.V = V[:self.num_factors,:]
            self.data_2d_mean = data_2d_mean
            self.utilities_data = data_2d_mean
            pickle.dump(self.utilities_data, open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'wb'))
            pickle.dump(self.U, open('%s/U_final.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(self.V, open('%s/V_final.pkl'%(self.save_models_path), 'wb'))

        elif self.model_type == 'bucketed_svd_2d':
            bucketer_obj = bucketer(
                            dataset_obj,
                            self.num_buckets,
                            self.bucketing_metric)
            buckets_index, buckets_table = bucketer_obj.bucket_poses()

            Us, ss, Vs, mus = fact_obj.bucketed_svd_2d(buckets_index)

            buckets_mean_poses = []
            for b in xrange(self.num_buckets):
                buckets_mean_poses.append(mus[b,:])

            self.U = Us[:,:,:self.num_factors]
            self.V = Vs[:,:self.num_factors,:]
            self.buckets_index = buckets_index
            self.buckets_table = buckets_table
            self.buckets_mean_poses = mus
            self.utilities_data = mus

            pickle.dump(buckets_index, open('%s/bkindex.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(self.utilities_data, open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'wb'))
            pickle.dump(Us, open('%s/U_final.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(Vs, open('%s/V_final.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(ss, open('%s/sigma_final.pkl'%(self.save_models_path), 'wb'))

        elif self.model_type == 'bucketed_svd_3d':
            bucketer_obj = bucketer(
                            dataset_obj,
                            self.num_buckets,
                            self.bucketing_metric)
            buckets_index, buckets_table = bucketer_obj.bucket_poses()

            Us, ss, Vs, mus = fact_obj.bucketed_svd_3d(buckets_index)

            buckets_mean_poses = []
            for b in xrange(self.num_buckets):
                buckets_mean_poses.append(mus[b,:])

            self.U = Us[:,:,:self.num_factors]
            self.V = Vs[:,:self.num_factors,:]
            self.buckets_index = buckets_index
            self.buckets_table = buckets_table
            self.buckets_mean_poses = mus
            self.utilities_data = mus

            pickle.dump(buckets_index, open('%s/bkindex.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(self.utilities_data, open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'wb'))
            pickle.dump(Us, open('%s/U_final.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(Vs, open('%s/V_final.pkl'%(self.save_models_path), 'wb'))
            pickle.dump(ss, open('%s/sigma_final.pkl'%(self.save_models_path), 'wb'))

        elif self.model_type == 'lfa_2d':
            bucketer_obj = bucketer(
                            dataset_obj,
                            self.num_buckets,
                            self.bucketing_metric)
            buckets_index, buckets_table = bucketer_obj.bucket_poses()

            U, V, buckets_mean_poses = fact_obj.lfa_2d(
                        self.num_factors,
                        self.num_buckets,
                        buckets_index,
                        buckets_table,
                        self.objective_f_type,
                        self.hyper_params,
                        self.save_path + '/' + self.time_str)

            self.U = U
            self.V = V
            self.buckets_index = buckets_index
            self.buckets_table = buckets_table
            self.buckets_mean_poses = buckets_mean_poses
            self.utilities_data = buckets_mean_poses
            pickle.dump(self.utilities_data, open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'wb'))

        elif self.model_type == 'lfa_3d':
            bucketer_obj = bucketer(
                            dataset_obj,
                            self.num_buckets,
                            self.bucketing_metric)
            angle_anns = bucketer_obj.get_angle_anns()

            U, V, theta, data_3d_c_mean = fact_obj.lfa_3d(
                    self.init_type,
                    angle_anns,
                    self.num_factors,
                    self.objective_f_type,
                    self.hyper_params,
                    self.save_path + '/' + self.time_str, self.U_test)

            self.U = U
            self.V = V
            self.angles = theta
            self.data_3d_c_mean = data_3d_c_mean
            self.utilities_data = data_3d_c_mean
            pickle.dump(self.utilities_data, open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'wb'))

        else:
            raise ValueError("[TRAIN_MODEL]: Uknown model_type [%s]!"%self.model_type)

        print "[TRAIN_MODEL]: Done Training, (t=%.2fs)."%(time.time() - tic)

    def load(self, time_stamp):
        print "[TRAIN_MODEL]: Loading..."
        tic = time.time()

        load_path = self.save_path + '/' + time_stamp
        with open(load_path + '/dataset_obj.pkl','rb') as fp:
            dataset_obj = pickle.load(fp)
        self.dataset_obj = dataset_obj

        with open(load_path + '/params.json','rb') as fp:
            params = json.load(fp)

        self.dataset_path      = params['dataset_path']
        self.images_path       = params['images_path']
        self.model_type        = params['model_type']
        self.init_type         = params['init_type']
        self.objective_f_type  = params['objective_f_type']
        self.num_factors       = params['num_factors']
        self.num_buckets       = params['num_buckets']
        self.bucketing_metric  = params['bucketing_metric']
        self.save_path         = params['save_path']
        self.save_plots_path   = params['save_plots_path']
        self.time_str          = params['time_str']
        self.save_models_path  = params['save_models_path']
        self.hyper_params      = params['hyper_params']
        self.activity_list     = params['activity_list']

        models_path  = load_path + '/models/'
        models_files = [f for f in os.listdir(models_path) if \
                        os.path.isfile(os.path.join(models_path, f)) and 'final' in f]

        U_path = [f for f in models_files if 'U' in f][0]
        V_path = [f for f in models_files if 'V' in f][0]

        if self.model_type == 'lfa_3d':
            angles_path = [f for f in models_files if 'angles' in f][0]
            with open(load_path + '/models/' + angles_path,'rb') as fp:
                self.angles = pickle.load(fp)

        with open(load_path + '/models/' + U_path,'rb') as fp:
            self.U = pickle.load(fp)

        with open(load_path + '/models/' + V_path,'rb') as fp:
            self.V = pickle.load(fp)

        with open(self.save_path + '/' + self.time_str + '/utilities_data.pkl', 'rb') as fp:
            self.utilities_data = pickle.load(fp)

        print "[TRAIN_MODEL]: Done Loading, (t=%.2fs)."%(time.time() - tic)
