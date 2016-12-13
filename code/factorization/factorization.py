from gradient_descent.gradient_descent_2d import gradient_descent_2d
from gradient_descent.gradient_descent_3d import gradient_descent_3d

import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import svd as npsvd
import time
import copy, pickle

"""
Class for several baselines: pure kmeans, svd, lfa2d, lfa3d
"""
class factorization:
    def __init__(self, dataset_obj, save_partial_models):
        print "[FACTORIZATION]: Initializing..."
        tic = time.time()

        self.save_partial_models = save_partial_models

        self.data_2d = dataset_obj.data_2d
        self.data_3d = dataset_obj.data_3d

        self.num_dim_2d = dataset_obj.num_dim_2d
        self.num_dim_3d = dataset_obj.num_dim_3d

        self.num_images = dataset_obj.num_images
        self.num_joints = dataset_obj.num_joints

        self.real2new = dataset_obj.real2new
        self.new2real = dataset_obj.new2real

        self.x_portion_2d = dataset_obj.x_portion_2d
        self.y_portion_2d = dataset_obj.y_portion_2d

        self.x_portion_3d = dataset_obj.x_portion_3d
        self.y_portion_3d = dataset_obj.y_portion_3d
        self.z_portion_3d = dataset_obj.z_portion_3d

        print "[FACTORIZATION]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def svd(self):
        print "[FACTORIZATION]: Performing Regular SVD..."
        tic = time.time()

        # center data before performing SVD
        data_2d_mean = np.array(np.matrix(self.data_2d).mean(1))
        data_2d_centered = self.data_2d - data_2d_mean
        data_2d_mean = data_2d_mean.T[0]

        U, s, V = npsvd(data_2d_centered, full_matrices=False)
        print "[FACTORIZATION]: Done, (t=%.2fs)."%(time.time() - tic)

        return U, s, V, data_2d_mean

    def bucketed_svd_2d(self, buckets_index):
        print "[FACTORIZATION]: Performing 2D Bucketed SVD..."
        tic = time.time()

        num_buckets = len(buckets_index)

        Us  = np.zeros((num_buckets,self.num_dim_2d,self.num_dim_2d))
        ss  = np.zeros((num_buckets,self.num_dim_2d))
        Vs  = np.zeros((num_buckets,self.num_dim_2d,self.num_images))
        mus = np.zeros((num_buckets,self.num_dim_2d))

        # SVD is done separately for every bucket
        for b in range(num_buckets):
            bucket_idxs = [i['new_id'] for i in buckets_index[b]]
            bucket_data = self.data_2d[:,bucket_idxs]

            # center data before performing SVD
            data_2d_mean = np.array(np.matrix(bucket_data).mean(1))
            data_2d_centered = bucket_data - data_2d_mean
            mu = data_2d_mean.T[0]

            U, s, V = npsvd(data_2d_centered, full_matrices=False)

            if np.shape(U)[1] < np.shape(Us)[2]:
                # this means that there are less examples than dimensions
                # in the set we are doing PCA on.
                # result will be at most number of examples orthogonal directions
                # pad the rest and insert into U
                Us[b,:,0:np.shape(U)[1]] = U
            else:
                Us[b,:,:] = U

            ss[b,0:min(np.shape(s)[0],np.shape(ss)[1])] = s

            a = Vs[b,:,:]
            a[0:min(np.shape(V)[0],np.shape(Vs)[1]),bucket_idxs] = V
            Vs[b,:,:] = a

            mus[b,:]  = np.array(mu)

        print "[FACTORIZATION]: Done 2D Bucketed SVD, (t=%.2fs)."%(time.time()-tic)
        return Us, ss, Vs, mus

    def bucketed_svd_3d(self, buckets_index):
        print "[FACTORIZATION]: Performing 3D Bucketed SVD..."
        tic = time.time()

        num_buckets = len(buckets_index)

        Us  = np.zeros((num_buckets,self.num_dim_3d,self.num_dim_3d))
        ss  = np.zeros((num_buckets,self.num_dim_3d))
        Vs  = np.zeros((num_buckets,self.num_dim_3d,self.num_images))
        mus = np.zeros((num_buckets,self.num_dim_3d))

        # SVD is done separately for every bucket
        for b in range(num_buckets):
            bucket_idxs = [i['new_id'] for i in buckets_index[b]]
            bucket_data = self.data_3d[:,bucket_idxs]

            # center data before performing SVD
            data_3d_mean = np.array(np.matrix(bucket_data).mean(1))
            data_3d_centered = bucket_data - data_3d_mean
            mu = data_3d_mean.T[0]

            U, s, V = npsvd(data_3d_centered, full_matrices=False)

            if np.shape(U)[1] < np.shape(Us)[2]:
                # this means that there are less examples than dimensions
                # in the set we are doing PCA on.
                # result will be at most number of examples orthogonal directions
                # pad the rest and insert into U
                Us[b,:,0:np.shape(U)[1]] = U
            else:
                Us[b,:,:] = U

            ss[b,0:min(np.shape(s)[0],np.shape(ss)[1])] = s

            a = Vs[b,:,:]
            a[0:min(np.shape(V)[0],np.shape(Vs)[1]),bucket_idxs] = V
            Vs[b,:,:] = a

            mus[b,:]  = np.array(mu)

        print "[FACTORIZATION]: Done 3D Bucketed SVD, (t=%.2fs)."%(time.time()-tic)
        return Us, ss, Vs, mus

    def lfa_2d(self,
               num_factors,
               num_buckets,
               buckets_index,
               buckets_table,
               objective_f_type,
               hyper_params,
               save_dir):
        print "[FACTORIZATION]: Performing 2D Latent Factor Analysis..."
        tic = time.time()

        # get the mean pose for each bucket of poses
        buckets_mean_poses = []
        for b in xrange(num_buckets):
            bucket_idxs = [i['new_id'] for i in buckets_index[b]]
            bucket_data = self.data_2d[:,bucket_idxs]

            data_2d_bucket_mean = np.array(np.matrix(bucket_data).mean(1))
            mu = data_2d_bucket_mean.T[0]
            buckets_mean_poses.append(mu)

        # remove mean pose from each pose to obtain the movemes
        data_2d_centered = copy.deepcopy(self.data_2d)
        for idx in xrange(self.num_images):
            data_2d_centered[:,idx] -= buckets_mean_poses[buckets_table[idx]]

        # perform gradient descent over U and V to minimize reconstruction error
        grad_obj = gradient_descent_2d(
                    num_factors,
                    data_2d_centered,
                    self.x_portion_2d,
                    self.y_portion_2d,
                    self.num_images,
                    self.num_dim_2d,
                    self.num_joints,
                    buckets_index,
                    buckets_table,
                    num_buckets,
                    objective_f_type,
                    hyper_params,
                    self.save_partial_models,
                    save_dir)

        # by default lfa2d model is initialized with random U and V
        grad_obj.initialize_lfa()
        U, V = grad_obj.solve()

        print "[FACTORIZATION]: Done 2D Latent Factor Analysis (t=%.2fs)."\
              %(time.time()-tic)
        return U, V, buckets_mean_poses

    def lfa_3d(self,
               init_type_3d,
               angle_anns,
               num_factors,
               objective_f_type,
               hyper_params,
               save_dir,
               U_test=None):
        print "[FACTORIZATION]: Performing 3D Latent Factor Analysis..."
        tic = time.time()

        # rotate data to face direction with angle 270 degrees (frontal view).
        neg_angle_anns = [{'angle_rad': np.pi * 3.5 - a['angle_rad'],
                           'new_id':  a['new_id'],
                           'real_id': a['real_id']} for a in angle_anns]

        # the 3d poses rotated to frontal view
        data_3d_c = self.rotate_data(self.data_3d, neg_angle_anns)

        # demean rotated data to obtain movement information on rotated data
        data_3d_c_mean = np.array(np.matrix(data_3d_c).mean(1))
        data_3d_c_moves = data_3d_c - data_3d_c_mean

        # we do not demean the 2d data cause in lfa 3d we learn movements when
        # poses are in 270 (frontal view) and apply the movements to mean pose
        # 3d facing 270 then rotate by angle alpha and project to 2d and compare
        # to the ground truth 2d pose
        grad_obj = gradient_descent_3d(
                    self.data_2d,
                    data_3d_c_mean,
                    data_3d_c_moves,
                    angle_anns,
                    self.num_images, self.num_dim_2d, self.num_dim_3d,
                    self.x_portion_2d, self.y_portion_2d,
                    self.x_portion_3d, self.y_portion_3d, self.z_portion_3d,
                    num_factors, self.num_joints,
                    objective_f_type,
                    hyper_params,
                    self.save_partial_models,
                    save_dir)

        grad_obj.initialize(init_type_3d, U_test)

        U, V, theta = grad_obj.solve()

        print "[FACTORIZATION]: Done 3D Latent Factor Analysis (t=%.2fs)."\
              %(time.time()-tic)
        return U, V, theta, data_3d_c_mean

    def rotate_data(self, data_3d, angle_anns):
        data_rotated = np.zeros(data_3d.shape)

        for a in angle_anns:
            alpha_rot = a['angle_rad']
            new_idx   = a['new_id']
            real_idx  = a['real_id']

            # rotation about the vertical Y axis
            x_rot = [np.cos(alpha_rot), 0, np.sin(alpha_rot)]
            y_rot = [0, 1, 0]
            z_rot = [-np.sin(alpha_rot), 0, np.cos(alpha_rot)]
            R_mat = np.vstack((x_rot,y_rot,z_rot))

            # extract joints from image
            joints_mat = np.zeros((3,self.num_joints))
            joints_mat[0,:] = \
                data_3d[self.x_portion_3d,new_idx]
            joints_mat[1,:] = \
                data_3d[self.y_portion_3d,new_idx]
            joints_mat[2,:] = \
                data_3d[self.z_portion_3d,new_idx]

            # apply rotation
            rotated_joints = np.dot(R_mat,joints_mat)

            # put joints back in data format
            data_rotated[:,new_idx] = np.hstack(
               (rotated_joints[0,:],
                rotated_joints[1,:],
                rotated_joints[2,:]))

        return data_rotated
