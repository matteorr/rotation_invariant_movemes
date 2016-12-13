import numpy as np, math
import datetime, time, sys
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import svd as npsvd
import copy

"""
Stochastic Gradient Descent for 2D Latent Factor Model
"""

class gradient_descent_2d:
    def __init__(self,
                 num_factors,
                 data_2d,
                 x_portion_2d,
                 y_portion_2d,
                 num_images,
                 num_dim_2d,
                 num_joints,
                 buckets_index,
                 buckets_table,
                 num_buckets,
                 objective_f_type,
                 hyper_params,
                 save_partial_models,
                 save_dir):

        print "[GRADIENT_DESCENT]: Initializing..."
        tic = time.time()

        self.num_factors   = num_factors
        self.data_2d       = data_2d
        self.x_portion_2d  = x_portion_2d
        self.y_portion_2d  = y_portion_2d
        self.num_images    = num_images
        self.num_dim_2d    = num_dim_2d
        self.num_joints    = num_joints
        self.buckets_index = buckets_index
        self.buckets_table = buckets_table
        self.num_buckets   = num_buckets

        self.obj_type        = objective_f_type
        self.positive_V_flag = hyper_params['positive_V_flag']
        
        self.l_rate_U     = hyper_params['l_rate_U']
        self.m_rate_U     = hyper_params['m_rate_U']
        self.l_rate_V     = hyper_params['l_rate_V'] 
        self.m_rate_V     = hyper_params['m_rate_V'] 

        self.lamb            = 0.001
        self.kappa           = 0.001
        self.lr_bound        = hyper_params['lr_bound']
        self.max_iter        = hyper_params['max_iter']
        self.UV_batch_step   = hyper_params['UV_batch_step']
        self.error_window    = hyper_params['error_window']

        self.rmse_tolerance     = hyper_params['rmse_tolerance']
        self.lr_decay           = hyper_params['lr_decay']
        self.obj_func_tolerance = hyper_params['obj_func_tolerance']

        self.save_partial_models = save_partial_models

        self.buckets_mean  = []
        self.buckets_order = []
        
        self.save_dir = save_dir

        print "[GRADIENT_DESCENT]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def initialize_lfa(self):
        # compute bucket mean angles
        for b in xrange(self.num_buckets):
            bucket_angles = [a['angle_rad'] for a in self.buckets_index[b]]

            unit_alphas = [(np.cos(a),np.sin(a)) for a in bucket_angles]
            unit_mean   = [sum(a) / float(len(a)) for a in zip(*unit_alphas)]

            centroid = math.degrees(math.atan2(unit_mean[1],unit_mean[0]))
            if centroid < 0: centroid += 360
            elif centroid > 360: centroid -= 360
            self.buckets_mean.append(centroid)
        self.buckets_order = np.argsort(self.buckets_mean)

        # compute initial U and V using random values
        self.U = np.zeros((self.num_buckets, self.num_dim_2d, self.num_factors))
        self.V = np.zeros((self.num_buckets, self.num_factors, self.num_images))

        # randomly initialize between -1 and 1
        u = np.random.random((self.num_dim_2d, self.num_factors)) * 2 - 1
        v = np.random.random((self.num_factors, self.num_images)) * 2 - 1
        for b in xrange(self.num_buckets):
            b_idxs = [x['new_id'] for x in self.buckets_index[b]]
            self.U[b,:,:]      = u
            self.V[b,:,b_idxs] = v[:,b_idxs].T

    def resume(self, U_file_name, V_file_name):
        raise NotImplementedError

    def solve(self):
        print "[GRADIENT_DESCENT]: Solving with objective function [%s]."%(self.obj_type)
        start_time = time.time()
        np.seterr(all='raise')

        rmse_prev  = self.compute_rmse()
        obj_prev   = self.objective_f()
        init_error = rmse_prev
        init_obj   = obj_prev

        print "=" * 30
        print " * Initial RMSE: %f"%(rmse_prev)
        print " * Initial Obj:  %f"%(obj_prev)
        print '-' * 30
        print " * l_rate_U: %f"%(self.l_rate_U)
        print " * l_rate_V: %f"%(self.l_rate_V)
        print "=" * 30

        # Save the errors/objective for record (plotting)
        errors = [rmse_prev]
        objs   = [obj_prev]

        # Previous weight update info for momentum
        dux_prev = np.zeros((self.num_buckets, self.num_factors))
        duy_prev = np.zeros((self.num_factors))
        dv_prev  = np.zeros((self.num_factors))

        # Loop and repeat updating U and V matrices
        it_time = time.time()
        for it in xrange(self.max_iter):

            # Randomly select one datapoint
            _i = np.random.randint(self.num_joints)
            _j = np.random.randint(self.num_images)
            _b = self.buckets_table[_j]

            # Compute the local gradient of U at the selected datapoint
            # If dealing with an x coordinate, update each slice of U with
            # gradients computed from all exaples in the corresponding bucket
            # If dealing with a y coordinate, update all slices of U equally
            if self.obj_type == 'l2_reg':
                dux = self.grad_ux_l2(_i)
                duy = self.grad_uy_l2(_i)

            elif self.obj_type == 'l1_reg':
                dux = self.grad_ux_l1(i, j, bucket)
                duy = self.grad_uy_l1(i+self.num_joints, j, bucket)

            else:
                raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

            # Compute the local gradient of V at the selected datapoint
            # gradient of V is computed over all joints i for the selected _j
            if self.obj_type == 'l2_reg':
                dv = self.grad_v_l2(_j)

            elif self.obj_type == 'l1_reg':
                dv = self.grad_v_l1(_j)

            else:
                raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

            u_x_coord = _i
            u_y_coord = _i + self.num_joints

            # Update U gradients x portion
            for b in xrange(self.num_buckets):
                cur_bucket = self.buckets_order[b]
                self.U[cur_bucket,u_x_coord,:] -= self.l_rate_U * dux[cur_bucket,:]
                self.U[cur_bucket,u_x_coord,:] -= self.m_rate_U * dux_prev[cur_bucket,:]

            # Update U gradients y portion
            duy_sum = sum(duy)
            for b in xrange(self.num_buckets):
                cur_bucket = self.buckets_order[b]
                self.U[cur_bucket,u_y_coord,:] -= self.l_rate_U * duy_sum
                self.U[cur_bucket,u_y_coord,:] -= self.m_rate_U * duy_prev

            # Update V gradients
            self.V[_b,:,_j] -= self.l_rate_V * dv
            self.V[_b,:,_j] -= self.m_rate_V * dv_prev

            # store gradients for momentum term
            dux_prev = dux
            duy_prev = duy_sum
            dv_prev  = dv

            # Soft thresholding for l1 regularization
            if self.obj_type == 'l1_reg':
                for j in xrange(self.num_factors):

                    if self.U[a,i,j] >= self.l_rate_U * self.lamb:
                        self.U[a,i,j] -= self.l_rate_U * self.lamb

                    elif self.U[a,i,j] < self.l_rate_U * self.lamb:
                        self.U[a,i,j] += self.l_rate_U * self.lamb

                    else:
                        self.U[a,i,j] = 0.

            # Compute errors/objectives in a batched intervals for faster iteration
            if (it+1) % self.UV_batch_step == 0:
                print "Iteration [%d] t=[%.2f s]"%(it+1,time.time() - it_time)

                # Compute error
                rmse_curr = self.compute_rmse()
                errors.append(rmse_curr)
                delta = rmse_curr - rmse_prev
                diff  = rmse_curr - init_error

                # Compute objective
                obj_curr = self.objective_f()
                objs.append(obj_curr)
                delta_obj = obj_curr - obj_prev
                obj_diff  = obj_curr - init_obj

                print "=" * 30
                print " OBJ   : %f"%(obj_curr)
                print " Batch : %f"%(delta_obj)
                print " Total : %f"%(obj_diff)
                print '-' * 30
                print " RMSE  : %f"%(rmse_curr)
                print " Batch : %f"%(delta)
                print " Total : %f"%(diff)
                print '-' * 30
                print " l_rate_U: %f"%(self.l_rate_U)
                print " l_rate_V: %f"%(self.l_rate_V)

                if (np.abs(delta) < self.rmse_tolerance):
                    if self.l_rate_U > self.lr_bound:
                        self.l_rate_U *= self.lr_decay
                        print " ==> Learning rate U Updated"
                    if self.l_rate_V > self.lr_bound:
                        self.l_rate_V *= self.lr_decay
                        print " ==> Learning rate V Updated"

                # Observe if the relative change of the objective is less than certain
                # threshold for termination
                if len(objs) >= 2 * self.error_window:
                    curr_batch = np.mean(objs[len(objs)-self.error_window:it])
                    prev_batch = np.mean(objs[len(objs)-2*self.error_window:it-self.error_window])
                    print '-' * 30
                    print " Average OBJ Delta check: "
                    print "  Prev Batch : %f"%(prev_batch)
                    print "  Curr Batch : %f"%(curr_batch)
                    if prev_batch - curr_batch < self.obj_func_tolerance:
                        break

                rmse_prev = rmse_curr
                obj_prev  = obj_curr

                it_time = time.time()

                if self.save_partial_models:
                    time_str = datetime.datetime.fromtimestamp(
                                time.time()).strftime('%Y_%m_%d_%H_%M_%S')
                    U_file_path = '%s/models/U_%s_%s_%s_%s_%s_%s.pkl'%\
                                   (self.save_dir, self.num_dim_2d, self.num_images,
                                    self.num_factors, it+1, self.obj_type, time_str)
                    V_file_path = '%s/models/V_%s_%s_%s_%s_%s_%s.pkl'%\
                                   (self.save_dir, self.num_dim_2d, self.num_images,
                                    self.num_factors, it+1, self.obj_type, time_str)
                    pickle.dump(self.U, open(U_file_path, 'wb'))
                    pickle.dump(self.V, open(V_file_path, 'wb'))

        end_time = time.time() - start_time
        print "#" * 50
        print "[GRADIENT_DESCENT]: SUMMARY"
        print "  * Optimization Completed in t=[%.2f] secs"%(end_time)
        print "  * Final Obj           : %f"%(obj_curr)
        print "  * Final Obj Change    : %f"%(obj_diff)
        print "  * Final RMSE          : %f"%(rmse_curr)
        print "  * Final RMSE Change   : %f"%(diff)
        print "-" * 50
        print "  * Learning Rate U     : %f"%(self.l_rate_U)
        print "  * Learning Rate V     : %f"%(self.l_rate_V)
        print "#" * 50

        time_str = datetime.datetime.fromtimestamp(
                    time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        U_file_path = '%s/models/U_final_%s_%s_%s_%s_%s_%s.pkl'%\
                       (self.save_dir, self.num_dim_2d, self.num_images,
                        self.num_factors, it+1, self.obj_type, time_str)
        V_file_path = '%s/models/V_final_%s_%s_%s_%s_%s_%s.pkl'%\
                       (self.save_dir, self.num_dim_2d, self.num_images,
                        self.num_factors, it+1, self.obj_type, time_str)
        pickle.dump(self.U, open(U_file_path, 'wb'))
        pickle.dump(self.V, open(V_file_path, 'wb'))

        pickle.dump(errors, open('%s/errors.pkl'%(self.save_dir), 'wb'))
        self.plot_error(errors, '%s/plots/errors.png'%(self.save_dir),
                                'Reconstruction Error (RMSE)',
                                'Number of Iterations (x%d)'%(self.UV_batch_step),
                                'RMSE')
        pickle.dump(objs, open('%s/objectives.pkl'%(self.save_dir), 'wb'))
        self.plot_error(objs, '%s/plots/objectives.png'%(self.save_dir),
                              'Objective Value',
                              'Number of Iterations (x%d)'%(self.UV_batch_step),
                              'Obj')

        print "[GRADIENT_DESCENT]: Done, (t=%.2fs)."%(time.time() - start_time)
        return self.U, self.V

    def compute_rmse(self):
        """
        Compute mean-squared error using the current U, V matrix.
        Return the average RMSE across all clusters.
        """
        err = 0.

        for b in xrange(self.num_buckets):
            b_idxs = [x['new_id'] for x in self.buckets_index[b]]

            data_bucket = self.data_2d[:,b_idxs]
            U_bucket    = self.U[b,:,:]
            V_bucket    = self.V[b,:,b_idxs].T

            error_b = sum(sum((data_bucket - np.dot(U_bucket, V_bucket)) ** 2))
            err += np.sqrt(error_b / len(self.buckets_index[b]))

        return err / self.num_buckets

    def objective_f(self):

        if self.obj_type == 'l2_reg':
            s = self.obj_func_l2_reg()

        elif self.obj_type == 'l1_reg':
            s = self.obj_func_l1_reg()

        else:
            raise ValueError("[GRADIENT_DESCENT]: Unknown objective_f_type [%s]."%self.obj_type)

        return s

    def obj_func_l2_reg(self):
        """
        The objective function to minimize. (l-2 regularization)
        """
        opt = 0.
        for b in xrange(self.num_buckets):
            # compute order of buckets
            bucket       = self.buckets_order[b]
            right_bucket = self.buckets_order[(b+1)%self.num_buckets]
            left_bucket  = self.buckets_order[b-1]

            # Add the error term only for the images in the current bucket
            b_idxs = [x['new_id'] for x in self.buckets_index[bucket]]
            data_bucket = self.data_2d[:,b_idxs]
            U_bucket    = self.U[bucket,:,:]
            V_bucket    = self.V[bucket,:,b_idxs].T
            error_b = sum(sum((data_bucket - np.dot(U_bucket, V_bucket)) ** 2))
            opt += error_b

            # Add L2 frobenious regularization
            V_fro   = sum(sum(self.V[bucket] ** 2))
            U_x_fro = sum(sum(self.U[bucket,self.x_portion_2d,:] ** 2))
            U_y_fro = sum(sum(self.U[bucket,self.y_portion_2d,:] ** 2))
            opt += self.lamb * (U_y_fro / self.num_buckets + U_x_fro + V_fro)

            # Add spatial regularization
            U_right = self.U[right_bucket,:,:]
            U_left  = self.U[left_bucket,:,:]
            opt    += self.lamb * self.kappa * ( \
                        sum(sum((U_bucket[self.x_portion_2d,:] - \
                                 U_right[self.x_portion_2d,:]) ** 2)) + \
                        sum(sum((U_bucket[self.x_portion_2d,:] - \
                                 U_left[self.x_portion_2d,:]) ** 2)) \
                      )

        return opt

    def grad_v_l2(self, j):
        """
        Find the gradient of V matrix for a single angle
        """
        b   = self.buckets_table[j]
        vj  = self.V[b,:,j]
        U_b = self.U[b,:,:]

        dv     = -2 * np.dot(self.data_2d[:,j] - np.dot(U_b,vj), U_b)
        l2_reg = 2 * vj

        return dv + self.lamb * l2_reg

    def grad_ux_l2(self, i):
        """
        Find the gradient of the U matrix for a single angle, and only the
        x coordinate
        """
        u_x_coord = i

        dux = np.zeros((self.num_buckets, self.num_factors))

        for b in xrange(self.num_buckets):
            cur_bucket   = self.buckets_order[b]
            right_bucket = self.buckets_order[(b+1)%self.num_buckets]
            left_bucket  = self.buckets_order[b-1]

            ui     = self.U[cur_bucket,u_x_coord,:]
            b_idxs = [x['new_id'] for x in self.buckets_index[cur_bucket]]

            du          = np.zeros(self.num_factors)
            l2_reg      = np.zeros(self.num_factors)
            spatial_reg = np.zeros(self.num_factors)

            # reconstruction error gradient
            V_b = self.V[cur_bucket,:,b_idxs]
            du     = -2 * np.dot(self.data_2d[u_x_coord,b_idxs] - np.dot(V_b,ui), V_b)

            # l2 norm regularization
            l2_reg += 2 * ui

            # Spatial regularization
            U_right = self.U[right_bucket,:,:]
            U_left  = self.U[left_bucket,:,:]
            spatial_reg += self.kappa * 2 * (ui - U_right[u_x_coord,:]) * 1
            spatial_reg += self.kappa * 2 * (ui - U_left[u_x_coord,:]) * 1

            dux[cur_bucket,:] = du + self.lamb * (l2_reg + spatial_reg)

        return dux

    def grad_uy_l2(self, i):
        """
        Find the gradient of the U matrix for a single angle, and only the
        y coordinate
        """
        u_y_coord = i + self.num_joints

        duy = np.zeros((self.num_buckets, self.num_factors))

        for b in xrange(self.num_buckets):
            cur_bucket = self.buckets_order[b]

            ui     = self.U[cur_bucket,u_y_coord,:]
            b_idxs = [x['new_id'] for x in self.buckets_index[cur_bucket]]

            du     = np.zeros(self.num_factors)
            l2_reg = np.zeros(self.num_factors)

            # reconstruction error gradient
            V_b = self.V[cur_bucket,:,b_idxs]
            du  = -2 * np.dot(self.data_2d[u_y_coord,b_idxs] - np.dot(V_b,ui), V_b)

            # l2 norm regularization
            l2_reg += (1/float(self.num_buckets)) * 2 * ui

            duy[cur_bucket, :] = du + self.lamb * l2_reg

        return duy

    def obj_func_l1_reg(self):
        """
        The objective function to minimize. (l1-regularization)
        """
        opt = 0.
        for b in xrange(self.num_buckets):
            # compute order of buckets
            bucket       = self.buckets_order[b]
            right_bucket = self.buckets_order[(b+1)%self.num_buckets]
            left_bucket  = self.buckets_order[b-1]

            # Add the error term only for the images in the current bucket
            b_idxs = [x['new_id'] for x in self.buckets_index[bucket]]
            data_bucket = self.data_2d[:,b_idxs]
            U_bucket    = self.U[bucket,:,:]
            V_bucket    = self.V[bucket,:,b_idxs].T
            error_b = sum(sum((data_bucket - np.dot(U_bucket, V_bucket)) ** 2))
            opt += error_b

            # Add L1 absolute value regularization
            V_l1   = sum(sum(np.abs(self.V[bucket])))
            U_x_l1 = sum(sum(np.abs(self.U[bucket,self.x_portion_2d,:])))
            U_y_l1 = sum(sum(np.abs(self.U[bucket,self.y_portion_2d,:])))
            opt += self.lamb * (U_y_l1 / self.num_buckets + U_x_l1 + V_l1)

            # Add spatial regularization
            U_right = self.U[right_bucket,:,:]
            U_left  = self.U[left_bucket,:,:]
            opt    += self.lamb * self.kappa * ( \
                        sum(sum(np.abs(U_bucket[self.x_portion_2d,:] - \
                                 U_right[self.x_portion_2d,:]))) + \
                        sum(sum(np.abs(U_bucket[self.x_portion_2d,:] - \
                                 U_left[self.x_portion_2d,:]))) \
                      )

        return opt

    def grad_v_l1(self, i):
        """
        Find the gradient of V matrix for a single angle
        """
        b  = self.buckets_order[self.buckets_table[j]]
        ui = self.U[b,i,:]
        vj = self.V[b,:,j]

        opt = (self.data_2d[i,j] - np.dot(ui, vj)) * ui

        return 2 * self.lamb * vj - 2 * opt

    def grad_uy_l1(self, i, j):
        """
        Find the gradient of the U matrix for a single angle, and only the
        y coordinate
        """
        b  = self.buckets_order[self.buckets_table[j]]
        ui = self.U[b,i,:]
        vj = self.V[b,:,j]

        opt = (self.data[i,j] - np.dot(ui,vj)) * vj

        return -2 * opt

    def grad_ux_l1(self, i, j):
        """
        Find the gradient of the U matrix for a single angle, and only the
        x coordinate
        """
        b  = self.buckets_order[self.buckets_table[j]]
        ui = self.U[b,i,:]
        vj = self.V[b,:,j]

        opt1 = (self.data_2d[i, j] - np.dot(ui, vj)) * vj

        # Spatial regularization
        right_adj = (a+1) % self.num_buckets
        left_adj  = a-1
        U1 = self.U[right_adj,i,:]
        U2 = self.U[left_adj,i,:]

        opt2 = self.kappa * ((U1 - ui_a) + (U2 - ui_a))

        return -2 * opt1 + 2 * self.lamb * opt2

    @staticmethod
    def plot_error(errors, filename, title, xlabel, ylabel):
        """
        Given a list of errors, plot the objectives of the training and show
        """
        plt.close('all')
        iterations = range(len(errors)+1)
        errors.insert(0, 0)
        plt.semilogx(iterations, errors, 'x-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.close('all')
