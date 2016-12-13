import numpy as np, math
import datetime, time, sys
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import svd as npsvd
import copy

"""
Stochastic Gradient Descent for 3D Latent Factor Model
"""

class gradient_descent_3d:
    def __init__(self,
                 data_2d,
                 data_3d_c_mean,
                 data_3d_c_movemes,
                 angle_anns,
                 num_images,
                 num_dim_2d,
                 num_dim_3d,
                 x_portion_2d,
                 y_portion_2d,
                 x_portion_3d,
                 y_portion_3d,
                 z_portion_3d,
                 num_factors,
                 num_joints,
                 objective_f_type,
                 hyper_params,
                 save_partial_models,
                 save_dir):
        print "[GRADIENT_DESCENT]: Initializing..."
        tic = time.time()

        self.data_2d = data_2d

        self.data_3d_c_movemes = data_3d_c_movemes
        self.data_3d_c_mean    = data_3d_c_mean

        tmp = sorted(angle_anns, key=lambda k: k['new_id'])
        self.angles = [a['angle_rad'] for a in tmp]

        self.x_portion_2d = x_portion_2d
        self.y_portion_2d = y_portion_2d
        self.x_portion_3d = x_portion_3d
        self.y_portion_3d = y_portion_3d
        self.z_portion_3d = z_portion_3d

        self.num_dim_2d  = num_dim_2d
        self.num_dim_3d  = num_dim_3d

        self.num_factors = num_factors
        self.num_images  = num_images
        self.num_joints  = num_joints

        self.obj_type        = objective_f_type
        self.positive_V_flag = hyper_params['positive_V_flag']

        self.l_rate_U     = hyper_params['l_rate_U']
        self.m_rate_U     = hyper_params['m_rate_U']
        self.l_rate_V     = hyper_params['l_rate_V']
        self.m_rate_V     = hyper_params['m_rate_V']
        self.l_rate_theta = hyper_params['l_rate_theta']
        self.m_rate_theta = hyper_params['m_rate_theta']

        self.lr_bound     = hyper_params['lr_bound']
        self.max_iter     = hyper_params['max_iter']

        self.lambda_l1    = 0.002
        self.lambda_l2    = 0.00002
        self.num_epochs   = 3
        self.error_window = hyper_params['error_window']

        self.rmse_tolerance     = hyper_params['rmse_tolerance']
        self.lr_decay           = hyper_params['lr_decay']
        self.obj_func_tolerance = hyper_params['obj_func_tolerance']

        self.UV_batch_step    = hyper_params['UV_batch_step']
        self.theta_batch_step = hyper_params['theta_batch_step']

        self.save_partial_models = save_partial_models

        self.save_dir = save_dir

        print "[GRADIENT_DESCENT]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def initialize(self, init_type_3d, U_test=None):

        if init_type_3d == 'svd':
            # perform regular SVD on the 3D data demeaned to capture movemes
            # use resulting U and V matrices as initialization for the gradient descent
            init_U, s, init_V = npsvd(self.data_3d_c_movemes,full_matrices=False)

            init_U = np.array(init_U[:,:self.num_factors])
            init_V = np.array(init_V[:self.num_factors,:])

            if self.positive_V_flag:
                raise ValueError("Initialization with SVD cannot enforce positive V.")

        elif init_type_3d == 'random':
            # perform random initialization for U and V matrices between 0 and 1
            init_U = np.random.random((self.num_dim_3d, self.num_factors))
            init_V = np.random.random((self.num_factors, self.num_images))
            if self.positive_V_flag:
                assert(np.all(init_V>=0))

        elif init_type_3d == 'test':
            # initialize U as the passed value
            init_U = U_test
            # perform random initialization for V between 0 and 1
            init_V = np.random.random((self.num_factors, self.num_images))
            if self.positive_V_flag:
                assert(np.all(init_V>=0))

        else:
            raise ValueError('Unknown initialization type [%s]!'%init_type_3d)

        self.U = init_U
        self.V = init_V

    def resume(self, U_file_name, V_file_name):
        raise NotImplementedError

    def solve(self):
        """
        Alternating gradient descent U & V , theta
        """
        print "[GRADIENT_DESCENT]: Solving with objective function [%s]."%(self.obj_type)
        start_time = time.time()
        np.seterr(all='raise')

        errors = None
        objs   = None
        for e in xrange(self.num_epochs):
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            print "+++++ EPOCH: [%d] +++++++++++++++++++++++++++++++++++++"%(e+1)
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"

            errors, objs = self.gd_UV(self.UV_batch_step, errors, objs)
            errors, objs = self.gd_theta(self.theta_batch_step, errors, objs)

        end_time = time.time() - start_time
        print "#" * 50
        print "[GRADIENT_DESCENT]: SUMMARY"
        print "  * Optimization Completed in t=[%.2f] secs"%(end_time)
        print "  * Final Obj           : %f"%(objs[-1])
        print "  * Final Obj Change    : %f"%(objs[-1] - objs[0])
        print "  * Final RMSE          : %f"%(errors[-1])
        print "  * Final RMSE Change   : %f"%(errors[-1] - errors[0])
        print "-" * 50
        print "  * Learning Rate U     : %f"%(self.l_rate_U)
        print "  * Learning Rate V     : %f"%(self.l_rate_V)
        print "  * Learning Rate theta : %f"%(self.l_rate_theta)
        print "#" * 50

        time_str = datetime.datetime.fromtimestamp(
                    time.time()).strftime('%Y_%m_%d_%H_%M_%S')

        U_file_path = '%s/models/U_final_%s_%s_%s_%s_%s_%s.pkl'%\
                       (self.save_dir, self.num_dim_2d, self.num_images,
                        self.num_factors, e+1, self.obj_type, time_str)
        pickle.dump(self.U, open(U_file_path, 'wb'))

        V_file_path = '%s/models/V_final_%s_%s_%s_%s_%s_%s.pkl'%\
                       (self.save_dir, self.num_dim_2d, self.num_images,
                        self.num_factors, e+1, self.obj_type, time_str)
        pickle.dump(self.V, open(V_file_path, 'wb'))

        angles_file_path = '%s/models/angles_final_%s_%s_%s_%s_%s_%s.pkl'%\
                       (self.save_dir, self.num_dim_2d, self.num_images,
                        self.num_factors, e+1, self.obj_type, time_str)
        pickle.dump(self.angles, open(angles_file_path, 'wb'))

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

        return self.U, self.V, self.angles

    def gd_UV(self, batch_step, errors=None, objs=None):
        """
        Gradient descent w.r.t U and V
        """
        print "[GRADIENT_DESCENT]: Gradient step for U and V..."
        start_time = time.time()

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
        if errors is None:
            errors = [rmse_prev]
        else:
            errors = errors

        if objs is None:
            objs = [obj_prev]
        else:
            objs = objs

        # Offset the error to count how many iterations are happening this time
        err_offset = len(objs)-1

        # Previous weight update info for momentum
        dux_prev = np.zeros((self.num_factors))
        duy_prev = np.zeros((self.num_factors))
        duz_prev = np.zeros((self.num_factors))
        dv_prev = np.zeros((self.num_factors))

        # Loop and repeat updating U and V matrices
        it_time = time.time()
        for it in xrange(self.max_iter):
            # Randomly select one datapoint
            _i = np.random.randint(self.num_joints)
            _j = np.random.randint(self.num_images)

            ####################################################################
            # Compute the local gradient of V at selected datapoint
            ####################################################################
            # NOTE: grad_V is ALWAYS differentiable
            dv = self.grad_V(_j)

            if self.obj_type == 'l2_reg':
                # Update V gradients
                self.V[:,_j] -= (self.l_rate_V * dv + self.m_rate_V * dv_prev)

            elif self.obj_type in ['l1_reg','l2_l1_ista_reg']:
                # first update V with the regular l2 plus reconstruction
                # error gradient, then threshold for l1 regularization
                # details from iterative soft thresholding algorithm
                v_tmp = self.V[:,_j] - (self.l_rate_V * dv + self.m_rate_V * dv_prev)

                for l in xrange(len(v_tmp)):
                    if v_tmp[l] <= - self.l_rate_V * self.lambda_l1:
                        self.V[l,_j] = v_tmp[l] + self.l_rate_V * self.lambda_l1

                    elif v_tmp[l] >= self.l_rate_V * self.lambda_l1:
                        self.V[l,_j] = v_tmp[l] - self.l_rate_V * self.lambda_l1

                    else:
                        self.V[l,_j] = 0

            else:
                raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

            # clamp v coefficient to zero if the update makes it negative
            if self.positive_V_flag:
                for l in xrange(len(self.V[:,_j])):
                    if self.V[l,_j] < 0:
                        self.V[l,_j] = 0
                    if self.V[l,_j] > 1:
                        self.V[l,_j] = 1
            ####################################################################
            # Compute the local gradient of U at selected joint
            ####################################################################

            u_x_coord = _i
            u_y_coord = _i + self.num_joints
            u_z_coord = _i + 2 * self.num_joints

            # NOTE: grad_U is ALWAYS differentiable
            dux = self.grad_U(u_x_coord)
            duy = self.grad_U(u_y_coord)
            duz = self.grad_U(u_z_coord)

            if self.obj_type == 'l2_reg':
                # Update U gradients portions
                self.U[u_x_coord,:] -= self.l_rate_U * dux + self.m_rate_U * dux_prev
                self.U[u_y_coord,:] -= self.l_rate_U * duy + self.m_rate_U * duy_prev
                self.U[u_z_coord,:] -= self.l_rate_U * duz + self.m_rate_U * duz_prev

            elif self.obj_type in ['l1_reg','l2_l1_ista_reg']:
                # first update U with the regular l2 plus reconstruction
                # error gradient, then threshold for l1 regularization
                # details from iterative soft thresholding algorithm

                u_x_tmp = self.U[u_x_coord,:] - (self.l_rate_U * dux + self.m_rate_U * dux_prev)
                u_y_tmp = self.U[u_y_coord,:] - (self.l_rate_U * duy + self.m_rate_U * duy_prev)
                u_z_tmp = self.U[u_z_coord,:] - (self.l_rate_U * duz + self.m_rate_U * duz_prev)

                for l in xrange(len(u_x_tmp)):
                    # Update U gradients x portions
                    if u_x_tmp[l] <= - self.l_rate_U * self.lambda_l1:
                        self.U[u_x_coord,l] = u_x_tmp[l] + self.l_rate_U * self.lambda_l1

                    elif u_x_tmp[l] >= self.l_rate_U * self.lambda_l1:
                        self.U[u_x_coord,l] = u_x_tmp[l] - self.l_rate_U * self.lambda_l1

                    else:
                        self.U[u_x_coord,l] = 0

                    # Update U gradients y portions
                    if u_y_tmp[l] <= - self.l_rate_U * self.lambda_l1:
                        self.U[u_y_coord,l] = u_y_tmp[l] + self.l_rate_U * self.lambda_l1

                    elif u_x_tmp[l] >= self.l_rate_U * self.lambda_l1:
                        self.U[u_y_coord,l] = u_y_tmp[l] - self.l_rate_U * self.lambda_l1

                    else:
                        self.U[u_y_coord,l] = 0

                    # Update U gradients z portions
                    if u_z_tmp[l] <= - self.l_rate_U * self.lambda_l1:
                        self.U[u_z_coord,l] = u_z_tmp[l] + self.l_rate_U * self.lambda_l1

                    elif u_z_tmp[l] >= self.l_rate_U * self.lambda_l1:
                        self.U[u_z_coord,l] = u_z_tmp[l] - self.l_rate_U * self.lambda_l1

                    else:
                        self.U[u_z_coord,l] = 0

            else:
                raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

            dv_prev  = dv
            dux_prev = dux
            duy_prev = duy
            duz_prev = duz

            if (it+1) % batch_step == 0:
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
                if len(objs) - err_offset >= 2 * self.error_window:
                    curr_batch = np.mean(objs[-self.error_window:])
                    prev_batch = np.mean(objs[-2*self.error_window:-self.error_window])
                    print '-' * 30
                    print " Average OBJ Delta check: "
                    print "  Prev Batch : %f"%(prev_batch)
                    print "  Curr Batch : %f"%(curr_batch)
                    if prev_batch - curr_batch < self.obj_func_tolerance:
                        break

                rmse_prev = rmse_curr
                obj_prev  = obj_curr

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

                print "=" * 30
                it_time = time.time()

        print "#" * 50
        print "[GRADIENT_DESCENT]: Done U and V step, (t=%.2f)."%(time.time() - start_time)

        return errors, objs

    def gd_theta(self, batch_step, errors=None, objs=None):
        """
        Gradient descent w.r.t theta
        """

        print "[GRADIENT_DESCENT]: Gradient step for Theta..."
        start_time = time.time()

        rmse_prev  = self.compute_rmse()
        obj_prev   = self.objective_f()
        init_error = rmse_prev
        init_obj   = obj_prev

        print "=" * 30
        print " * Initial RMSE: %f"%(rmse_prev)
        print " * Initial Obj:  %f"%(obj_prev)
        print '-' * 30
        print " * l_rate_theta: %f"%(self.l_rate_theta)
        print "=" * 30

        # Save the errors/objective for record (plotting)
        if errors is None:
            errors = [rmse_prev]
        else:
            errors = errors

        if objs is None:
            objs = [obj_prev]
        else:
            objs = objs

        # Offset the error to count how many iterations are happening this time
        err_offset = len(objs)-1

        # Previous weight update info for momentum
        dtheta_prev = 0.

        # Loop and repeat Theta values
        it_time = time.time()
        for it in xrange(self.max_iter):

            # Randomly select one datapoint.
            _j = np.random.randint(self.num_images)

            # Compute the local gradient of Theta at selected datapoint
            dtheta = self.grad_theta(_j)

            # Update Theta values based on gradient
            # NOTE: Theta is in radians, so there is no need to worry about the
            # value not being valid.
            self.angles[_j] -= self.l_rate_theta * dtheta + self.m_rate_theta * dtheta_prev

            # set current value as previous for next momentum
            dtheta_prev = dtheta

            # Compute errors/objectives in a batched intervals for faster iteration
            if (it+1) % batch_step == 0:
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
                print " l_rate_theta: %f"%(self.l_rate_theta)

                if (np.abs(delta) < self.rmse_tolerance) and self.l_rate_theta > self.lr_bound:
                    self.l_rate_theta *= self.lr_decay
                    print " ==> Learning rate Updated"

                # Observe if the relative change of the objective is less than certain
                # threshold for termination
                if len(objs) - err_offset >= 2 * self.error_window:
                    curr_batch = np.mean(objs[-self.error_window:])
                    prev_batch = np.mean(objs[-2*self.error_window:-self.error_window])
                    print '-' * 30
                    print " Average OBJ Delta check: "
                    print "  Prev Batch : %f"%(prev_batch)
                    print "  Curr Batch : %f"%(curr_batch)
                    if prev_batch - curr_batch < self.obj_func_tolerance:
                        break

                rmse_prev = rmse_curr
                obj_prev  = obj_curr

                if self.save_partial_models:
                    time_str = datetime.datetime.fromtimestamp(
                                time.time()).strftime('%Y_%m_%d_%H_%M_%S')
                    angles_file_path = '%s/models/angles_%s_%s_%s_%s_%s_%s.pkl'%\
                                   (self.save_dir, self.num_dim_2d, self.num_images,
                                    self.num_factors, it+1, self.obj_type, time_str)
                    pickle.dump(self.angles, open(angles_file_path, 'wb'))

                print "=" * 30
                it_time = time.time()

        print "#" * 50
        print "[GRADIENT_DESCENT]: Done Theta step, (t=%.2f)."%(time.time() - start_time)

        return errors, objs

    def compute_rmse(self):
        """
        Compute mean-squared error using the current U, V matrix.
        Return the average RMSE across all clusters.
        """
        projected_data = self.project_UV()
        err = np.sqrt(self.squared_err() / self.num_images)

        return err

    def objective_f(self):

        s = self.squared_err()

        if self.obj_type == 'l2_reg':
            s += self.l2_reg()

        elif self.obj_type == 'l1_reg':
            s += self.l1_reg()

        elif self.obj_type == 'l2_l1_ista_reg':
            s += self.l2_reg()
            s += self.l1_reg()

        else:
            raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

        return s

    def squared_err(self):
        """
        Reconstruction error
        """
        projected_data = self.project_UV()
        squared_err    = sum(sum((self.data_2d - projected_data) ** 2))

        return squared_err

    def l2_reg(self):
        """
        (l-2 regularization)
        """
        reg = sum(sum(self.U ** 2)) + sum(sum(self.V ** 2))
        return self.lambda_l2 * reg

    def l1_reg(self):
        """
        (l-1 regularization)
        """
        reg = sum(sum(abs(self.U))) + sum(sum(abs(self.V)))
        return self.lambda_l1 * reg

    def project_UV(self, _j=None):
        """
        This function gets current U and V and multiplies them to obtain
        approximation of the 3D data (which represents a movement).
        The rotates the movement applied to the mean pose to obtain the 3d pose
        Then projects it on a 2D plane given the current angle of view.
        obtaining a 2d pose to compare to the real 2d pose for that image.
        """
        # these are the approximated movements from the mean
        data_3d_approx_moves = np.dot(self.U,self.V)
        #data_3d_approx = np.dot(self.U,self.V)
        # apply moves to mean to obtain the 3d poses
        data_3d_approx = self.data_3d_c_mean + data_3d_approx_moves

        if _j is None:
            projected_data = np.zeros(self.data_2d.shape)

            cos_a = [np.cos(a) for a in self.angles]
            sin_a = [np.sin(a) for a in self.angles]

            # when looking at an angle of alpha the projected x will be
            # x * cos(alpha) + z * sin(alpha)
            projected_data[self.x_portion_2d,:] = \
                np.multiply(data_3d_approx[self.x_portion_3d,:],cos_a) + \
                np.multiply(data_3d_approx[self.z_portion_3d,:],sin_a)

            # when looking at an angle of alpha the projected y is the same
            projected_data[self.y_portion_2d,:] = \
                data_3d_approx[self.y_portion_3d,:]

        else:
            projected_data = np.zeros(self.num_dim_2d)

            # when looking at an angle of alpha the projected x will be
            # x * cos(alpha) + z * sin(alpha)
            projected_data[self.x_portion_2d] = \
                np.multiply(data_3d_approx[self.x_portion_3d,_j],np.cos(self.angles[_j])) + \
                np.multiply(data_3d_approx[self.z_portion_3d,_j],np.sin(self.angles[_j]))

            # when looking at an angle of alpha the projected y is the same
            projected_data[self.y_portion_2d] = \
                data_3d_approx[self.y_portion_3d,_j]

        return projected_data

    def grad_V(self, j):
        """
        Gradient w.r.t. V_j
        """
        vj = self.V[:,j]
        projected_data = self.project_UV(j)

        dv  = - 2 * np.dot(self.data_2d[:,j] - projected_data, self.dfdV(j))
        reg = 0.

        if self.obj_type in ['l2_reg','l2_l1_ista_reg']:
            reg = self.lambda_l2 * 2 * vj

        elif self.obj_type == 'l1_reg':
            # do nothing cuz the l1 part will be implemented later through ista
            pass

        else:
            raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

        return dv + reg

    def dfdV(self, j):
        """
        Compute df/dV, where f is the projection function
        """
        dfdV = np.zeros((self.num_dim_2d, self.num_factors))
        a    = self.angles[j]

        U_x = self.U[self.x_portion_3d,:]
        U_y = self.U[self.y_portion_3d,:]
        U_z = self.U[self.z_portion_3d,:]

        for i in xrange(self.num_joints):
            u_x_coord = i
            u_y_coord = i + self.num_joints

            dfdV[u_x_coord,:] = U_x[i,:] * np.cos(a) + U_z[i,:] * np.sin(a)
            dfdV[u_y_coord,:] = U_y[i,:]

        return dfdV

    def grad_U(self, i):
        """
        Gradient w.r.t. U_i
        """
        projected_data = self.project_UV()
        ui = self.U[i,:]

        du  = - 2 * np.sum(np.einsum('ijk,jk->ik', self.dfdU(i), self.data_2d - projected_data), axis=1)
        reg = 0.

        if self.obj_type in ['l2_reg','l2_l1_ista_reg']:
            reg = self.lambda_l2 * 2 * ui

        elif self.obj_type == 'l1_reg':
            # do nothing cuz the l1 part will be implemented later through ista
            pass

        else:
            raise ValueError("[GRADIENT_DESCENT]: Uknown obj_type [%s]!"%self.obj_type)

        return du + reg

    def dfdU(self, i):
        """
        Compute df/dU, where f is the projection function
        """
        dfdU = np.zeros((self.num_factors, self.num_dim_2d, self.num_images))

        cos_a = [np.cos(a) for a in self.angles]
        sin_a = [np.sin(a) for a in self.angles]

        if i in self.x_portion_3d:
            dfdU_coord_x = i
            dfdU[:,dfdU_coord_x,:] = np.multiply(self.V, cos_a)

        elif i in self.y_portion_3d:
            dfdU_coord_y = i
            dfdU[:,dfdU_coord_y,:] = self.V

        elif i in self.z_portion_3d:
            dfdU_coord_x = i - 2 * self.num_joints
            dfdU[:,dfdU_coord_x,:] = np.multiply(self.V, sin_a)

        else:
            raise ValueError("[GRADIENT_DESCENT_3D]: index for U_i not valid.")

        return dfdU

    def grad_theta(self, j):
        """
        Gradient w.r.t. theta
        """
        dtheta = - 2 * np.dot(self.data_2d[:,j] - self.project_UV(j), self.dfdTheta(j))

        return dtheta

    def dfdTheta(self, j):
        """
        Compute df/dTheta, where f is the projection function
        """
        dfdTheta = np.zeros(self.num_dim_2d)

        a  = self.angles[j]
        vj = self.V[:,j]

        U_x = self.U[self.x_portion_3d,:]
        U_y = self.U[self.y_portion_3d,:]
        U_z = self.U[self.z_portion_3d,:]

        for i in xrange(self.num_joints):
            u_x_coord = i
            u_y_coord = i + self.num_joints
            u_z_coord = i + 2 * self.num_joints

            dfdTheta[u_x_coord] = \
                (self.data_3d_c_mean[u_x_coord][0] + np.dot(U_x[i,:],vj)) * -1 * np.sin(a) + \
                (self.data_3d_c_mean[u_z_coord][0] + np.dot(U_z[i,:],vj)) * np.cos(a)
            dfdTheta[u_y_coord] = 0

        return dfdTheta

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
