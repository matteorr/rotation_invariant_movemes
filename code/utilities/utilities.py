import matplotlib

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import sys, os
import numpy as np
import time

"""
Class of utilities : functions for plotting
"""
class utilities:
    def __init__(self, train_model_obj, colors, basis_coeff):
        print "[UTILITIES]: Initializing..."
        tic = time.time()

        self.num_factors = train_model_obj.num_factors
        self.num_buckets = train_model_obj.num_buckets

        self.x_portion_2d = train_model_obj.dataset_obj.x_portion_2d
        self.y_portion_2d = train_model_obj.dataset_obj.y_portion_2d

        self.x_portion_3d = train_model_obj.dataset_obj.x_portion_3d
        self.y_portion_3d = train_model_obj.dataset_obj.y_portion_3d
        self.z_portion_3d = train_model_obj.dataset_obj.z_portion_3d

        self.skeleton = train_model_obj.dataset_obj.skeleton
        self.HEAD_JOINT = train_model_obj.dataset_obj.HEAD_JOINT
        self.colors = colors
        self.basis_coeff = basis_coeff

        self.U = train_model_obj.U
        self.V = train_model_obj.V
        self.model_type = train_model_obj.model_type

        self.mean_pose = train_model_obj.utilities_data

        # for each bone extract mean length over all dataset
        self.mean_bone_lengths_2d = {}
        x_2d = train_model_obj.dataset_obj.data_2d[self.x_portion_2d,:]
        y_2d = train_model_obj.dataset_obj.data_2d[self.y_portion_2d,:]
        for bone in self.skeleton:
            self.mean_bone_lengths_2d[bone[0],bone[1]] = np.mean(np.sqrt(\
                    (x_2d[bone[0],:]-x_2d[bone[1],:])**2 + \
                    (y_2d[bone[0],:]-y_2d[bone[1],:])**2 ))

        if '3d' in self.model_type:
            self.mean_bone_lengths_3d = {}
            x_3d = train_model_obj.dataset_obj.data_3d[self.x_portion_3d,:]
            y_3d = train_model_obj.dataset_obj.data_3d[self.y_portion_3d,:]
            z_3d = train_model_obj.dataset_obj.data_3d[self.z_portion_3d,:]
            for bone in self.skeleton:
                self.mean_bone_lengths_3d[bone[0],bone[1]] = np.mean(np.sqrt(\
                    (x_3d[bone[0],:]-x_3d[bone[1],:])**2 + \
                    (y_3d[bone[0],:]-y_3d[bone[1],:])**2 + \
                    (z_3d[bone[0],:]-z_3d[bone[1],:])**2 ))

        self.root_save_dir = train_model_obj.save_plots_path

        print "[UTILITIES]: Done, (t=%.2fs)."%(time.time() - tic)

    def plot_movemes(self):
        """
        Plot movemes for the trained model.
        """
        if self.model_type in ['svd', 'lfa_3d']:
            self.draw_global_basis()
        elif self.model_type in ['lfa_2d', 'bucketed_svd_2d', 'bucketed_svd_3d']:
            self.draw_bucketed_basis()

    def draw_bucket_centroids(self):
        """
        Plot centroid poses for each clusters : should not be mean-centered.
        """
        if self.model_type not in ['bucketed_svd_3d', 'bucketed_svd_2d', 'lfa_2d']:
            raise ValueError('%s does not use buckets.'%self.model_type)

        save_dir = '%s/bucket_centroids'%self.root_save_dir
        if not os.path.exists(save_dir):
            os.system('mkdir %s'%(save_dir))

        for b in xrange(self.num_buckets):
            if self.model_type == 'bucketed_svd_3d':
                fig = plt.figure()
                ax = p3.Axes3D(fig)

                x_joints = self.mean_pose[b][self.x_portion_3d]
                y_joints = self.mean_pose[b][self.y_portion_3d]
                z_joints = self.mean_pose[b][self.z_portion_3d]

                ax = self.draw_3d_skeleton(ax, x_joints, y_joints, z_joints, self.skeleton, self.colors)

                plt.suptitle('Centroid Pose of Cluster %d'%(b))
                plt.savefig('%s/centroid_%d.png'%(save_dir, b))
                plt.close('all')

            elif self.model_type == 'bucketed_svd_2d' or 'lfa_2d':
                fig = plt.figure()
                ax = fig.gca()
                x_joints = self.mean_pose[b][self.x_portion_2d]
                y_joints = self.mean_pose[b][self.y_portion_2d]

                ax = self.draw_2d_skeleton(ax, x_joints, y_joints, self.skeleton, self.colors)

                plt.suptitle('Centroid Pose of Cluster %d'%(b))
                plt.savefig('%s/centroid_%d.png'%(save_dir, b))
                plt.close('all')

            print "[UTILITIES]: Saved centroid skeleton %d / %d"%(b+1, self.num_buckets)

    def draw_bucketed_basis(self):
        """
        Plotting function for models using buckets
        """
        U = self.U
        V = self.V
        mean = self.mean_pose
        basis_coeff = self.basis_coeff
        save_dir = '%s/%s_basis'%(self.root_save_dir, self.model_type)
        if not os.path.exists(save_dir):
            os.system('mkdir %s'%(save_dir))
            os.system('mkdir %s/img'%(save_dir))

        ##### LFA2D OR BUCKETED_SVD_2D
        if self.model_type in ['lfa_2d', 'bucketed_svd_2d']:
            for a in xrange(self.num_buckets):
                for k in xrange(self.num_factors):
                    # Use the maximum and minimum of the learned coefficient matrix
                    # as the basis coefficeint range for visualization
                    if self.model_type == 'lfa_2d':
                        min_basis_coeff = np.min(V[a,k,:])
                        max_basis_coeff = np.max(V[a,k,:])
                        basis_coeff_list = np.linspace(min_basis_coeff, max_basis_coeff, 50)
                    else:
                        basis_coeff_list = np.linspace(-basis_coeff, basis_coeff, 50)
                    idx = 0
                    pix_margin = 25

                    # Set the frame size for video
                    mm_frame1 = mean[a] + basis_coeff_list[0] * U[a,:,k]
                    mm_frame2 = mean[a] + basis_coeff_list[-1] * U[a,:,k]

                    max_x = max(np.hstack((np.abs(mm_frame1[self.x_portion_2d]),
                                           np.abs(mm_frame2[self.x_portion_2d]))))
                    max_y = max(np.hstack((np.abs(mm_frame1[self.y_portion_2d]),
                                           np.abs(mm_frame2[self.y_portion_2d]))))

                    xlim = [ -max_x - pix_margin,  max_x + pix_margin ]
                    ylim = [ -max_y - pix_margin,  max_y + pix_margin ]

                    for b in basis_coeff_list:
                        idx += 1
                        fig = plt.figure()
                        ax = fig.gca()

                        mua = mean[a] + b * U[a,:,k]
                        x_joints = mua[self.x_portion_2d]
                        y_joints = mua[self.y_portion_2d]
                        x_joints, y_joints = self.normalize2D(x_joints, y_joints, self.skeleton, self.HEAD_JOINT, self.mean_bone_lengths_2d)

                        ax = self.draw_2d_skeleton(ax, x_joints, y_joints, self.skeleton, self.colors, xlim=xlim, ylim=ylim)

                        # ax.set_xlim(x_lim[0], x_lim[1])
                        # ax.set_ylim(y_lim[0], y_lim[1])

                        plt.suptitle('Pose %d of Bucket %d'%(k,a))
                        plt.savefig('%s/img/pose%d.png'%(save_dir, idx))
                        # plt.savefig('%s/moveme_%d_bucket_%d.png'%(save_dir, k, a))
                        plt.close('all')

                    command = 'ffmpeg -framerate 20 -i '
                    command = command + save_dir + '/img/pose%d.png'
                    #command = command + ' -vcodec mpeg4 -y '
                    command = command + ' -vcodec libx264 -y -pix_fmt yuv420p '
                    command = '%s%s/moveme_%d_bucket_%d.mp4'%(command, save_dir, k, a)
                    os.system(command)

        ##### BUCKETED_SVD_3D
        elif self.model_type == 'bucketed_svd_3d':
            for a in xrange(self.num_buckets):
                for k in xrange(self.num_factors):
                    # Use the maximum and minimum of the learned coefficient matrix
                    # as the basis coefficeint range for visualization
                    basis_coeff_list = np.linspace(-basis_coeff, basis_coeff, 50)
                    idx = 0
                    pix_margin = 25

                    # Set the frame size for video
                    mm_frame1 = mean[a] + basis_coeff_list[0] * U[a,:,k]
                    mm_frame2 = mean[a] + basis_coeff_list[-1] * U[a,:,k]

                    max_x = max(np.hstack((np.abs(mm_frame1[self.x_portion_3d]),
                                           np.abs(mm_frame2[self.x_portion_3d]))))
                    max_y = max(np.hstack((np.abs(mm_frame1[self.y_portion_3d]),
                                           np.abs(mm_frame2[self.y_portion_3d]))))
                    max_z = max(np.hstack((np.abs(mm_frame1[self.z_portion_3d]),
                                           np.abs(mm_frame2[self.z_portion_3d]))))

                    xlim = [ -max_x - pix_margin,  max_x + pix_margin ]
                    ylim = [ -max_y - pix_margin,  max_y + pix_margin ]
                    zlim = [ -max_z - pix_margin,  max_z + pix_margin ]

                    for b in basis_coeff_list:
                        idx += 1
                        fig = plt.figure()
                        ax = p3.Axes3D(fig)

                        mua = mean[a] + b * U[:,k]
                        x_joints = mua[self.x_portion_3d]
                        y_joints = mua[self.y_portion_3d]
                        z_joints = -mua[self.z_portion_3d]
                        x_joints, y_joints, z_joints = self.normalize3D(x_joints, y_joints, z_joints, self.skeleton, self.HEAD_JOINT, self.mean_bone_lengths_3d)

                        ax = self.draw_3d_skeleton(ax, x_joints, y_joints, z_joints, self.skeleton, self.colors, xlim=xlim, ylim=ylim, zlim=zlim)

                        #ax.invert_xaxis()
                        #ax.invert_yaxis()
                        #ax.invert_zaxis()
                        ax.view_init(elev=-85, azim=-90)

                        ax.set_xlim(x_lim[0], x_lim[1])
                        ax.set_ylim(y_lim[0], y_lim[1])
                        ax.set_zlim(z_lim[0], z_lim[1])

                        plt.suptitle('Pose %d of Bucket %d'%(k,a))
                        plt.savefig('%s/img/pose%d.png'%(save_dir, idx))
                        # plt.savefig('%s/moveme_%d_bucket_%d.png'%(save_dir, k, a))
                        plt.close('all')

                    command = 'ffmpeg -framerate 20 -i '
                    command = command + save_dir + '/img/pose%d.png'
                    #command = command + ' -vcodec mpeg4 -y '
                    command = command + ' -vcodec libx264 -y -pix_fmt yuv420p '
                    command = '%s%s/moveme%d_bucket_%d.mp4'%(command, save_dir, k, a)
                    os.system(command)


    def draw_global_basis(self):
        """
        Plotting function for models NOT using buckets
        """
        U = self.U
        V = self.V
        mean = np.squeeze(self.mean_pose)
        basis_coeff = self.basis_coeff
        #basis_coeff_list = np.linspace(0, np.max(np.abs(V)), 100)
        save_dir = '%s/videos'%self.root_save_dir
        if not os.path.exists(save_dir):
            os.system('mkdir %s'%(save_dir))
            os.system('mkdir %s/img'%(save_dir))

        # Make a video with the 3dmean pose moving according to the learned
        # movements contained in the U matrix

        ##### SVD
        if self.model_type == 'svd':
            for k in xrange(self.num_factors):
                #basis_coeff_list = np.linspace(-basis_coeff, basis_coeff, 50)
                basis_coeff_list = np.linspace(0, 1, 50)

                idx = 0

                mm_frame1 = mean + basis_coeff_list[0] * U[:,k]
                mm_frame2 = mean + basis_coeff_list[-1] * U[:,k]

                max_x = max(np.hstack((np.abs(mm_frame1[self.x_portion_2d]),
                                       np.abs(mm_frame2[self.x_portion_2d]))))
                max_y = max(np.hstack((np.abs(mm_frame1[self.y_portion_2d]),
                                       np.abs(mm_frame2[self.y_portion_2d]))))

                #head_y = max(mm_frame1[self.y_portion_2d][self.HEAD_JOINT], mm_frame2[self.y_portion_2d][self.HEAD_JOINT])

                xlim = [ -max_x - 5,  max_x + 5 ]
                ylim = [ -max_y - 5,  max_y + 5 ]

                for b in basis_coeff_list:
                    idx += 1
                    # Attaching 3D axis to the figure
                    fig = plt.figure()
                    ax = fig.gca()

                    mua = mean + b * U[:,k]
                    x_joints = mua[self.x_portion_2d]
                    y_joints = mua[self.y_portion_2d]
                    #x_joints, y_joints = self.normalize2D(x_joints, y_joints, self.skeleton, self.HEAD_JOINT, self.mean_bone_lengths_2d)

                    ax = self.draw_2d_skeleton(ax, x_joints, y_joints, self.skeleton, self.colors, xlim=xlim, ylim=ylim)

                    # ax.set_xlim(xlim[0], xlim[1])
                    # ax.set_ylim(ylim[0], ylim[1])
                    plt.suptitle('moveme %d'%(k))
                    plt.savefig('%s/img/moveme%d.png'%(save_dir, idx))
                    #plt.savefig('%s/moveme_%d.png'%(save_dir, k))

                    plt.close('all')

                command = 'ffmpeg -framerate 20 -i '
                command = command + save_dir + '/img/moveme%d.png'
                #command = command + ' -vcodec mpeg4 -y '
                command = command + ' -vcodec libx264 -y -pix_fmt yuv420p '
                command = '%s%s/moveme_%d.mp4'%(command, save_dir, k)
                os.system(command)

        ##### LFA3D
        elif self.model_type == 'lfa_3d':
            for k in xrange(self.num_factors):
                idx = 0
                max_val = np.max(np.abs(V[k,:]))
                basis_coeff_list = np.linspace(0, max_val, 100*np.ceil(max_val))
                # Set the frame size for video
                mm_frame1 = mean + basis_coeff_list[0] * U[:,k]
                mm_frame2 = mean + basis_coeff_list[-1] * U[:,k]

                max_x = max(np.hstack((np.abs(mm_frame1[self.x_portion_3d]),
                                       np.abs(mm_frame2[self.x_portion_3d]))))
                max_y = max(np.hstack((np.abs(mm_frame1[self.y_portion_3d]),
                                       np.abs(mm_frame2[self.y_portion_3d]))))
                max_z = max(np.hstack((np.abs(mm_frame1[self.z_portion_3d]),
                                       np.abs(mm_frame2[self.z_portion_3d]))))

                xlim = [ -max_x - 5,  max_x + 5 ]
                ylim = [ -max_y - 5,  max_y + 5 ]
                zlim = [ -max_z - 5,  max_z + 5 ]

                for b in basis_coeff_list:
                    idx += 1
                    # Attaching 3D axis to the figure
                    fig = plt.figure()
                    ax = p3.Axes3D(fig)

                    mua = mean + b * U[:,k]
                    x_joints = mua[self.x_portion_3d]
                    y_joints = mua[self.y_portion_3d]
                    z_joints = -mua[self.z_portion_3d]
                    x_joints, y_joints, z_joints = self.normalize3D(x_joints, y_joints, z_joints, self.skeleton, self.HEAD_JOINT, self.mean_bone_lengths_3d)

                    ax = self.draw_3d_skeleton(ax, x_joints, y_joints, z_joints, self.skeleton, self.colors, xlim=xlim, ylim=ylim, zlim=zlim)

                    #ax.invert_xaxis()
                    #ax.invert_yaxis()
                    #ax.invert_zaxis()
                    ax.view_init(elev=-85, azim=-90)

                    ax.set_xlim(xlim[0], xlim[1])
                    ax.set_ylim(ylim[0], ylim[1])
                    ax.set_zlim(zlim[0], zlim[1])
                    plt.suptitle('moveme %d'%(k))
                    plt.savefig('%s/img/moveme%d.png'%(save_dir, idx))
                    #plt.savefig('%s/moveme_%d.png'%(save_dir, k))

                    plt.close('all')

                command = 'ffmpeg -framerate 20 -i '
                command = command + save_dir + '/img/moveme%d.png'
                #command = command + ' -vcodec mpeg4 -y '
                command = command + ' -vcodec libx264 -y -pix_fmt yuv420p '
                command = '%s%s/moveme_%d.mp4'%(command, save_dir, k)
                os.system(command)

    @staticmethod
    def draw_2d_skeleton(ax, x_joints, y_joints, skeleton, colors,
                         xlim=[], ylim=[]):
        if xlim == []:
            xlim = [-40, 40]
        if ylim == []:
            ylim = [-120, 20]

        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax.invert_yaxis()
        ax.scatter(x_joints, y_joints)
        for b in skeleton:
            ja = b[0]
            jb = b[1]
            x_coord = [x_joints[ja],x_joints[jb]]
            y_coord = [y_joints[ja],y_joints[jb]]
            color_string = '-%s'%(colors[[k for k in skeleton if k==b][0][1]])
            ax.plot(x_coord,y_coord,color_string,linewidth=2)

        return ax

    @staticmethod
    def draw_3d_skeleton(ax, x_joints, y_joints, z_joints, skeleton, colors,
                        limit_axis=False, xlim=[-40,40], ylim=[-120,20], zlim=[]):

        if xlim == []:
            xlim = [-40, 40]
        if ylim == []:
            ylim = [-120, 20]

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_zlim(zlim[0], zlim[1])

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.zaxis.set_visible(False)

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.zaxis.set_ticks([])

        ax.scatter(x_joints, y_joints, z_joints)
        for b in skeleton:
            ja = b[0]
            jb = b[1]
            x_coord = [x_joints[ja],x_joints[jb]]
            y_coord = [y_joints[ja],y_joints[jb]]
            z_coord = [z_joints[ja],z_joints[jb]]
            color_string = '-%s'%(colors[[k for k in skeleton if k==b][0][1]])
            ax.plot(x_coord,y_coord,z_coord,color_string,linewidth=2)
        ax.view_init(elev=-90, azim=-90)

        return ax

    @staticmethod
    def normalize2D(x, y, skeleton, HEAD_JOINT, mean_bone_lengths):
        """
        Normalize the length of each joint with the average length
        for that segment over all training set.
        Starting from the head down to the rest of the body
        """
        ## NOTE: The code assumes that for every skeleton pair
        #        the smaller index is the parent of the larger index.
        #        I.e.: [2,8] -> [left shoulder, left hip]

        x_norm = np.zeros(x.shape)
        y_norm = np.zeros(y.shape)

        # assume that the location of the head joint does not change
        x_norm[HEAD_JOINT] = x[HEAD_JOINT]
        y_norm[HEAD_JOINT] = y[HEAD_JOINT]

        for bone in skeleton:
            # get parent and child coordinates
            j_p = bone[0]
            j_c = bone[1]

            p_coord = np.array([x[j_p],y[j_p]])
            c_coord = np.array([x[j_c],y[j_c]])

            # get difference vector
            bone_vec    = c_coord - p_coord
            bone_length = np.sqrt( bone_vec[0]**2 + bone_vec[1]**2)

            if bone_length == 0:
                assert(False)

            # add normalized difference vector to parent coordinates
            norm_bone_vec   = bone_vec * (mean_bone_lengths[bone[0],bone[1]] / bone_length)
            norm_p_coord    = np.array([x_norm[j_p],y_norm[j_p]])
            norm_bone_coord = norm_bone_vec + norm_p_coord

            x_norm[j_c] = norm_bone_coord[0]
            y_norm[j_c] = norm_bone_coord[1]

        return x_norm, y_norm

    @staticmethod
    def normalize3D(x, y, z, skeleton, HEAD_JOINT, mean_bone_lengths):
        """
        Normalize the length of each joint with the average length
        for that segment over all training set.
        """
        ## NOTE: The code assumes that for every skeleton pair
        #        the smaller index is the parent of the larger index.
        #        I.e.: [2,8] -> [left shoulder, left hip]

        x_norm = np.zeros(x.shape)
        y_norm = np.zeros(y.shape)
        z_norm = np.zeros(z.shape)

        x_norm[HEAD_JOINT] = x[HEAD_JOINT]
        y_norm[HEAD_JOINT] = y[HEAD_JOINT]
        z_norm[HEAD_JOINT] = z[HEAD_JOINT]

        for bone in skeleton:
            j_p = bone[0]
            j_c = bone[1]

            p_coord = np.array([x[j_p],y[j_p],z[j_p]])
            c_coord = np.array([x[j_c],y[j_c],z[j_c]])

            # get difference vector
            bone_vec    = c_coord - p_coord
            bone_length = np.sqrt( bone_vec[0]**2 + bone_vec[1]**2 + bone_vec[2]**2 )

            if bone_length == 0:
                assert(False)

            # add normalized difference vector to parent coordinates
            norm_bone_vec   = bone_vec * (mean_bone_lengths[bone[0],bone[1]] / bone_length)
            norm_p_coord    = np.array([x_norm[j_p],y_norm[j_p],z_norm[j_p]])
            norm_bone_coord = norm_bone_vec + norm_p_coord

            x_norm[j_c] = norm_bone_coord[0]
            y_norm[j_c] = norm_bone_coord[1]
            z_norm[j_c] = norm_bone_coord[2]

        return x_norm, y_norm, z_norm
