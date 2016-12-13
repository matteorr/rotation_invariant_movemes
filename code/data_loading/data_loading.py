from CO_LSP.CO_LSP import CO_LSP
from pose_dataset.pose_dataset import pose_dataset
import time, sys
import scipy.io as scio
import numpy as np
import math
import json, pickle

"""
Loads the data required to train the model
"""
class data_loader:
    def __init__(self,
                 dataset_path,
                 images_path,
                 activity_list=[]):
        print "[DATA_LOADER]: Initializing..."
        tic = time.time()

        self.dataset_path       = dataset_path
        self.images_path        = images_path
        self.activity_list      = activity_list

        self.dataset = CO_LSP(self.dataset_path, self.images_path)

        print "[DATA_LOADER]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def load2D(self):
        print "[DATA_LOADER]: Loading 2D data..."
        print "[DATA_LOADER]: Loading activities [%s]..."\
                    %(','.join(self.activity_list))
        tic = time.time()

        ann_ids = self.dataset.get_ann_ids(activity_list=self.activity_list)
        anns    = sorted(self.dataset.get_anns(ann_ids),key=lambda k: k['id'])

        # used for mapping index from original to new data
        real2new = dict()
        # used for mapping index from new data to original
        new2real = dict()

        data_raw_2d = []
        angles_gt   = []
        for aind, a in enumerate(anns):
            # append keypoints in matrix
            data_raw_2d.append(a['2d_keypoints'][0::2] + a['2d_keypoints'][1::2])

            # save real_id and new_id correspondence
            real_id = a['id']
            new_id  = aind
            real2new[real_id] = new_id
            new2real[new_id]  = real_id

            ####################################################################
            ## NOTE: we are assuming that dataset has gt viewing angle available
            ####################################################################
            angle = (a['angle_avg'] + 270) % 360
            angle_dict = {'real_id':   real_id,
                          'new_id':    new_id,
                          'angle_deg': angle,
                          'angle_rad': math.radians(angle)}
            angles_gt.append(angle_dict)

        data_raw_2d = np.array(data_raw_2d).T

        print "[DATA_LOADER]: Training set has [%d] instances."%(data_raw_2d.shape[1])

        num_joints = self.dataset.get_num_keypoints()
        num_images = data_raw_2d.shape[1]

        ## procrustes analysis to align poses and normalize joint length
        HEAD_JOINT = self.dataset.get_keypoints().index('head')
        NECK_JOINT = self.dataset.get_keypoints().index('neck')

        x_portion_2d = range(0*num_joints, 1*num_joints)
        y_portion_2d = range(1*num_joints, 2*num_joints)

        skeleton = self.dataset.get_skeleton()

        # normalize joint lengths to be size of average
        x_norm, y_norm = self.normalize2D(
                                data_raw_2d[x_portion_2d,:],
                                data_raw_2d[y_portion_2d,:],
                                skeleton,
                                HEAD_JOINT)

        # center the coordinates of the neck into (0,0)
        data_2d_x = x_norm - x_norm[NECK_JOINT,:]
        data_2d_y = y_norm - y_norm[NECK_JOINT,:]

        # create the normalized feature representation matrix
        data_2d = np.concatenate((data_2d_x, data_2d_y), axis=0)

        pose_data_obj = pose_dataset( num_images,
                                      num_joints,
                                      data_raw_2d,
                                      data_2d,
                                      angles_gt,
                                      real2new,
                                      new2real,
                                      skeleton,
                                      HEAD_JOINT)

        print "[DATA_LOADER]: Done Loading 2D data, (t=%.2fs)."%(time.time() - tic)
        return pose_data_obj

    def load3D(self):
        print "[DATA_LOADER]: Loading 3D data.."
        tic = time.time()

        # load the 2D data
        dataset_obj = self.load2D()
        new2real    = dataset_obj.new2real

        num_joints = dataset_obj.num_joints
        num_images = dataset_obj.num_images
        HEAD_JOINT = self.dataset.get_keypoints().index('head')
        NECK_JOINT = self.dataset.get_keypoints().index('neck')

        x_portion_3d = range(0*num_joints, 1*num_joints)
        y_portion_3d = range(1*num_joints, 2*num_joints)
        z_portion_3d = range(2*num_joints, 3*num_joints)
        num_dim_3d   = 3*num_joints

        ann_ids = self.dataset.get_ann_ids(activity_list=self.activity_list)
        anns    = sorted(self.dataset.get_anns(ann_ids),key=lambda k: k['id'])

        ####################################################################
        ## NOTE: we are assuming that dataset has 3d keypoints available
        ####################################################################

        data_raw_3d = []
        for aind, a in enumerate(anns):
            # append keypoints in matrix
            data_raw_3d.append(a['3d_keypoints'][0::3] + a['3d_keypoints'][1::3] \
                             + a['3d_keypoints'][2::3])

        data_raw_3d = np.array(data_raw_3d).T

        skeleton = self.dataset.get_skeleton()

        # normalize joint lengths to be size of average
        x_norm, y_norm, z_norm = self.normalize3D(
                                        data_raw_3d[x_portion_3d,:],
                                        data_raw_3d[y_portion_3d,:],
                                        data_raw_3d[z_portion_3d,:],
                                        skeleton,
                                        HEAD_JOINT)

        # center in the coordinates of the ROOTINDEX (neck)
        data_3d_x = x_norm - x_norm[NECK_JOINT,:]
        data_3d_y = y_norm - y_norm[NECK_JOINT,:]
        data_3d_z = z_norm - z_norm[NECK_JOINT,:]

        # create the feature representation matrix
        data_3d = np.concatenate((data_3d_x, data_3d_y, data_3d_z), axis=0)

        dataset_obj.data_raw_3d  = data_raw_3d
        dataset_obj.data_3d      = data_3d
        dataset_obj.x_portion_3d = x_portion_3d
        dataset_obj.y_portion_3d = y_portion_3d
        dataset_obj.z_portion_3d = z_portion_3d
        dataset_obj.num_dim_3d   = num_dim_3d

        print "[DATA_LOADER]: Done Loading 3D data, (t=%.2fs)."%(time.time() - tic)
        return dataset_obj

    @staticmethod
    def normalize2D(x, y, skeleton, HEAD_JOINT):
        """
        Normalize the length of each joint with the average length
        for that segment over all training set.
        Starting from the head down to the rest of the body
        """
        ## NOTE: The code assumes that for every skeleton pair
        #        the smaller index is the parent of the larger index.
        #        I.e.: [2,8] -> [left shoulder, left hip]

        num_joints = x.shape[0]
        num_images = x.shape[1]

        x_norm = np.zeros((num_joints,num_images))
        y_norm = np.zeros((num_joints,num_images))

        # for each bone extract mean length over all data
        mean_bone_lengths = {}
        for bone in skeleton:
            mean_bone_lengths[bone[0],bone[1]] = np.mean(np.sqrt(\
                (x[bone[0],:]-x[bone[1],:])**2 + \
                (y[bone[0],:]-y[bone[1],:])**2 ))

        # normalize one image at the time by
        for im in range(num_images):
            x_coords = x[:,im]
            y_coords = y[:,im]

            # assume that the location of the head joint does not change
            x_norm[HEAD_JOINT,im] = x_coords[HEAD_JOINT]
            y_norm[HEAD_JOINT,im] = y_coords[HEAD_JOINT]

            for bone in skeleton:
                # get parent and child coordinates
                j_p = bone[0]
                j_c = bone[1]

                p_coord = np.array([x_coords[j_p],y_coords[j_p]])
                c_coord = np.array([x_coords[j_c],y_coords[j_c]])

                # get difference vector
                bone_vec    = c_coord - p_coord
                bone_length = np.sqrt( bone_vec[0]**2 + bone_vec[1]**2)

                if bone_length == 0:
                    print im,bone
                    assert(False)

                # add normalized difference vector to parent coordinates
                norm_bone_vec   = bone_vec * (mean_bone_lengths[bone[0],bone[1]] / bone_length)
                norm_p_coord    = np.array([x_norm[j_p,im],y_norm[j_p,im]])
                norm_bone_coord = norm_bone_vec + norm_p_coord

                x_norm[j_c,im] = norm_bone_coord[0]
                y_norm[j_c,im] = norm_bone_coord[1]

        return x_norm, y_norm

    @staticmethod
    def normalize3D(x, y, z, skeleton, HEAD_JOINT):
        """
        Normalize the length of each joint with the average length
        for that segment over all training set.
        Starting from the head down to the rest of the body
        """
        ## NOTE: The code assumes that for every skeleton pair
        #        the smaller index is the parent of the larger index.
        #        I.e.: [2,8] -> [left shoulder, left hip]

        num_joints = x.shape[0]
        num_images = x.shape[1]

        x_norm = np.zeros((num_joints,num_images))
        y_norm = np.zeros((num_joints,num_images))
        z_norm = np.zeros((num_joints,num_images))

        # for each bone extract mean length over all data
        mean_bone_lengths = {}
        for bone in skeleton:
            mean_bone_lengths[bone[0],bone[1]] = np.mean(np.sqrt(\
                (x[bone[0],:]-x[bone[1],:])**2 + \
                (y[bone[0],:]-y[bone[1],:])**2 + \
                (z[bone[0],:]-z[bone[1],:])**2 ))

        # normalize one image at the time by
        for im in range(num_images):
            x_coords = x[:,im]
            y_coords = y[:,im]
            z_coords = z[:,im]

            x_norm[HEAD_JOINT,im] = x_coords[HEAD_JOINT]
            y_norm[HEAD_JOINT,im] = y_coords[HEAD_JOINT]
            z_norm[HEAD_JOINT,im] = z_coords[HEAD_JOINT]

            for bone in skeleton:
                j_p = bone[0]
                j_c = bone[1]

                p_coord = np.array([x_coords[j_p],y_coords[j_p],z_coords[j_p]])
                c_coord = np.array([x_coords[j_c],y_coords[j_c],z_coords[j_c]])

                # get difference vector
                bone_vec    = c_coord - p_coord
                bone_length = np.sqrt( bone_vec[0]**2 + bone_vec[1]**2 + bone_vec[2]**2 )

                if bone_length == 0:
                    print im,bone
                    assert(False)

                # add normalized difference vector to parent coordinates
                norm_bone_vec   = bone_vec * (mean_bone_lengths[bone[0],bone[1]] / bone_length)
                norm_p_coord    = np.array([x_norm[j_p,im],y_norm[j_p,im],z_norm[j_p,im]])
                norm_bone_coord = norm_bone_vec + norm_p_coord

                x_norm[j_c,im] = norm_bone_coord[0]
                y_norm[j_c,im] = norm_bone_coord[1]
                z_norm[j_c,im] = norm_bone_coord[2]

        return x_norm, y_norm, z_norm
