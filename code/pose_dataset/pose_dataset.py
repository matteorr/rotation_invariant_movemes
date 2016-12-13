import time
import numpy as np

"""
Class that stores relevant parameters and data for LSP
"""
class pose_dataset:
    def __init__(self,
                 num_images,
                 num_joints,
                 data_raw_2d,
                 data_2d,
                 angles_gt,
                 real2new,
                 new2real,
                 skeleton,
                 HEAD_JOINT):

        print "[POSE_DATASET]: Initializing..."
        tic = time.time()

        # original dataset info
        self.num_joints = num_joints
        self.num_images = num_images
        self.angles_gt  = angles_gt

        # 2d data
        self.data_raw_2d  = data_raw_2d
        self.data_2d      = data_2d
        self.num_dim_2d   = np.shape(self.data_2d)[0]
        self.x_portion_2d = range(0, self.num_joints)
        self.y_portion_2d = range(self.num_joints,2*self.num_joints)

        # 3d data
        self.data_raw_3d  = None
        self.data_3d      = None
        self.num_dim_3d   = None
        self.x_portion_3d = None
        self.y_portion_3d = None
        self.z_portion_3d = None

        # info to go from old to new indexes
        self.real2new = real2new
        self.new2real = new2real

        self.skeleton = skeleton
        self.HEAD_JOINT = HEAD_JOINT

        print "[POSE_DATASET]: Done Initializing, (t=%.2fs)."%(time.time() - tic)
