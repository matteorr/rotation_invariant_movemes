import numpy as np
import time
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
import operator
import random

"""
Class for bucketing poses into clusters based on the viewing angles
"""
class bucketer:
    def __init__(self, dataset_obj, num_buckets, metric_type):
        print "[BUCKETING]: Initializing..."
        tic = time.time()

        self.num_buckets     = num_buckets
        self.metric_type     = metric_type

        self.data_2d  = dataset_obj.data_2d
        self.angle_gt = dataset_obj.angles_gt
        self.real2new = dataset_obj.real2new
        self.new2real = dataset_obj.new2real

        self.x_portion_2d = dataset_obj.x_portion_2d
        self.y_portion_2d = dataset_obj.y_portion_2d

        self.num_dim_2d = dataset_obj.num_dim_2d
        self.num_images = dataset_obj.num_images
        self.num_joints = dataset_obj.num_joints

        print "[BUCKETING]: Done Initializing, (t=%.2fs)."%(time.time() - tic)

    def get_angle_anns(self):
        """
        Recover the angle of view that each pose is facing.
        Use ground truth if available, a heuristic, a coarsening of gt
        or random values.
        """
        print "[BUCKETING]: Getting angle anns with metric:[%s]"%(self.metric_type)
        tic = time.time()

        angle_anns = []
        ## Metric type
        if self.metric_type == 'gt':
            angle_anns = self.angle_gt

        elif self.metric_type == 'heuristic':
            # vector with [image_id, angle of view in radiants, slope of mid torso]
            raise NotImplementedError("Use metric_type='coarse' for heuristic.")
            angle_anns = self.heuristic_metric()

        elif self.metric_type == 'coarse':
            buckets_index, buckets_table, buckets_centers = self.get_coarse_angles()
            angle_anns = []
            for a in self.angle_gt:
                new_angle = dict()
                new_angle['real_id'] = a['real_id']
                new_angle['new_id']  = a['new_id']

                coarse_angle = buckets_centers[buckets_table[a['new_id']]]
                new_angle['angle_rad'] = np.radians(coarse_angle)
                new_angle['angle_deg'] = coarse_angle
                angle_anns.append(new_angle)

        elif self.metric_type == 'random':
            angle_anns = []
            for a in self.angle_gt:
                new_angle = dict()
                new_angle['real_id'] = a['real_id']
                new_angle['new_id'] = a['new_id']

                random_angle = random.random() * 360.
                new_angle['angle_rad'] = np.radians(random_angle)
                new_angle['angle_deg'] = random_angle
                angle_anns.append(new_angle)

        else:
            raise ValueError("[BUCKETING]: Uknown metric_type [%s]!"%self.metric_type)

        print "[BUCKETING]: Done, (t=%.2fs)."%(time.time() - tic)
        return angle_anns

    def get_coarse_angles(self):
        """
        Coarse initialization of angles
        """
        angle_anns = self.angle_gt

        # NOTE: in order to cluster angular data we must do it in the 2d space
        # on the unit circle. Otherwise angles 360 and 1 will be considered far
        angle_data      = np.zeros((len(angle_anns),2))
        angle_data[:,0] = [math.cos(a['angle_rad']) for a in angle_anns]
        angle_data[:,1] = [math.sin(a['angle_rad']) for a in angle_anns]

        id_data    = [a['new_id'] for a in angle_anns]

        index_clust  = []
        new_id_table = {}
        bucket_centers = {}

        ## Use kmeans to cluster the poses by their angle of view
        clf = KMeans(n_clusters=self.num_buckets)

        clf.fit(angle_data)
        labels  = clf.labels_

        for l in np.unique(labels):
            l_label_entries = \
              [angle_anns[k] for k in range(len(angle_anns)) if labels[k]==l]
            index_clust.append(l_label_entries)
            tmp_alphas = [(np.cos(a['angle_rad']), np.sin(a['angle_rad'])) for a in l_label_entries]
            tmp_mean = [sum(a) / float(len(a)) for a in zip(*tmp_alphas)]
            tmp_centroid = math.degrees(math.atan2(tmp_mean[1], tmp_mean[0]))
            bucket_centers[l] = tmp_centroid % 360

        for l in xrange(len(labels)):
            new_id_table[id_data[l]] = labels[l]

        total_images = 0
        for b in xrange(self.num_buckets):
            total_images += len(index_clust[b])

        assert(len(index_clust)==self.num_buckets)
        assert(total_images == self.num_images)
        assert(len(new_id_table) == self.num_images)

        return index_clust, new_id_table, bucket_centers

    def bucket_poses(self):
        print "[BUCKETING]: Bucketing poses..."
        tic = time.time()

        angle_anns = self.get_angle_anns()

        # NOTE: in order to cluster angular data we must do it in the 2d space
        # on the unit circle. Otherwise angles 360 and 1 will be considered far
        angle_data      = np.zeros((len(angle_anns),2))
        angle_data[:,0] = [math.cos(a['angle_rad']) for a in angle_anns]
        angle_data[:,1] = [math.sin(a['angle_rad']) for a in angle_anns]

        id_data    = [a['new_id'] for a in angle_anns]

        index_clust  = []
        new_id_table = {}
        bucket_centers = {}

        ## Use kmeans to cluster the poses by their angle of view
        clf = KMeans(n_clusters=self.num_buckets)

        clf.fit(angle_data)
        labels  = clf.labels_

        for l in np.unique(labels):
            l_label_entries = \
              [angle_anns[k] for k in range(len(angle_anns)) if labels[k]==l]
            index_clust.append(l_label_entries)
            tmp_alphas = [(np.cos(a['angle_rad']), np.sin(a['angle_rad'])) for a in l_label_entries]
            tmp_mean = [sum(a) / float(len(a)) for a in zip(*tmp_alphas)]
            tmp_centroid = math.degrees(math.atan2(tmp_mean[1], tmp_mean[0]))
            bucket_centers[l] = tmp_centroid % 360

        for l in xrange(len(labels)):
            new_id_table[id_data[l]] = labels[l]

        print "[BUCKETING]: Number of buckets: [%d]"%self.num_buckets
        total_images = 0

        for b in xrange(self.num_buckets):
            print " - Bucket [%d]: %d"%(b,len(index_clust[b]))
            total_images += len(index_clust[b])

        assert(len(index_clust)==self.num_buckets)
        assert(total_images == self.num_images)
        assert(len(new_id_table) == self.num_images)

        self.buckets_index = index_clust
        self.buckets_table = new_id_table
        self.buckets_centers = bucket_centers

        print "[BUCKETING]: Done Bucketing, (t=%.2fs)."%(time.time() - tic)
        return index_clust, new_id_table

    def heuristic_metric(self):
        return -1
