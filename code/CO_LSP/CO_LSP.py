import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import skimage.io as io
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

"""
Comprehensive LSP Dataset
"""
class CO_LSP:
    def __init__(self, ann_file, img_path):
        try:
            self.dataset = json.load(open(ann_file, 'rb'))
        except:
            raise ValueError("wrong path: %s."%ann_file)

        self.img_path    = img_path
        self.info        = self.dataset['info']
        self.licenses    = self.dataset['licenses']
        self.keypoints   = self.dataset['keypoints']
        self.skeleton    = self.dataset['skeleton']
        self.n_keypoints = len(self.keypoints)

        # Annotations
        self.anns = {a['id']:a for a in self.dataset['annotations']}

    def get_keypoints(self):
        return self.keypoints

    def get_skeleton(self):
        return self.skeleton

    def get_num_keypoints(self):
        return self.n_keypoints

    def get_info(self):
        for key, value in self.info.items():
            print '%s : %s'%(key, value)

    def get_ann_ids(self,imgs=[],activity_list=[],angle_lims=[0,360]):
        """
        Obtain the annotation ids that satisfy the conditions specified in the arguments.
        """
        ann_ids = []

        if len(angle_lims) != 2:
            raise ValueError("wrong angle_lims, please give [angle_min,angle_max]")
        else:
            # verify that amin < amax
            a1 = angle_lims[0] % 360
            a2 = angle_lims[1] % 360

        for a in self.dataset['annotations']:
            img_select = a['image'] in imgs if len(imgs) > 0 else True
            act_select = a['activity'] in activity_list if len(activity_list) > 0 else True
            if a1 < a2:
                ang_select = a1 <= a['angle_avg'] <= a2
            else:
                ang_select = a1 < a['angle_avg'] or a['angle_avg'] < a2

            if img_select and act_select and ang_select:
                ann_ids.append(a['id'])

        return ann_ids

    def get_anns(self,ann_ids=[]):
        """
        Obtain the annotations from the given annotation ids.
        """
        if type(ann_ids)==list:
            return [self.anns[id] for id in ann_ids]
        elif type(ann_ids)==int:
            return [self.anns[ann_ids]]

    def show_anns(self,anns):
        """
        Show the plots of image, angle of view, 2D skeleton, and 3D skeleton
        for the annotations selected.
        """
        #pdf = PdfPages('./test.pdf')
        for a in anns:
            plt.figure(figsize=(20, 5))
            plt.suptitle('%s'%a['image'], fontsize=16)
            gs = gridspec.GridSpec(1,4)
            ax1 = plt.subplot(gs[0,0])
            ax1.set_title(a['activity'])
            ax2 = plt.subplot(gs[0,1])
            ax2.set_title('angle = %0.2f degrees'%a['angle_avg'])
            ax3 = plt.subplot(gs[0,2])
            ax3.set_title('2D Skeleton')
            ax4 = plt.subplot(gs[0,3], projection='3d')
            ax4.set_title('3D Skeleton')

            i = io.imread('%s/%s'%(self.img_path, a['image']))
            ax1.imshow(i)
            ax1.axis('off')
            ax2 = self.plot_angle_view(ax2, a)
            ax3 = self.draw_2d_skeleton(ax3, a['2d_keypoints'][0::2],a['2d_keypoints'][1::2], self.skeleton)
            ax4 = self.draw_3d_skeleton(ax4, a['3d_keypoints'][0::3],a['3d_keypoints'][1::3],a['3d_keypoints'][2::3], self.skeleton)

            plt.show()
            #pdf.savefig()
            #plt.close()
            #pdf.close()

    ## Helper Functions for skeleton plots
    @staticmethod
    def draw_2d_skeleton(ax, x_joints, y_joints, skeleton,
                        limit_axis=False, xlim=[-40,40], ylim=[-120,20]):

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
        if limit_axis:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        ax.invert_yaxis()
        ax.scatter(x_joints, y_joints)
        for b in skeleton:
            ja = b[0]
            jb = b[1]
            x_coord = [x_joints[ja],x_joints[jb]]
            y_coord = [y_joints[ja],y_joints[jb]]
            color_string = '-%s'%(colors[[k for k in skeleton if k==b][0][1]])
            ax.plot(x_coord,y_coord,color_string)

        return ax

    @staticmethod
    def draw_3d_skeleton(ax, x_joints, y_joints, z_joints, skeleton,
                        limit_axis=False, xlim=[-40,40], ylim=[-120,20]):

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
        ax.scatter(x_joints, y_joints, z_joints)
        for b in skeleton:
            ja = b[0]
            jb = b[1]
            x_coord = [x_joints[ja],x_joints[jb]]
            y_coord = [y_joints[ja],y_joints[jb]]
            z_coord = [z_joints[ja],z_joints[jb]]
            color_string = '-%s'%(colors[[k for k in skeleton if k==b][0][1]])
            ax.plot(x_coord,y_coord,z_coord,color_string)
        ax.view_init(elev=-90, azim=-90)

        return ax

    @staticmethod
    def plot_angle_view(ax, ann):
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        ax.set_xticks([])
        ax.set_yticks([])
        angle = ann['angle_avg'] + 270 % 360
        circle1 = plt.Circle((0,0), 1, color='k', linewidth=2, fill=False)
        y = np.sin(np.radians(angle))
        x = np.cos(np.radians(angle))
        circle2 = plt.Circle((x, y), 0.1, facecolor='r', edgecolor='k', linewidth=2, fill=True)
        line1 = ax.plot([0, x], [0, y], linewidth=2, color='b')
        line2 = ax.plot([0, 0], [0,-1], '--r', linewidth=2)

        ax.add_artist(circle1)
        ax.add_artist(circle2)

        return ax
