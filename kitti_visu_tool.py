# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:18:50 2018

@author: Kyungpyo
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

import utils
from utils.config import cfg
from utils.utils import*
from utils.kitti_loader import KittiLoader


dataset_dir = './data/kitti/training'

'''
Parameter:
  dataset_dir

    dataset_dir
      ├── image_2
      ├── label_2
      └── velodyne
'''

cls_color_dict = {
    'Car' : 'blue',
    'Cyclist' : 'cyan',
    'Pedestrian' : 'red'}


#----------------------------------------------------------------------------#
# Class ---------------------------------------------------------------------#
#----------------------------------------------------------------------------#

# point xyz class
class CPoint3D:

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __len__(self):
    return len(self.x)

  def __add__(self, point3d):
    if isinstance(point3d.x, list):
      self.x.extend(point3d.x)
      self.y.extend(point3d.y)
      self.z.extend(point3d.z)
    else:
      self.x.append(point3d.x)
      self.y.append(point3d.y)
      self.z.append(point3d.z)

    return self

  def __getitem__(self, key):
    point3d = CPoint3D(self.x[key], self.y[key], self.z[key])
    return point3d

  def pop(self, idx):
    for idx_ in idx:
      self.x.pop(idx_)
      self.y.pop(idx_)
      self.z.pop(idx_)
#------------------------------------------------------------------------------




# class plot builder
class CPlotBuilder:
    def __init__(self, ax, dataset):
      self.keyIn = ''
      self.ax = ax
      self.cid = self.ax.figure.canvas.mpl_connect('key_press_event', self)
      self.dataset = dataset
      self.idx = 0

      text_data_len = "Data range: 0 ~ " + str(len(dataset) - 1)

      self.ax.text2D(0.05, 0.95, text_data_len, transform=self.ax.transAxes)

      plot_main(self.ax, self.dataset, self.idx)
      print("[info] draw scene index: ", self.idx)

    def __call__(self, event):

      # number
      if event.key != 'enter':
        print('Press', event.key)

        self.keyIn = self.keyIn + event.key

        print(self.dataset)

      # enter
      if event.key == 'enter':
        print('[info] keyIn: ', self.keyIn)

        self.idx = int(self.keyIn)
        self.keyIn = ''


        plot_main(self.ax, self.dataset, self.idx)
        print("[info] draw scene index: ", self.idx)


      # left
      if event.key == 'left':
        print('[info] keyIn: ', self.keyIn)
        self.keyIn = ''

        if self.idx > 0:
          self.idx = self.idx - 1

        plot_main(self.ax, self.dataset, self.idx)
        print("[info] draw scene index: ", self.idx)

      # left
      if event.key == 'right':
        print('[info] keyIn: ', self.keyIn)
        self.keyIn = ''

        if self.idx < len(self.dataset) - 1:
          self.idx = self.idx + 1

        plot_main(self.ax, self.dataset, self.idx)
        print("[info] draw scene index: ", self.idx)
#------------------------------------------------------------------------------




#----------------------------------------------------------------------------#
# Function ------------------------------------------------------------------#
#----------------------------------------------------------------------------#

# Data processing
#   load kitti data
def load_kitti_data(dataset_dir = './data/training'):
  # loading dataset
  '''
  Parameter:
    dataset_dir

      dataset_dir
        ├── image_2
        ├── label_2
        └── velodyne
  '''

  with KittiLoader(object_dir=dataset_dir, queue_size=50, require_shuffle=False, is_testset=False,
            use_multi_process_num=1, multi_gpu_sum=1, aug=True) as dataset :
    print("[info] loaded kitti data")
    # dataset load example
    # dest = dataset.load_specified(index1)[index2]
    # Parameters
    # index1: number of file
    # index2: 0; tag(sequence)
    #         1; labels - bounding box in camera coordinates
    #         2; rgb
    #         3; lidar - x, y, z, rz
  return dataset
#------------------------------------------------------------------------------




# Data processing
#   filter boxes point
def filter_box_point(point3d, corner_point):

  x1 = min(corner_point[:, 0])
  x2 = max(corner_point[:, 0])
  y1 = min(corner_point[:, 1])
  y2 = max(corner_point[:, 1])
  z1 = min(corner_point[:, 2])
  z2 = max(corner_point[:, 2])

  x_np = np.array(point3d.x)
  y_np = np.array(point3d.y)
  z_np = np.array(point3d.z)

  idx = np.where((x_np > x1 - 0.1) & (x_np < x2 + 0.1)
                    &(y_np > y1 - 0.1) & (y_np < y2 + 0.1)
                    &(z_np > z1 - 0.1) & (z_np < z2 + 0.1))

  point3d_out = CPoint3D(list(),list(),list())

  for i in idx[0]:
    p = point3d[i]
    point3d_out = point3d_out + p

  return point3d_out, idx[0]
#------------------------------------------------------------------------------



# Data processing
#   filter box point
def filter_boxs_point(point3d, labels, cls_target = 'Car'):

  # get box from labels
  gt_box3d = label_to_gt_box3d(labels, cls_target, coordinate='lidar')

  # get box corner point
  corner_points = center_to_corner_box3d(gt_box3d[0])

  # get points in bbox
  filter_point3d = CPoint3D(list(),list(),list())
  idx = list()
  for i_cp in corner_points:
    box_point3d, idx_ = filter_box_point(point3d, i_cp)
    filter_point3d = filter_point3d + box_point3d
    idx.extend(idx_.tolist())
  return filter_point3d, idx
#------------------------------------------------------------------------------




# Data processing
#   ROI filter point
def filter_roi_point_xy(point3d, x = [5, 80], theta = 45.0, z = [-5, 5]):

  filter_point3d = CPoint3D(list(),list(),list())
  for i_ in range(len(point3d)):
    x_ = point3d[i_].x
    y_ = point3d[i_].y
    z_ = point3d[i_].z

    if x_ > x[0] and x_ < x[1]:
      if z_ > z[0] and z_ < z[1]:
        tan_theta = math.tan(math.pi * theta / 180.0)
        if tan_theta > abs(y_/x_):
          filter_point3d = filter_point3d + CPoint3D(x_, y_, z_)

  return filter_point3d
def filter_roi_point_xyz(point3d, x = [5, 75], y = [-40, 40], z = [-5, 5], theta = 45.0):
  filter_point3d = CPoint3D(list(),list(),list())
  for i_ in range(len(point3d)):
    x_ = point3d[i_].x
    y_ = point3d[i_].y
    z_ = point3d[i_].z

    if x_ > x[0] and x_ < x[1]:
      if y_ > y[0] and y_ < y[1]:  
        if z_ > z[0] and z_ < z[1]:
          tan_theta = math.tan(math.pi * theta / 180.0)
          if tan_theta > abs(y_/x_):
            filter_point3d = filter_point3d + CPoint3D(x_, y_, z_)

  return filter_point3d

#------------------------------------------------------------------------------





# Figure generation
#   draw box
def plot_draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices.T
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
      pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)
#------------------------------------------------------------------------------



# Figure generation
#   draw class box
def plot_draw_cls_box(pyplot_axis, labels, cls_target = 'Car', color = 'black'):
  # get box from labels
  gt_box3d = label_to_gt_box3d(labels, cls_target, coordinate='lidar')

  # get box corner point
  corner_points = center_to_corner_box3d(gt_box3d[0])
  for i_cp in corner_points:
    # plot bounding box
    plot_draw_box(pyplot_axis, i_cp, color = color)
#------------------------------------------------------------------------------




# Figure generation
#   plot cofig
def plot3d_config(pyplot_axis):
  axes_limits = [ [-20, 80],  # X axis range
                  [-40, 40],  # Y axis range
                  [-3 , 10] ] # Z axis range

  axes=[0, 1, 2]

  pyplot_axis.set_xlim3d(*axes_limits[axes[0]])
  pyplot_axis.set_ylim3d(*axes_limits[axes[1]])
  pyplot_axis.set_zlim3d(*axes_limits[axes[2]])
  plt.draw()
#------------------------------------------------------------------------------



# Figure generation
#   plot 3d points
def plot_3d_points(pyplot_axis, point3d, size = 0.5, color = 'green'):

  # draw point cloud
  pyplot_axis.scatter(point3d.x, point3d.y, point3d.z, s = size, c = color)
  plt.draw()
#------------------------------------------------------------------------------




# Figure generation
#  plot main
def plot_main(pyplot_axis, dataset, idx = 0):

  # clear
  pyplot_axis.clear()


  labels = dataset.load_specified(idx)[1]
  point_cloud = dataset.load_specified(idx)[3]

  x = point_cloud[0, :, 0].tolist()
  y = point_cloud[0, :, 1].tolist()
  z = point_cloud[0, :, 2].tolist()

  roi_filter_point = filter_roi_point_xyz(CPoint3D(x,y,z), x = [5, 45], y = [-20, 20], z = [-3,1])

  car_filter_points, car_idx = filter_boxs_point(roi_filter_point, labels, 'Car')
  cyclist_filter_points, cycl_idx = filter_boxs_point(roi_filter_point, labels, 'Cyclist')
  pedestrian_filter_points, pedes_idx = filter_boxs_point(roi_filter_point, labels, 'Pedestrian')

  #all_point3d = CPoint3D(x, y, z)
  # draw all points
  remove_idx = list()
  remove_idx.extend(car_idx)
  remove_idx.extend(cycl_idx)
  remove_idx.extend(pedes_idx)
  roi_filter_point.pop(remove_idx)
  
  plot_3d_points(pyplot_axis, roi_filter_point, size = 1)
  print("roi filter point:", len(roi_filter_point))

  # draw Car points
  plot_3d_points(pyplot_axis, car_filter_points, size = 1, color = cls_color_dict['Car'])
  # draw Car box
  plot_draw_cls_box(pyplot_axis, labels, 'Car', color = cls_color_dict['Car'])
  print("car filter point:", len(car_filter_points))


  # draw Cyclist points
  plot_3d_points(pyplot_axis, cyclist_filter_points, size = 1, color = cls_color_dict['Cyclist'])
  # draw Cyclist box
  plot_draw_cls_box(pyplot_axis, labels, 'Cyclist', color = cls_color_dict['Cyclist'])
  print("cycl filter point:", len(cyclist_filter_points))

  # draw Pedestrian points
  plot_3d_points(pyplot_axis, pedestrian_filter_points, size = 1, color = cls_color_dict['Pedestrian'])
  # draw Pedestrian box
  plot_draw_cls_box(pyplot_axis, labels, 'Pedestrian', color = cls_color_dict['Pedestrian'])
  print("pedes filter point:", len(pedestrian_filter_points))

  # plot cfg
  plot3d_config(pyplot_axis)
#------------------------------------------------------------------------------




# main
def main():
  print("[info] start of main")

  plt.close()

  dataset = load_kitti_data(dataset_dir)

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(111, projection='3d')

  plot_builder = CPlotBuilder(ax, dataset)
#------------------------------------------------------------------------------



if __name__ == "__main__":
  main()
