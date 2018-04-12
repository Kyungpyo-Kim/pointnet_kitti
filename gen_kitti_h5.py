# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:45:21 2018

@author: Kyungpyo
"""

import os
import numpy as np
import sys

from kitti_visu_tool import*
from utils.kitti_loader import KittiLoader
from utils.data_prep_util import*
from utils.gen_kitti_h5_util import*

# Constants

data_dir = os.path.join(ROOT_DIR, 'data')

data_dim = [POINT_NUM, INPUT_DIM]
label_dim = [POINT_NUM]

data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
data_dir = os.path.abspath('./data')
output_dir = os.path.join(data_dir, 'kitti_hdf5_data')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
output_visu_dir = os.path.join(output_dir, 'visu')
if not os.path.exists(output_visu_dir):
    os.mkdir(output_visu_dir)

output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_kitti_filelist = os.path.join(output_dir, 'kitti_filelist.txt')

fout_kitti = open(output_kitti_filelist, 'w')


# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save


# insert batch to h5 files
def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]

        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        fout_kitti.write(h5_filename + "\n")
        #print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))

        h5_index += 1
        buffer_size = 0

        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        fout_kitti.write(h5_filename + "\n")
        #print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return
#------------------------------------------------------------------------------


# load kitti data and convert to 'data' and 'label' from a file
def load_data_label(dataset, file_idx = 0):

  # point data parsing
  point_cloud = dataset.load_specified(file_idx)[3]
  labels = dataset.load_specified(file_idx)[1]

  x = point_cloud[0, :, 0].tolist()
  y = point_cloud[0, :, 1].tolist()
  z = point_cloud[0, :, 2].tolist()

  roi_filter_point = filter_roi_point_xyz(point3d = CPoint3D(x,y,z),
                                          x = ROI_X ,y = ROI_Y ,z = ROI_Z , theta = ROI_THETA)
  
  data = np.array([roi_filter_point.x, roi_filter_point.y, roi_filter_point.z])

  # reshape
  kitti_data_label = np.zeros((len(roi_filter_point.x),4)) # for xyz
  kitti_data_label[:, :3] = data.T

  # label data parsing
  _, idx_car_ = filter_boxs_point(roi_filter_point, labels, 'Car')
  _, idx_cycl_ = filter_boxs_point(roi_filter_point, labels, 'Cyclist')
  _, idx_pedes_ = filter_boxs_point(roi_filter_point, labels, 'Pedestrian')

  idx_car = np.array(idx_car_).flatten()
  idx_cycl = np.array(idx_cycl_).flatten()
  idx_pedes = np.array(idx_pedes_).flatten()

  # reshape
  if len(idx_car) > 0 :
    kitti_data_label[idx_car, 3] = g_class2label['Car']
  if len(idx_cycl) > 0 :
    kitti_data_label[idx_cycl, 3] = g_class2label['Cyclist']
  if len(idx_pedes) > 0 :
    kitti_data_label[idx_pedes, 3] = g_class2label['Pedestrian']

  return kitti_data_label
#------------------------------------------------------------------------------


def load_data_label_test(dataset, file_idx = 0):
  # point data parsing
  point_cloud = dataset.load_specified(file_idx)[3]
  labels = dataset.load_specified(file_idx)[1]

  x = point_cloud[0, :, 0].tolist()
  y = point_cloud[0, :, 1].tolist()
  z = point_cloud[0, :, 2].tolist()

  roi_filter_point = filter_roi_point_xyz(point3d = CPoint3D(x,y,z),
                                          x = ROI_X ,y = ROI_Y ,z = ROI_Z , theta = ROI_THETA)
  
  data = np.array([roi_filter_point.x, roi_filter_point.y, roi_filter_point.z])

  # reshape
  kitti_data_label = np.zeros((len(roi_filter_point.x),4)) # for xyz and label
  kitti_data_label[:, :3] = data.T


  for i in range(len(kitti_data_label)):
    if z[i] < 3:
        kitti_data_label[i, 3] = 1
    if z[i] < 1:
        kitti_data_label[i, 3] = 2
    if z[i] < -1:
        kitti_data_label[i, 3] = 3

  return kitti_data_label
#------------------------------------------------------------------------------


# kitti data to room to block
def kitti_room2blocks_wrapper_normalized(dataset, file_idx = 0, is_train = True, block_size = 5.0, stride = 2.5):

  kitti_data_label = load_data_label(dataset, file_idx)

  return room2blocks_plus_normalized_xyz(kitti_data_label, 
                                                       x = ROI_X,
                                                       y = ROI_Y, 
                                                       z = ROI_Z,
                                                       num_point = POINT_NUM, 
                                                       block_size = block_size, 
                                                       stride = stride, 
                                                       random_sample=False, 
                                                       sample_num=None, 
                                                       sample_aug=1,
						       is_train = is_train)
#------------------------------------------------------------------------------


# main
def main():
  print("--main--")

  dataset = load_kitti_data(DATASET_DIR)

  visu = True
  
  range_ = len(dataset)
  #range_ = 1000
  for idx in range(range_):
    if idx % 10 == 0:
      print ("-", end=" ")
    if idx % 100 == 0:
      print ("-")  
      print("gen kitti hdf5 dataset:", idx, range_)
    data, label = kitti_room2blocks_wrapper_normalized(dataset, file_idx = idx)
    #print('{0}, {1}'.format(data.shape, label.shape))
    if len(data) == 0:
      continue
    
    # Accumulate data to batch size and save data into hd5f file format
    insert_batch(data, label, idx == range_-1)
    
    if visu:
      fout = open(os.path.join(output_dir, "visu/" + "{0}".format(idx)+'_gt.obj'), 'w')

      for idx_pt in range(len(data[0])):
        color = g_label2color[label[0,idx_pt]]
        fout.write('v %f %f %f %d %d %d\n' % (data[0,idx_pt,0], 
                                              data[0,idx_pt,1], 
                                              data[0,idx_pt,2], 
                                              color[0], 
                                              color[1], 
                                              color[2]))

      fout.close()      
  fout_kitti.close()
#------------------------------------------------------------------------------


if __name__ == "__main__":
  main()

