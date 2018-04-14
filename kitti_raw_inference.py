import argparse
import os
import sys
import time

import pykitti

from model import *
from gen_kitti_h5 import *
from utils.gen_kitti_h5_util import *
from utils.provider import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--out_dir', default='./kitti_raw_infer', help='output folder path')
parser.add_argument('--date', default='2011_09_26', help='date of kitti raw data')
parser.add_argument('--drive', default='0001', help='drive number of kitti raw data')
parser.add_argument('--scene_num', type=int, default=0, help='number of scene')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
SCENE_NUM = FLAGS.scene_num
DATE = FLAGS.date
DRIVE = FLAGS.drive
IN_DIR = os.path.join(os.getcwd(), 'data/raw')

OUT_DIR = FLAGS.out_dir
if not os.path.exists(OUT_DIR):  os.mkdir(OUT_DIR)

DUMP_DIR = os.path.join(OUT_DIR, 'dump')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

DUMP_EVAL_DIR = os.path.join(OUT_DIR, 'eval')
if not os.path.exists(DUMP_EVAL_DIR): os.mkdir(DUMP_EVAL_DIR)

LOG_FOUT = open(os.path.join(DUMP_EVAL_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = OUT_DIM


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, POINT_NUM)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    total_correct = 0
    total_seen = 0
    #fout_out_filelist = open(FLAGS.output_filelist, 'w')

    for i in range(7000):
      eval_one_scene(sess, ops, i)

    #eval_one_epoch_kitti(sess, ops)

    #fout_out_filelist.close()
    #log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_scene(sess, ops, idx):
  
  error_cnt = 0
  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  total_seen_class = [0 for _ in range(NUM_CLASSES)]
  total_correct_class = [0 for _ in range(NUM_CLASSES)]

  date = DATE
  drive = DRIVE
  dataset = load_dataset(date, drive)
  
  points = 1 # Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
  points_step = int(1. / points)
  point_size = 0.01 * (1. / points)
  dataset_velo = list(dataset.velo)
  frame = idx
  velo_range = range(0, dataset_velo[frame].shape[0], points_step)
  velo_frame = dataset_velo[frame][velo_range, :]    

  velo_frame[:,3] = 0

  kitti_data_label = velo_frame

  file_idx = 0
  is_train = False
  block_size = 5.0
  stride = 5.0


  data, label =  room2blocks_plus_normalized_xyz(
			kitti_data_label, 
			x = [-20, 80], 
			y = [-20, 20], 
			z = [-3, 10], 
			num_point = POINT_NUM, 
			block_size = block_size, 
			stride = stride, 
			random_sample=False, 
			sample_num=None, 
			sample_aug=1, 
			is_train = is_train)
  
  fout_gt = open(os.path.join(DUMP_DIR, '%06d_gt.obj'%idx), 'w')
  fout_pred = open(os.path.join(DUMP_DIR, '%06d_pred.obj'%idx), 'w')
  fout_eval = open(os.path.join(DUMP_DIR, '%06d_eval.obj'%idx), 'w')
  
  # ground truth object file
  for i_ in range(len(data)):
    for j_ in range(POINT_NUM):
        color_gt = g_label2color[label[i_,j_]] 
        fout_gt.write('v %f %f %f %d %d %d\n' % (data[i_,j_,0], data[i_,j_,1], data[i_,j_,2], color_gt[0], color_gt[1], color_gt[2]))
  
  start = time.time()
  # run session
  for batch_idx in range(len(data)):
  #for batch_idx in range(3):
    feed_dict = {ops['pointclouds_pl']: [data[batch_idx]],
                     ops['labels_pl']: [label[batch_idx]],
                     ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
    # predicted object file
    pred_label = np.argmax(pred_val[:,:,0:OUT_DIM], 2)
    
    for j_ in range(POINT_NUM):
      color_pred = g_label2color[pred_label[0,j_]]
      fout_pred.write('v %f %f %f %d %d %d\n' % (data[batch_idx,j_,0], data[batch_idx,j_,1], data[batch_idx,j_,2], color_pred[0], color_pred[1], color_pred[2]))  
      total_seen += 1
      if label[batch_idx,j_] == pred_label[0,j_]:
        color_eval = 1.0
        total_correct += 1
      else:
        color_eval = 0.0
      fout_eval.write('v %f %f %f %d %d %d\n' % (data[batch_idx,j_,0], data[batch_idx,j_,1], data[batch_idx,j_,2], color_eval, color_eval, color_eval))    
    
  end = time.time() - start
  print(total_correct/total_seen)
  print(end)
  print(total_seen)

  fout_gt.close()
  fout_pred.close()
  fout_eval.close()


def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.
    
    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """

    """
    Usage example

    date = '2011_09_26'
    drive = '0001'
    dataset = load_dataset(date, drive)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), 'data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

    """

    dataset = pykitti.raw(IN_DIR, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset



if __name__=='__main__':
  with tf.Graph().as_default():
    evaluate()
  LOG_FOUT.close()
