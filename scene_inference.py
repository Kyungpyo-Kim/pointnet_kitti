import argparse
import os
import sys
import time

from model import *
from gen_kitti_h5 import *
from utils.gen_kitti_h5_util import *
from utils.provider import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--out_dir', default='./scene_infer', help='output folder path')
parser.add_argument('--scene_num', type=int, default=0, help='number of scene')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
SCENE_NUM = FLAGS.scene_num

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

    for i in range(10):
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

  dataset = load_kitti_data(DATASET_DIR)
  data, label = kitti_room2blocks_wrapper_normalized(dataset, file_idx = idx, is_train = False, block_size = 5.0, stride = 5.0)
  
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

  #print(data)
  #print(label)

def eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    if FLAGS.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_pred.obj'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_gt.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')

    current_data_xyzrgb = test_data
    current_data = np.concatenate((current_data_xyzrgb[:,:,0:3], current_data_xyzrgb[:,:,6:9]), axis = 2)
    current_label = test_label
    #print("current data samlpe: ", current_data[0,0,:])
    #print("current label samlpe: ", current_label[0])

    exit()
    # Get room dimension..
    data_label = np.load(room_path)
    data = data_label[:,0:6]
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)


    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        print("start/end: ", start_idx, end_idx)
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN

        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data_xyzrgb[start_idx+b, :, :]
            l = current_label[start_idx+b,:]
            pts[:,6] *= max_room_x
            pts[:,7] *= max_room_y
            pts[:,8] *= max_room_z
            pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(POINT_NUM):
                color = indoor3d_util.g_label2color[pred[i]]
                if current_label[start_idx+b, i] > NUM_CLASSES -1:
                    current_label[start_idx+b, i] = NUM_CLASSES -1
                color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
                if FLAGS.visu:
                    fout.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))
                fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i,6], pts[i,7], pts[i,8], pts[i,3], pts[i,4], pts[i,5], pred_val[b,i,pred[i]], pred[i]))
                fout_gt_label.write('%d\n' % (l[i]))
        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*POINT_NUM)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(POINT_NUM):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/POINT_NUM)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    fout_data_label.close()
    fout_gt_label.close()
    if FLAGS.visu:
        fout.close()
        fout_gt.close()
    return total_correct, total_seen

def eval_one_epoch_kitti(sess, ops):
    error_cnt = 0
    is_training = False
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_data_xyzrgb = test_data
    current_data = np.concatenate((current_data_xyzrgb[:,:,0:3], current_data_xyzrgb[:,:,6:9]), axis = 2)
    current_label = test_label

    print("current test data samlpe: ", current_data[0,0,:])
    print("current test label samlpe: ", current_label[0,0])

    print(len(current_data[:,0,0]))

    for idx in range(len(current_data[:,0,0])):
        if FLAGS.visu:
            fout = open(DUMP_DIR + str(idx) +'_pred.obj', 'w')
            fout_gt = open(DUMP_DIR + str(idx) +'_gt.obj', 'w')

            fout_txt = open(DUMP_EVAL_DIR + str(idx) +'_pred.txt', 'w')

        feed_dict = {ops['pointclouds_pl']: [current_data[idx, :, :]],
                     ops['labels_pl']: [current_label[idx]],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        fout_txt.write('%f, %f, %f, %f\n' % (   pred_val[0, 0, 0],
                                                pred_val[0, 0, 1],
                                                pred_val[0, 0, 2],
                                                pred_val[0, 0, 3]))
        fout_txt.write(str(current_data[idx, :, :]))
        fout_txt.write(str(current_label[idx]))
        fout_txt.write(str(pred_val))
        print(current_data[idx, :, :])
        print(current_label[idx])
        print(pred_val)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:3], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN

        pts = current_data[idx, :, :]
        for i in range(POINT_NUM):
            #fout_txt.write('%f %f %f %d %d %d %f %d\n' % (pts[i,6], pts[i,7], pts[i,8], pts[i,3], pts[i,4], pts[i,5], pred_val[b,i,pred[i]], pred[i]))

            color = indoor3d_util.g_label2color[pred_label[0,i]]
            color_gt = indoor3d_util.g_label2color[current_label[idx, i]]
            if FLAGS.visu:
                fout.write('v %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], color[0], color[1], color[2]))
                fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], color_gt[0], color_gt[1], color_gt[2]))

        fout.close()
        fout_gt.close()
        fout_txt.close()

    #return total_correct, total_seen

if __name__=='__main__':
  with tf.Graph().as_default():
    evaluate()
  LOG_FOUT.close()
