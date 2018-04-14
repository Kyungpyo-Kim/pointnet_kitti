# README

## Environment

* Ubuntu14.04LTS
* Python3.6
* Spyder3
* Tensorflow-gpu 1.5.0

## Folder-structure

```folder
.
├── build	# Cython build file
├── doc
├── utils	# some src files
├── setup.py
├── train.py
├── model.py
├── gen_kitti_h5.py
├── kitti_visu_tool.py
├── scene_inference.py
├── README.md
└── data    	# KITTI data directory 
    └── kitti
        ├── training		# 3D object detection benchmark
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne  
        ├── testing  
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne  
        └── raw             # kitti data(raw)
            ├── 2011_09_26	
            |   ├── 2011_09_26_drive_0001_sync
            |   ├── calib_cam_to_cam.txt
            |   ├── calib_imu_to_velo.txt
            |   └── calib_velo_to_cam.txt
             .
             .
             .
```

## Setup - build box overapping

```bash
cd utils
python setup.py build_ext --inplace
```

## Training/Testing dataset preperation

```bash
python gen_kitti_h5.py
```

## Training

```bash
python train.py --log_dir log --batch_size 16 --max_epoch 1 --learning_rate 0.0001 --train_num 3000
```

## Testing

### KITTI 3D Object Benchmark

```bash
python scene_inference.py --model_path ./log2/model0.ckpt
```

### KITTI Raw Data

```bash
python kitti_raw_inference.py --model_path ./log2/model0.ckpt
```


/home/acepc01/Kyungpyo/git/pointnet/sem_seg/log_test7/model40.ckpt
