# README

## Environment
* Ubuntu14.04LTS
* Python3.6
* Spyder3
* Tensorflow-gpu 1.5.0
## 
├── build   <-- Cython build file
├── doc
├── utils   <-- some src files
├── setup.py
├── train.py
├── model.py
├── gen_kitti_h5.py
├── kitti_visu_tool.py
├── scene_inference.py
├── README.md
└── data    <-- KITTI data directory 
    └── kitti
        ├── training   <-- training data
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne  
        ├── testing  
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne  
        └── raw
            ├── 2011_09_26 <-- kitti data(raw)
             .
             .
             .
       
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

```bash
python scene_inference.py --model_path ./log2/model0.ckpt
```

/home/acepc01/Kyungpyo/git/pointnet/sem_seg/log_test7/model40.ckpt
