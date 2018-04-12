
```bash
cd utils
python setup.py build_ext --inplace
```

```bash
python gen_kitti_h5.py
```

```bash
python train.py --log_dir log --batch_size 1 --max_epoch 50 --learning_rate 0.0001 --train_num 25000
```

```bash
python scene_inference.py --model_path ./log2/model0.ckpt
```

/home/acepc01/Kyungpyo/git/pointnet/sem_seg/log_test7/model40.ckpt
