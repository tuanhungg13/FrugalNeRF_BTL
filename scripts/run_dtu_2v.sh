#!/bin/bash

# Set CUDA device
DEVICE=0

# Set root directory
ROOT_DIR="./data/rs_dtu_4/DTU"

# Set base directory
BASE_DIR="./log/dtu2v"

# Set default config
CONFIG="configs/dtu_default_2v.txt"

# Define experiments and their frame IDs 21 31 34 38 40 41 45 55 63 82 103 114
declare -a experiments=("scan21" "scan31" "scan34" "scan38" "scan40" "scan41" "scan45" "scan55" "scan63" "scan82" "scan103" "scan114")
declare -a train_frame_nums="22 25"
declare -a test_frame_nums="1 2 3 4 5 6 7 9 10 11 12 14 15 16 17 18 19 20 21 23 24 26 27 29 30 31 32 33 34 35 36 37 38 39 41 42 43 45 46 47"

# Run the training scripts
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    train_frame_num="${train_frame_nums}"
    test_frame_num="${test_frame_nums}"
    vis_every=20000
    echo "Running training for $exp on device $DEVICE"
    CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config $CONFIG --datadir $ROOT_DIR/$exp --expname $exp --train_frame_num $train_frame_num --test_frame_num $test_frame_num --basedir $BASE_DIR --vis_every $vis_every
    CUDA_VISIBLE_DEVICES=$DEVICE python extra/compute_metrics.py --render_dir $BASE_DIR/$exp/imgs_test_all --gt_dir $ROOT_DIR/$exp/image --mask_dir $ROOT_DIR/$exp/mask
    echo "Training for $exp completed"
done

cp $CONFIG $BASE_DIR

python extra/read_metrics.py --expname test_2v --log_dir log/dtu2v