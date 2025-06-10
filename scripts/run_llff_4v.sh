#!/bin/bash

# Set CUDA device
DEVICE=0

# Set root directory
ROOT_DIR="./data/nerf_llff_data"

# Set base directory
BASE_DIR="./log/llff4v"

# Set default config
CONFIG="configs/llff_default_4v.txt"

# Define experiments and their frame IDs
declare -a experiments=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
declare -a train_frame_nums=("4 7 12 15" "6 13 20 27" "7 17 25 34" "12 25 37 50" "5 10 15 20" "4 10 14 20" "7 15 25 33" "11 22 33 44")
declare -a test_frame_nums=("0 8 16" "0 8 16 24 32" "0 8 16 24 32 40" "0 8 16 24 32 40 48 56" "0 8 16 24" "0 8 16 24" "0 8 16 24 32 40" "0 8 16 24 32 40 48")

# Run the training scripts
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    train_frame_num="${train_frame_nums[$i]}"
    test_frame_num="${test_frame_nums[$i]}"
    echo "Running training for $exp on device $DEVICE"
    CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config $CONFIG --datadir $ROOT_DIR/$exp --expname $exp --train_frame_num $train_frame_num --test_frame_num $test_frame_num --basedir $BASE_DIR
    echo "Training for $exp completed"
done

# python extra/read_metrics.py --expname test_4v --log_dir log/llff4v --weight 3 5 6 8 4 4 6 7