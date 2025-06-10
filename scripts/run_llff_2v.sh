#!/bin/bash

# Set CUDA device
DEVICE=0

# Set root directory
ROOT_DIR="./data/nerf_llff_data"

# Set base directory
BASE_DIR="./log/llff2v"

# Set default config
CONFIG="configs/llff_default_2v.txt"

# Define experiments and their frame IDs
declare -a experiments=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
declare -a train_frame_nums=("6 13" "11 22" "13 28" "20 42" "9 17" "7 17" "13 27" "18 37")
declare -a test_frame_nums=("0 8 16" "0 8 16 24 32" "0 8 16 24 32 40" "0 8 16 24 32 40 48 56" "0 8 16 24" "0 8 16 24" "0 8 16 24 32 40" "0 8 16 24 32 40 48")

# Run the training scripts
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    train_frame_num="${train_frame_nums[$i]}"
    test_frame_num="${test_frame_nums[$i]}"
    echo "Running training for $exp on device $DEVICE"
    CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config $CONFIG --datadir $ROOT_DIR/$exp --expname $exp --train_frame_num $train_frame_num --test_frame_num $test_frame_num --basedir $BASE_DIR
    CUDA_VISIBLE_DEVICES=$DEVICE python extra/compute_metrics.py --render_dir $BASE_DIR/$exp/imgs_test_all --gt_dir $ROOT_DIR/$exp/image
    echo "Training for $exp completed"
done

cp $CONFIG $BASE_DIR/

python extra/read_metrics.py --expname test_2v --log_dir log/llff2v --weight 3 5 6 8 4 4 6 7