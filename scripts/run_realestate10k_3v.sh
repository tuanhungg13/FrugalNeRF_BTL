#!/bin/bash

# Set CUDA device
DEVICE=0

# Set root directory
ROOT_DIR="./data/realestate10k"

# Set base directory
BASE_DIR="./log/realestate10k3v"

# Set default config
CONFIG="configs/realestate10k_defalt_3v.txt"

# Define experiments and their frame IDs 21 31 34 38 40 41 45 55 63 82 103 114
declare -a experiments=("00000" "00001" "00003" "00004" "00006")

# Run the training scripts
for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    train_frame_num="${train_frame_nums}"
    test_frame_num="${test_frame_nums}"
    vis_every=20000
    echo "Running training for $exp on device $DEVICE"
    CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config $CONFIG --datadir $ROOT_DIR/$exp --expname $exp --basedir $BASE_DIR --vis_every $vis_every
    CUDA_VISIBLE_DEVICES=$DEVICE python extra/compute_metrics.py --render_dir $BASE_DIR/$exp/imgs_test_all --gt_dir $ROOT_DIR/$exp/image --mask_dir $ROOT_DIR/$exp/mask
    echo "Training for $exp completed"
done

cp $CONFIG $BASE_DIR

python extra/read_metrics.py --expname test_3v --log_dir log/realestate10k3v