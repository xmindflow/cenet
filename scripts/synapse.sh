#!/bin/bash -l
# This script is used to run Synapse dataset training or testing with CENet.


CENET_DIR="path/to/cenet" # Update this path to your CENet directory (this repository)
DATA_DIR="/path/to/Synapse" # Update this path to your ACDC dataset directory
PT_DIR="${CENET_DIR}/pretrained_pth" # Path to the pre-trained encoder weights (PVT) [for training only]
RESULTS_DIR="${CENET_DIR}/results/cenet/synapse" # Directory to save results
EVAL_PT="${CENET_DIR}/weights/synapse_2418962.pth" # Path to the evaluation weights (if needed). You must uncomment the --checkpoint argument in the <TEST> section
EVAL_PT_ORG="${CENET_DIR}/weights/synapse_cenet_org_2352272.pth" # Path to the evaluation weights for the [original before cleaning and refactoring] CENet. You must uncomment the --checkpoint argument in the <TEST_ORG> section


# Define the mode to execute
MODE=$1
if [ -z "$MODE" ]; then
  echo "Error: No mode specified."
  echo "Valid modes are: TRAIN, TEST, TEST_ORG"
  exit 1
fi

echo "MODE: $MODE"
nvidia-smi

# Check if the CENet directory exists
if [ ! -d "${CENET_DIR}" ]; then
  echo "CENet directory not found: ${CENET_DIR}"
  echo "Please clone the CENet repository to ${CENET_DIR}."
  exit 1
fi
# Check if the data directory exists
if [ ! -d "${DATA_DIR}" ]; then
  echo "Data directory not found: ${DATA_DIR}"
  echo "Please ensure the Synapse dataset is available at ${DATA_DIR}."
  exit 1
fi

base_src_dir="${CENET_DIR}/src"
train_data_dir=${DATA_DIR}/train_npz
volume_path=${DATA_DIR}/test_vol_h5
list_dir=${DATA_DIR}/lists/lists_Synapse

max_epochs=250
batch_size=8
base_lr=0.015
tag="CENet_Synapse_CHECK"

cd ${base_src_dir}
# python -m pip install --upgrade pip
# pip install -r requirements.txt

case "$MODE" in
  TRAIN)
    echo "Running SYNAPSE [TRAIN]..."
    python main_synapse.py \
      --model_version "cenet" \
      --tag ${tag} \
      --max_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --eval_interval 20 \
      --root_dir $train_data_dir \
      --volume_path $volume_path \
      --list_dir ${list_dir} \
      --output_dir ${RESULTS_DIR} \
      --save_path ${RESULTS_DIR} \
      --fast_data \
      --num_workers 11 \
      --loss_type 'boundary' \
      --optimizer 'SGD' \
      --loss_weights '1' \
      --scheduler 'poly' \
      --base_lr ${base_lr} \
      --encoder 'pvt_v2_b2' \
      --encoder_ptdir ${PT_DIR} \
      --scale_factors '0.8,0.4' \
      --num_heads '16,8,8' \
      --skip_mode 'cat' \
      --dec_up_block 'eucb' \
      --out_merge_mode "cat" \
      --out_up_block "upcn" \
      --out_up_ks 3 \
      --amp \
      # --compile \
      # --no_ptenc

    ;;
  TEST)
    echo "Running SYNAPSE-3D [TEST]..."
    python main_synapse.py \
      --model_version "cenet" \
      --eval \
      --tag ${tag} \
      --max_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --root_dir $train_data_dir \
      --volume_path $volume_path \
      --list_dir ${list_dir} \
      --output_dir ${RESULTS_DIR} \
      --save_path ${RESULTS_DIR} \
      --fast_data \
      --num_workers 11 \
      --base_lr ${base_lr} \
      --encoder 'pvt_v2_b2' \
      --scale_factors '1.0,0.75,0.5' \
      --num_heads '16,8,8' \
      --skip_mode 'cat' \
      --dec_up_block 'eucb' \
      --out_merge_mode "cat" \
      --out_up_block "upcn" \
      --out_up_ks 3 \
      # --checkpoint ${EVAL_PT} \
      # --amp \
      # --compile \

    ;;
  TEST_ORG)
    echo "Running SYNAPSE-3D [TEST_ORG]..."
    python main_synapse.py \
      --model_version "cenet_org" \
      --eval \
      --tag ${tag} \
      --max_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --root_dir $train_data_dir \
      --volume_path $volume_path \
      --list_dir $list_dir \
      --output_dir "${RESULTS_DIR}_org" \
      --save_path "${RESULTS_DIR}_org" \
      --fast_data \
      --num_workers 11 \
      --base_lr ${base_lr} \
      --encoder 'pvt_v2_b2' \
      --scale_factors '0.8,0.4' \
      --num_heads '16,8,8' \
      --skip_mode 'cat' \
      --dec_up_block 'eucb' \
      --out_merge_mode "cat" \
      --out_up_block "upcn" \
      --out_up_ks 3 \
      # --checkpoint ${EVAL_PT_ORG} \
      # --amp \
      # --compile \

    ;;
  *)
    echo "Invalid mode specified: $MODE"
    echo "Valid mode are: TRAIN, TEST, TEST_ORG"
    exit 1
    ;;
esac
