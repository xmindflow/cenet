#!/bin/bash -l
# This script is used to run ACDC-2D training or testing with CENet.

CENET_DIR="path/to/cenet" # Update this path to your CENet directory (this repository)
DATA_DIR="/path/to/ACDC_2D" # Update this path to your ACDC dataset directory
RESULTS_DIR="${CENET_DIR}/results/cenet/acdc" # Directory to save results
PT_DIR="${CENET_DIR}/pretrained_pth" # Path to the pre-trained encoder weights (PVT) [for training only]
EVAL_PT="${CENET_DIR}/weights/acdc_2418732.pth" # Path to the evaluation weights (if needed). You must uncomment the --checkpoint argument in the test section


# Define the mode to execute
MODE=$1
if [ -z "$MODE" ]; then
  echo "Error: No mode specified."
  echo "Valid modes are: TRAIN, TEST"
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
  echo "Please ensure the ACDC dataset is available at ${DATA_DIR}."
  exit 1
fi


base_src_dir="${CENET_DIR}/src"
root_dir="${DATA_DIR}"
list_dir="${DATA_DIR}/list_ACDC"
volume_path="${DATA_DIR}/test"

batch_size=8
base_lr=0.01
max_epochs=120
tag="CENet_ACDC_CHECK"

cd ${base_src_dir}
# python -m pip install --upgrade pip
# pip install -r requirements.txt

case "$MODE" in
  TRAIN)
    echo "Running ACDC [TRAIN]..."
    python main_acdc.py \
      --tag ${tag} \
      --max_epochs ${max_epochs} \
      --root_dir ${root_dir} \
      --list_dir ${list_dir} \
      --volume_path ${volume_path} \
      --fast_data \
      --save_path ${RESULTS_DIR} \
      --batch_size ${batch_size} \
      --num_workers 11 \
      --loss_type 'boundary' \
      --optimizer 'SGD' \
      --loss_weights '1' \
      --scheduler 'poly' \
      --base_lr ${base_lr} \
      --encoder 'pvt_v2_b2' \
      --encoder_ptdir ${PT_DIR} \
      --scale_factors '1.0,0.5' \
      --num_heads '4,4,4' \
      --skip_mode 'cat' \
      --dec_up_block 'eucb' \
      --out_merge_mode "cat" \
      --out_up_block "upcn" \
      --out_up_ks 3 \
      --amp \
      # --no_ptenc \
      # --compile \

    ;;
  TEST)
    echo "Running ACDC-2D [TEST]..."
    python main_acdc.py \
      --eval \
      --tag ${tag} \
      --max_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --root_dir ${root_dir} \
      --list_dir ${list_dir} \
      --volume_path ${volume_path} \
      --fast_data \
      --save_path ${RESULTS_DIR} \
      --num_workers 11 \
      --base_lr ${base_lr} \
      --encoder 'pvt_v2_b2' \
      --scale_factors '1.0,0.5' \
      --num_heads '4,4,4' \
      --skip_mode 'cat' \
      --dec_up_block 'eucb' \
      --out_merge_mode "cat" \
      --out_up_block "upcn" \
      --out_up_ks 3 \
      # --checkpoint ${EVAL_PT} \
      # --amp \
      # --compile \

    ;;
  *)
    echo "Invalid mode specified: $MODE"
    echo "Valid mode are: TRAIN, TEST"
    exit 1
    ;;
esac
