#!/bin/bash -l
# This script is used to run Skin datasets [HAM10K, PH2] training or testing with CENet.


CENET_DIR="path/to/cenet" # Update this path to your CENet directory (this repository)
DATA_DIR="/path/to/Skin" # Update this path to your Skin datasets directory
PT_DIR="${CENET_DIR}/pretrained_pth" # Path to the pre-trained encoder weights (PVT) [for training only]
RESULTS_DIR="${CENET_DIR}/results/cenet/skin" # Directory to save results
EVAL_PT_HAM="${CENET_DIR}/weights/ham10k_2418602.pth" # Path to the evaluation weights for HAM10K (if needed). You must uncomment the --checkpoint argument in the <TEST> section
EVAL_PT_PH2="${CENET_DIR}/weights/ph2_2418571.pth" # Path to the evaluation weights for PH2 (if needed). You must uncomment the --checkpoint argument in the <TEST> section


# Define the section to execute
DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Error: No dataset specified."
  echo "Valid datasets are: HAM, PH2"
  exit 1
fi
MODE=$2
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
  echo "Please ensure the Skin datasets are available at ${DATA_DIR}."
  exit 1
fi


base_src_dir="${CENET_DIR}/src"
max_epochs=5
batch_size=8
base_lr=0.01

case "$DATASET" in
    HAM)
        data_dir="${DATA_DIR}/HAM10000"
        save_path="${RESULTS_DIR}/ham"
        EVAL_PT="${EVAL_PT_HAM}"
        loss_weights="0.5,0.5"
        ;;
    PH2)
        data_dir="${DATA_DIR}/PH2"
        save_path="${RESULTS_DIR}/ph2"
        EVAL_PT="${EVAL_PT_PH2}"
        loss_weights="0.7,0.3"
        ;;
    *)
        echo "Invalid dataset specified: $DATASET"
        echo "Valid datasets are: HAM, PH2"
        exit 1
        ;;
esac

tag="CENet_Skin_${DATASET}_CHECK"

cd ${base_src_dir}
# python -m pip install --upgrade pip
# pip install -r requirements.txt

case "$MODE" in
    TRAIN)
        echo "Running ${DATASET} [TRAIN]..."
        python main_skin.py \
            --tag ${tag} \
            --max_epochs ${max_epochs} \
            --data_dir ${data_dir} \
            --fast_data \
            --save_path ${save_path} \
            --batch_size ${batch_size} \
            --num_workers 11 \
            --optimizer 'SGD' \
            --loss_type "dice,ce" \
            --loss_weights ${loss_weights} \
            --scheduler 'poly' \
            --base_lr ${base_lr} \
            --encoder 'pvt_v2_b2' \
            --encoder_ptdir ${PT_DIR} \
            --scale_factors '1.0,0.75,0.5' \
            --num_heads '2,2,2' \
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
        echo "Running ${DATASET} [TEST]..."
        python main_skin.py \
            --eval \
            --tag ${tag} \
            --max_epochs ${max_epochs} \
            --batch_size ${batch_size} \
            --data_dir ${data_dir} \
            --fast_data \
            --save_path ${save_path} \
            --num_workers 11 \
            --base_lr ${base_lr} \
            --encoder 'pvt_v2_b2' \
            --scale_factors '1.0,0.75,0.5' \
            --num_heads '2,2,2' \
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
