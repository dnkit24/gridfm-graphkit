#!/bin/bash

#SBATCH --job-name=graphkit_train
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00    # be careful to set enough runtime
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:16g
#SBATCH --tmp=10g
#SBATCH --output=/cluster/home/dnelischer/Untitled/gridfm-graphkit/logs/train/graphkit_train_%j.out
#SBATCH --error=/cluster/home/dnelischer/Untitled/gridfm-graphkit/logs/train/graphkit_train_%j.err


BASE_PATH="/cluster/home/dnelischer/Untitled/gridfm-graphkit"
DATA_PATH="${BASE_PATH}/examples/data"
CONFIG_PATH="${BASE_PATH}/config/train.yaml"
EXP_NAME="run_1"


mkdir -p ${BASE_PATH}/results/${EXP_NAME}

gridfm_graphkit train --config ${CONFIG_PATH} \
    --exp ${EXP_NAME} \
    --data_path ${DATA_PATH}

echo "=== Training Complete ==="