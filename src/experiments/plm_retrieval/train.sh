#!/bin/bash

WORKING_DIR=/data2/private/liqingquan/mjjd_simcase/plm_retrieval

GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node=$GPUS_PER_NODE"

MAIN_RANK=0
WORKERS=32
MODEL_PATH="thunlp/Lawformer"
BATCH_SIZE=8
LR=1e-5
EPOCHS=50
OUTPUT_PATH="${WORKING_DIR}/checkpoints_more/"
 
OPTS=""
OPTS+=" --main_rank=${MAIN_RANK}"
OPTS+=" --workers=${WORKERS}"
OPTS+=" --model_path=${MODEL_PATH}"
OPTS+=" --batch_size=${BATCH_SIZE}"
OPTS+=" --lr=${LR}"
OPTS+=" --epochs=${EPOCHS}"
OPTS+=" --output_path=${OUTPUT_PATH}"
OPTS+=" --gpu=0,1,2,3,4,5,6,7"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/distributed.py ${OPTS}"
 
${CMD}
