CONFIG=$1
GPUS=$2
NNODES=${NNODES:-2}
NODE_RANK=$3
PORT=${PORT:-12355}
MASTER_ADDR=${MASTER_ADDR:-"10.12.1.84"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_dist.py \
    $CONFIG \
    --launcher pytorch ${@:4}
