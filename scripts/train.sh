CONFIG=$1
GPU=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4

python src/train.py -c $CONFIG
