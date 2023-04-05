export PYTHONPATH=$PYTHONPATH:/mnt/weka/scratch/shaofei.huang/Code/Anchor3dlane
export PYTHONPATH=$PYTHONPATH:/mnt/weka/scratch/shaofei.huang/Code/Anchor3dlane/gen-efficientnet-pytorch

CUDA_VISIBLE_DEVICES=0 python tools/test.py output/openlane/check/temporal_2stage/train_iter.py \
output/openlane/check/temporal_2stage/iter_60000.pth --show-dir output/once/check_train/baseline_2stage/test_60000