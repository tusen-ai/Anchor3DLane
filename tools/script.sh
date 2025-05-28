export PDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PDIR
export PYTHONPATH=$PYTHONPATH:$PDIR/gen-efficientnet-pytorch

python tools/test.py configs_v2/openlane/anchor3dlane++_r50x2.py \
output/openlane_anchor3dlane++_r50x2.pth --show-dir output/openlane_anchor3dlane++_r50x2