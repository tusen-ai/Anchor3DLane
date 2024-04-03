## Installation process 


1. First, create and activate the environment:
```
conda create -n lane3d python=3.8 -y
conda activate lane3d
```

2. Use the following commands to install the correct version of pytorch:

```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

3. MMCV installation (to avoid **ModuleNotFoundError: No module named 'mmcv._ext'** error specifically install the version for your pytorch and cuda ):

```
pip install -U openmim
mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
```

4. The requirements.txt file had a few changes to fix the dependancy issues, use the requirements.txt file from this repository:
```
pip install -r requirements.txt
```

5. Install mmseg from the official rep:
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

6. Download jarvis wheel from [ONCE-3dlane](https://github.com/once-3dlanes/once_3dlanes_benchmark/tree/master/wheels) and  run the following command:

```
pip install ./wheels/jarvis-2021.4.2-py2.py3-none-any.whl
```

7. Finally, run the setup command:
```
python setup.py develop
```

## Troubleshooting

8. There are several additional libraries to install:

```
pip install mmengine
pip install ftfy
pip install regex
```

9. To solve the **TypeError: FormatCode() got an unexpected keyword argument 'verify'** error, run
```
pip install yapf==0.40.1
```