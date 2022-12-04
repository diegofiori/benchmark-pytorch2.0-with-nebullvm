# benchmark-pytorch2.0-with-nebullvm
Project implementing a benchmark between pytorch 2.0 and the main compilers available on the market. The OpenSource library [nebullvm](https://github.com/nebuly-ai/nebullvm) has been used for collecting data about the compilers running times. 

## Setup
For running the experiment it is necessary to create two distinct environments. The first one containing pytorch 2.0 and the second one for the compilers (which obviously are not compatible yet with the future version of pytorch). We use conda for managing the environments
```bash
conda create -n pytorch_2_env python=3.8 && conda activate pytorch_2_env
# on gpu
pip3 install numpy --pre torch[dynamo] torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
# on cpu 
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
For the environment with the compilers we can lavarage the nebullvm auto-installer
```bash
conda create -n nebullvm_env python=3.8 && conda activate nebullvm_env
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install nebullvm
python -m nebullvm.installers.auto_installer --frameworks torch onnx --compilers all
```
Note that pytorch must be installed before nebullvm. I suggest to use pytorch 1.12.1 since it is the one with the largest compatibility with nebullvm-supported compilers. Now we are ready for running our benchmark.
You can simply run on the `pytorch_2_env`
```bash
python main.py --batch_size 1 
```
for running the test on the torch2.0 compiler, with the selected batch size. Torch 2.0 can also be run in fp16 precision. You can test it adding the `--half` flag:
```bash
python main.py --batch_size 1 --half

On the other hand, you can run on the `nebullvm_env`
```bash
python main.py --batch_size 1 --nebullvm
```
for collecting the data about the compilers. Note that `pytorch 2.0` test will save the result in a json file named `results_torch_{batch_size}_dafault.json`, while nebullvm will store the best compiler info in a json named `results_nebullvm_{batch_size}.json` and a detailed summary of the performance of each compiler in another json file named `latencies_ResNet_*.json`.

