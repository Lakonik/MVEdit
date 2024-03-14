# MVEdit

Official PyTorch implementation of the paper:

**Generic 3D Diffusion Adapter Using Controlled Multi-View Editing**
<br>
[Hansheng Chen](https://lakonik.github.io/)<sup>1</sup>, 
[Ruoxi Shi](https://rshi.top/)<sup>2</sup>, 
[Yulin Liu](https://liuyulinn.github.io/)<sup>2</sup>, 
[Bokui Shen](https://cs.stanford.edu/people/bshen88/)<sup>3</sup>,
[Jiayuan Gu](https://pages.ucsd.edu/~ztu/)<sup>2</sup>, 
[Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>, 
[Hao Su](https://cseweb.ucsd.edu/~haosu/)<sup>2</sup>, 
[Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)<sup>1</sup><br>
<sup>1</sup>Stanford University, <sup>2</sup>UCSD, <sup>3</sup>Apparate Labs
<br>

[[project page](https://lakonik.github.io/mvedit)] [[Web UI](http://34.80.119.68:7860/)] [[paper](https://arxiv.org/abs/2403.12032)]

## Todos

This codebase is currently incomplete and under refactoring. We aim to release the complete codebase in two weeks. The following are the todos:

- [ ] Add Zero123++ v1.2 to the Web UI
- [ ] Release the complete codebase, including the Web UI that can be deployed on your own machine

## Installation

### Prerequisites

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 20 & 22)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 12
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 2.1
- FFmpeg, x264 (optional, for exporting videos)

Other dependencies can be installed via `pip install -r requirements.txt`. 

An example of commands for installing the Python packages is shown below (you may change the CUDA version yourself):

```bash
# Export the PATH of CUDA toolkit
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n mvedit python=3.10
conda activate mvedit

# Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone this repo and install other dependencies
git clone <this repo> && cd <repo folder> && git checkout mvedit
pip install -r requirements.txt

# Install FFmpeg (optional)
conda remove ffmpeg  # Remove the old version
conda install -c conda-forge ffmpeg x264
```

Optionally, you can install [xFormers](https://github.com/facebookresearch/xformers) for efficnt attention (for PyTorch 1.x). Also, this codebase should be able to work on Windows systems, but it has not been tested extensively.
