# diffraction AI code

To use, update your Python path to include the path containing `resonet` folder. 

In other words, if you clone `resonet` in `/home/username/something`, such that the path to `resonet` is `/home/username/something/resonet`, then

```
export PYTHONPATH=$PYTHONPATH:/home/username/something
```

# Resonet tutorial

## Install

Note: this install is only necessary if one wishes to synthesize training data.

Create a conda environment for doing simulations:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget https://raw.githubusercontent.com/dials/dials/main/.conda-envs/linux.txt
bash ./Miniconda3-latest-Linux-x86_64.sh -b -u -p $PWD/miniforge
source miniforge/etc/profile.d/conda.sh
conda install -y -c conda-forge mamba
mamba create -y -n py39 --file linux.txt python=3.9
```

Then, install cuda, version 12+ (older version work, but require different conda environments.

With the conda env and CUDA installed, one can build CCTBX with CUDA support. Verify CUDA is indeed installed correctly
 
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
```

Check you can see GPUs using ```nvidia-smi```. Then, activate the conda env:


```bash
conda activate py39
```

Now create a subfolder for building CCTBX and get the bootstrap command:

```bash
mkdir ~/xtal
cd xtal
wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
```

We will first use bootstrap to download all the sources into the `modules` folder:

```bash
python bootstrap.py  hot update --builder=dials --python=39 --use-conda=$CONDA_PREFIX
```

Now, you can run the build step:

```bash
python bootstrap.py build --builder=dials --python=39 --use-conda=$CONDA_PREFIX  \
  --nproc=22 --config-flags="--enable_openmp_if_possible=True" --config-flags="--use_environment_flags" \
  --config-flags="--enable_cuda" --config-flags="--enable_cxx11"

source build/setpaths.sh
```

The build step creates a `build` subfolder that contains a shell script to launch the CCTBX environment. Its now a good idea to create a single startup script to launch CCTBX+CUDA environement at startup:

```bash
# example startup script:

export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
source ~/xtal_reso/build/setpaths.sh
```

Lastly, install pytorch, a few useful tools like ipython, and lastly resonet:

```bash
# after sourcing setpaths.sh, libtbx.python will be in path
libtbx.python -m pip install torch torchvision jupyter ipython
cd ~/xtal/modules
git clone https://github.com/dermen/resonet.git
libtbx.refresh
```



## Synthesize training data
With the above environment, you should now download the simulation meta data:



## Train the model


## Check the results


