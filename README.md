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

Then, install cuda, version 12+ (older versions work, but require different conda environments).

With the conda env and CUDA installed, one can build CCTBX with CUDA support. Verify CUDA is indeed installed correctly and then set the standard CUDA environment
 
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
```

Check you can see the GPU(s) using ```nvidia-smi```. Then, activate the conda env:


```bash
conda activate py39
```

Now create a subfolder for building CCTBX and get the CCTBX bootstrap script:

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
libtbx.python -m pip install torch torchvision torchmetrics jupyter ipython
libtbx.refresh

# Note, migration to PyPi is in progress, however for now one will need to build manually
# libtbx.python -m pip install resonet

git clone https://github.com/ssrl-px/resonet.git
cd resonet
libtbx.python -m build
libtbx.python -m pip install dist/resonet-0.1.tar.gz
libtbx.python patch_shebangs.py  # only necessary is using CCTBX build for simulation functionality
```

## Synthesize training data
With the above environment, you should now download the simulation meta data:

```
wget https://smb.slac.stanford.edu/~dermen/for_tutorial.tar.gz
tar -xzvf for_tutorial.tar.gz
```

and then specify its location using the environment variable `RESONET_SIMDATA`:

```
export RESONET_SIMDATA=/path/to/for_tutorial/diffraction_ai_sims_data/
```

Now, siumulations can be run using:

```
resonet-simulate test_shots  --nshot 10 --pdbName $RESONET_SIMDATA/pdbs/3nxs
```

Note, if MPI is installed the above script can be envoked with mpirun (or srun if using SLURM):

```
mpirun -n 6 resonet-simulate test_shots --nshots 10000 --pdbName $RESONET_SIMDATA/pdbs/3nxs
```

The above mpirun command took 8.03 hours using an Nvidia A100 and an Intel(R) Xeon(R) Gold 6126. In the future, this runtime will be decreased significantly after the background routines are ported to GPU. In parallel mode, each MPI-rank will write a unique output file, and these can be combined using the `merge_h5s.py` script:

```
resonet-mergefiles test_shots test_shots/master.h5
```

This creates a `master.h5` which can be passed directly to the training script.


## Train the model
The script `net.py` has a lot of functionality, but is still under heavy development. Use it as follows:

```
resonet-train  100 test_shots/master.h5  test_opt --labelSel one_over_reso --useGeom  --testRange 0 1000 --trainRange 1000 10000 --bs 64 --lr 0.01
```

The first argument is the number of training epochs. The second argument is the input, and the third argument is the output folder where results and a log file will be written. Note, the first epoch is usually slower than the subsequent epochs.

## Check the results
One can plot the training progress:

```
resonet-plotloss test_opt/train.log
```

## Inference

Lets assume a model has been trained, and it is time to test its predictions for some images. 

There are two simple scripts (`resonet-imgeater`, `resonet-imgfeeder`) supplied in the repository for testing models. These scripts use [Pyro4](https://pyro4.readthedocs.io/en/stable/) and MPI to for inter process communication.  By exploring those scripts, one can hopefully design even more robust resonet frameworks. 

First, one should launch `resonet-imgeater`. The *eater* can be launched as a single process, or with mpirun as multiple processes:

```
libtbx.python -m Pyro4.naming &
mpirun -n 8 resonet-imgeater ./path/to/nety_ep100.nn res50 --gpu &
```

Note, both of the above jobs were launched in the background. Now, the *eater* process will remain active, while we use `resonet-imgfeeder` to send it images, in this case as [python glob strings](https://docs.python.org/3/library/glob.html):

```
resonet-imgfeeder "/path/to/some/images/*cbf" 8
```

where the second argument simply specifies the number of processes launched with `resonet-imgeater`. The *eater* will then write the inference results to STDOUT. Note, all images in the GLOB will be processed!
