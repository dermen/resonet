NOTE: this shows how to test the framework - currently its a template,  we need to add the psana bits and actual resonet bits (most are there), and then ensure GPU workers  and the master can both access the FFB system.

NOTE: the goal is to have the master worker read events as they come in - from the FFB file system, and then distribute events to workers. the master should only read the event index/ run index, and send that info to GPU workers which then need to load the data from those runs (without reading the full XTC of course).. 

The last piece is the control-node application which will receive the results from the resonet workers - simply event code + resolution estimate..

### Test the current version 

Load the env on the psana node 

```
source /sdf/group/lcls/ds/tools/braggsim/simforge/etc/profile.d/conda.sh
conda activate resonet2
```

Allocat a slurm session from which to launch the persistent RESONET  workers

```
[sdfiana027]$ salloc -p ampere -N 1 --account=mli:brave --gpus-per-node=4  --ntasks-per-node=16 --cpus-per-gpu=4
salloc: No OS_VER constraint specified. Defaulting to OS_VER:8.6 for partition 'ampere'
salloc: Granted job allocation 31905826
salloc: Waiting for resource configuration
salloc: Nodes sdfampere011 are ready for job
```

Verify you can access 16 tasks via srun (Note, we want to scale this up to >1 node): 

```
[sdfiana027]$ srun hostname
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
sdfampere011
```

and now launch the Resonet workers, which each load the model

```
# this is a persistent application
[sdfiana027]$ $ srun resonet.psana.worker_daemon  --ndev 4
... this process will remain active 
```

Now, on another psana terminal, verify GPUs processes are running on the worker host (in this example sdfampere0011):

```
[sdfiana025]$ ssh sdfampere011 "nvidia-smi"
$ ssh sdfampere011 "nvidia-smi"
tput: No value for $TERM and no -T specified
tput: No value for $TERM and no -T specified
Thu Jul 16 09:02:09 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:01:00.0 Off |                    0 |
| N/A   31C    P0              54W / 400W |   2156MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          On  | 00000000:41:00.0 Off |                    0 |
| N/A   33C    P0              58W / 400W |   1920MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          On  | 00000000:81:00.0 Off |                    0 |
| N/A   32C    P0              59W / 400W |   2156MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          On  | 00000000:C1:00.0 Off |                    0 |
| N/A   30C    P0              58W / 400W |   1802MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1220384      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    0   N/A  N/A   1220389      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    0   N/A  N/A   1220394      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    0   N/A  N/A   1220398      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    1   N/A  N/A   1220385      C   ...m/simforge/envs/resonet2/bin/python      414MiB |
|    1   N/A  N/A   1220390      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    1   N/A  N/A   1220395      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    1   N/A  N/A   1220399      C   ...m/simforge/envs/resonet2/bin/python      414MiB |
|    2   N/A  N/A   1220386      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    2   N/A  N/A   1220392      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    2   N/A  N/A   1220396      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    2   N/A  N/A   1220400      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    3   N/A  N/A   1220387      C   ...m/simforge/envs/resonet2/bin/python      414MiB |
|    3   N/A  N/A   1220393      C   ...m/simforge/envs/resonet2/bin/python      414MiB |
|    3   N/A  N/A   1220397      C   ...m/simforge/envs/resonet2/bin/python      532MiB |
|    3   N/A  N/A   1220401      C   ...m/simforge/envs/resonet2/bin/python      414MiB |
+---------------------------------------------------------------------------------------+
```

Looks good, 16 procs running across 4 GPUs on 1 node.. Now, launch the master distrbutor, which sends event codes to the workers...

```
[sdfiana027]$ resonet.psana.master_distributor  --hosts sdfampere011 --nwork 8
Connecting to persistent GPU workers...
 -> Connected to worker: tcp://sdfampere011:5550
 -> Connected to worker: tcp://sdfampere011:5551
 -> Connected to worker: tcp://sdfampere011:5552
 -> Connected to worker: tcp://sdfampere011:5553
 -> Connected to worker: tcp://sdfampere011:5554
 -> Connected to worker: tcp://sdfampere011:5555
 -> Connected to worker: tcp://sdfampere011:5556
 -> Connected to worker: tcp://sdfampere011:5557

Initialized 8 workers. Ready to process events.
Progress: 50/500 events processed. (10770.6 ev/sec)
Progress: 100/500 events processed. (17550.9 ev/sec)
Progress: 150/500 events processed. (22590.5 ev/sec)
Progress: 200/500 events processed. (25850.9 ev/sec)
Progress: 250/500 events processed. (28656.7 ev/sec)
Progress: 300/500 events processed. (30853.3 ev/sec)
Progress: 350/500 events processed. (32677.6 ev/sec)
Progress: 400/500 events processed. (34395.0 ev/sec)
Progress: 450/500 events processed. (35865.8 ev/sec)
Progress: 500/500 events processed. (37202.7 ev/sec)
Closing connections.

```

