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
[sdfiana027]$ salloc -p ampere -N 2 --account=mli:brave --gpus-per-node=4  --ntasks-per-node=20 --cpus-per-gpu=5
salloc: No OS_VER constraint specified. Defaulting to OS_VER:8.6 for partition 'ampere'
salloc: Granted job allocation 31909490
salloc: Waiting for resource configuration
salloc: Nodes sdfampere[032-033] are ready for job
```

Verify you can access 40 tasks via srun 

```
[sdfiana027]$ srun hostname | wc -l
40
[sdfiana027]$ srun hostname | sort -u
sdfampere032
sdfampere033
```

and now launch the Resonet workers, which each load the model

```
# this is a persistent application
[sdfiana027]$ $ srun resonet.psana.worker_daemon  --ndev 4
... this process will remain active 
```

Now, on another psana terminal, verify GPUs processes are running on the worker hosts

```
[sdfiana025]$ ssh sdfampere032 "nvidia-smi"
tput: No value for $TERM and no -T specified
tput: No value for $TERM and no -T specified
Thu Jul 16 09:33:21 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:01:00.0 Off |                    0 |
| N/A   31C    P0              57W / 400W |   2694MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          On  | 00000000:41:00.0 Off |                    0 |
| N/A   32C    P0              58W / 400W |   2458MiB / 40960MiB |      3%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          On  | 00000000:81:00.0 Off |                    0 |
| N/A   30C    P0              62W / 400W |   2694MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          On  | 00000000:C1:00.0 Off |                    0 |
| N/A   30C    P0              72W / 400W |   2222MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   3882544      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    0   N/A  N/A   3882548      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    0   N/A  N/A   3882552      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    0   N/A  N/A   3882556      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    0   N/A  N/A   3882560      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    1   N/A  N/A   3882545      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    1   N/A  N/A   3882549      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    1   N/A  N/A   3882553      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    1   N/A  N/A   3882557      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    1   N/A  N/A   3882561      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    2   N/A  N/A   3882546      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    2   N/A  N/A   3882550      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    2   N/A  N/A   3882554      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    2   N/A  N/A   3882558      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    2   N/A  N/A   3882562      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
|    3   N/A  N/A   3882547      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    3   N/A  N/A   3882551      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    3   N/A  N/A   3882555      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    3   N/A  N/A   3882559      C   ...mforge/envs/resonet2/bin/python3.11      414MiB |
|    3   N/A  N/A   3882563      C   ...mforge/envs/resonet2/bin/python3.11      532MiB |
+---------------------------------------------------------------------------------------+
```

Looks good, 20 procs running across 4 GPUs on sdfampere032 node. Verify the same for the other host(s).  
Now, launch the master distrbutor, which sends event codes to the workers...

```
[sdfiana025]$ resonet.psana.master_distributor  --hosts sdfampere032 sdfampere033 --nwork-per-host 20
Connecting to persistent GPU workers...
 -> Connected to worker: tcp://sdfampere032:5550
 -> Connected to worker: tcp://sdfampere032:5551
 -> Connected to worker: tcp://sdfampere032:5552
 -> Connected to worker: tcp://sdfampere032:5553
 -> Connected to worker: tcp://sdfampere032:5554
 -> Connected to worker: tcp://sdfampere032:5555
 -> Connected to worker: tcp://sdfampere032:5556
 -> Connected to worker: tcp://sdfampere032:5557
 -> Connected to worker: tcp://sdfampere032:5558
 -> Connected to worker: tcp://sdfampere032:5559
 -> Connected to worker: tcp://sdfampere032:5560
 -> Connected to worker: tcp://sdfampere032:5561
 -> Connected to worker: tcp://sdfampere032:5562
 -> Connected to worker: tcp://sdfampere032:5563
 -> Connected to worker: tcp://sdfampere032:5564
 -> Connected to worker: tcp://sdfampere032:5565
 -> Connected to worker: tcp://sdfampere032:5566
 -> Connected to worker: tcp://sdfampere032:5567
 -> Connected to worker: tcp://sdfampere032:5568
 -> Connected to worker: tcp://sdfampere032:5569
 -> Connected to worker: tcp://sdfampere033:5550
 -> Connected to worker: tcp://sdfampere033:5551
 -> Connected to worker: tcp://sdfampere033:5552
 -> Connected to worker: tcp://sdfampere033:5553
 -> Connected to worker: tcp://sdfampere033:5554
 -> Connected to worker: tcp://sdfampere033:5555
 -> Connected to worker: tcp://sdfampere033:5556
 -> Connected to worker: tcp://sdfampere033:5557
 -> Connected to worker: tcp://sdfampere033:5558
 -> Connected to worker: tcp://sdfampere033:5559
 -> Connected to worker: tcp://sdfampere033:5560
 -> Connected to worker: tcp://sdfampere033:5561
 -> Connected to worker: tcp://sdfampere033:5562
 -> Connected to worker: tcp://sdfampere033:5563
 -> Connected to worker: tcp://sdfampere033:5564
 -> Connected to worker: tcp://sdfampere033:5565
 -> Connected to worker: tcp://sdfampere033:5566
 -> Connected to worker: tcp://sdfampere033:5567
 -> Connected to worker: tcp://sdfampere033:5568
 -> Connected to worker: tcp://sdfampere033:5569

Initialized 40 workers. Ready to process events.
Progress: 50/500 events processed. (3643.0 ev/sec)
Progress: 100/500 events processed. (6978.1 ev/sec)
Progress: 150/500 events processed. (9980.6 ev/sec)
Progress: 200/500 events processed. (12825.6 ev/sec)
Progress: 250/500 events processed. (15523.2 ev/sec)
Progress: 300/500 events processed. (18064.1 ev/sec)
Progress: 350/500 events processed. (20317.3 ev/sec)
Progress: 400/500 events processed. (22461.6 ev/sec)
Progress: 450/500 events processed. (24419.6 ev/sec)
Progress: 500/500 events processed. (26607.5 ev/sec)
Closing connections.


```

