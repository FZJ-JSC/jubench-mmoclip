# MMoCLIP

## Purpose

This benchmark runs [OpenCLIP](https://github.com/mlfoundations/open_clip), an open source
implementation of [CLIP](https://openai.com/blog/clip/), which is a multi-modal contrastive
model that can learn image and text representations from a dataset of image-text pairs.

The benchmark trains a model on a synthetic dataset of 3.2M image-text pairs. The goal is to minimize the total training time.

_While this description tries to be agnostic with respect to the benchmarking infrastructure, we consider JUBE as our reference and give examples with it._

## Source

Archive name: mmoclip-bench.tar.gz

The file holds instructions to run the benchmark and a JUBE script. The OpenCLIP library is part of the archive and distributed in the `src/` directory. The version of the included source code is [ee286275](https://github.com/mlfoundations/open_clip/tree/ee286275771f4efccdd5ac6df63ce4233c7d9ce8).

## Building

The MMoCLIP benchmark requires PyTorch (at least 1.9.0), a CUDA toolkit (11.5 or higher), cuDNN (8.3.1 or higher), and NCCL (2.12.7 or higher).

To install the OpenCLIP Python requirements in a virtual environment, you can do the following:

```bash
python3 -m venv env
source env/bin/activate
pip install -U pip
cd src/open_clip
make install-training
```

See `src/open_clip/README.md` for more information.

### JUBE

If JUBE is used, the Python requirements are automatically installed in a virtual environment.

## Execution

### Command Line

An example of a command line execution is provided in the following:

```bash
[mpiexec] python src/open_clip/src/training/main.py --dataset-type synthetic --train-num-samples 3200000 --epochs 1 --batch-size=512 --model ViT-L-14 --dist-url="env://" --name test --logs logs --local-loss --gather-with-grad --grad-checkpointing --log-every-n-steps 1
```

Please make sure to set `MASTER_ADDR` and `MASTER_PORT` accordingly (see https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization).

- It is possible to modify the local batch size (`--batch-size`) depending on the available GPU memory
- `--local-loss` and `--gather-with-grad` are required; `--grad-checkpointing` is optional and can be used
to increase the maximum allowed local batch size
- Train samples (`--train-num-samples`) needs to be set to 3200000, and epochs (`--epochs`) need to be set to 1
- The model (`--model`) needs to be set to `ViT-L-14`
- `--log-every-n-steps 1` needs to be set to 1
- `--local-loss`, `--gather-with-grad` need to be set

### JUBE

The benchmark JUBE file is located at [default.yaml](benchmark/jube/default.yaml), and needs to be adapted
to the respective specific system it is executed on.

More concretely, the following Slurm parameters in [default.yaml](benchmark/jube/default.yaml)
should be adapted:

- `n_gpu` (GPUs per node, default is 4)
- `threadspertask` (default is 24)
- `nodes` (default is `8`)
- `time` (default is 10 minutes)
- `batch_size` (corresponds to local batch size, default is 512)
- `queue` (i.e., the partition)
- `account` (the slurm account)

The global batch size is determined by `n_gpu` * `nodes` * `batch_size`.

Additionally, the `module load` commands on [default.yaml](benchmark/jube/default.yaml) need to be adapted to your specific system to load
low level (MPI, CUDA, CuDNN, NCCL) and high level packages (PyTorch, TorchVision, etc.). The Python packages
required by OpenCLIP are installed via pip automatically by JUBE in a Python virtual environment.

Once [default.yaml](benchmark/jube/default.yaml) is modified, you can run the benchmark using JUBE:

```
jube run benchmark/jube/default.yaml
```

For evaluation purposes, several tags are provided: `--tag test` executes the benchmark on 1 node for a simple test run; `--tag scaling` executes the benchmark on multiple nodes to evaluate scaling behavior (default: 1, 2, 4, 8, 16, 32, 64, 128, and 256 nodes)

## Verification

A successful run will display the throughput (samples per second, and samples per second per GPU) and batch duration after each iteration to the standard output; like the following:

```
2023-04-21,12:03:13 | INFO | Train Epoch: 0 [  16384/3200000 (1%)] Data (t): 90.938 Batch (t): 137.606, 119.065/s, 3.72078/s/gpu LR: 0.000000 Logit Scale: 14.286 Contrastive_loss: 9.7031 (9.7031) Loss: 9.7031 (9.7031)
2023-04-21,12:03:20 | INFO | Train Epoch: 0 [  32768/3200000 (1%)] Data (t): 0.029 Batch (t): 6.655, 2461.78/s, 76.9308/s/gpu LR: 0.000000 Logit Scale: 14.286 Contrastive_loss: 9.7031 (9.7031) Loss: 9.7031 (9.7031)
2023-04-21,12:03:26 | INFO | Train Epoch: 0 [  49152/3200000 (2%)] Data (t): 0.034 Batch (t): 6.466, 2533.71/s, 79.1783/s/gpu LR: 0.000000 Logit Scale: 14.286 Contrastive_loss: 9.7031 (9.7031) Loss: 9.7031 (9.7031)
```

Considering the third line above as an example, the throughput is 2533.71 image-text pairs per second, throughput per GPU is 79.1783 image-text pairs per second per GPU, and batch duration is 6.466 seconds.

## Results

### Command line

In order to compute the total time in seconds, we sum over the batch durations, starting from the second iteration. The first iteration includes the starting phase and is skipped.

If `<STDOUT>` is the standard output, this can be done using the following:

```bash
grep -oP '(?<=Batch \(t\): )\d+\.\d+' <STDOUT>|tail -n +2|awk '{ sum += $1 } END { print "total_time_secs:"sum }'
```

As an example, utilizing 8 nodes (32 GPUs) of JUWELS Booster, one gets:

```
total_time_secs:1253.51
```

`total_time_secs` is the metric of the benchmark and should be minimized.

### JUBE

Through JUBE, the result of the benchmark can be analyzed:

```
jube result -a benchmark/jube/bench_run
```

The total runtime (seconds) is included in the column `total_time_secs`. This is the metric of this benchmark.

Picking the example from before, an output like the following can be gotten (abbreviated):

| nodes |  model   | batch_size | done_percent_last | throughput_avg | time_per_iteration_secs_avg | total_time_secs |
|-------|----------|------------|-------------------|----------------|-----------------------------|-----------------|
|     8 | ViT-L-14 |        512 |             100.0 |        2523.70 |                        6.94 |         1253.51 |


## Baseline

The baseline configuration of the benchmark must be chosen such that the Total Time in seconds (`total_time_secs`) is 1260 s or faster for training the ViT-L/14 model on the dataset of 3.2M samples. This value was achieved on JUWELS Booster using 8 nodes with 4 NVIDIA A100 GPUs each.
