# Introduction
This set of examples uses Ray Tune to train many PyTorch models in parallel. 
There are four example workflows showing the cross product of single- and multi-node with CPU and GPU runs.

## CPU single node
```
python pytorch_cpu.py --environment=conda run
```

## CPU multinode
```
python pytorch_cpu_multinode.py --environment=conda run
```

## GPU single node
```
python pytorch_gpu.py --environment=conda run
```

## GPU multinode
```
python pytorch_gpu_multinode.py --environment=conda run
```