# Introduction
This set of examples uses Ray Train to train many XGBoost models in parallel. 
There are four example workflows showing the cross product of single- and multi-node with CPU and GPU runs.

## CPU single node
```
python xgb_cpu.py --environment=conda run
```

## CPU multinode
```
python xgb_cpu_multinode.py --environment=conda run
```

## GPU single node
```
python xgb_gpu.py --environment=conda run
```

## GPU multinode
```
python xgb_gpu_multinode.py --environment=conda run
```
