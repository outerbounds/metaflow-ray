# Introduction
This set of examples uses Ray in a variety of Metaflow workflows to train or tune many models in parallel, evaluate them, and serve with Ray Serve and Fast API. 

## Training
### Single Node
```
python train.py --environment=conda run
```

### Multinode
```
python train_parallel.py --environment=conda run
```

## Tuning
### Single Node
```
python tune.py --environment=conda run
```

### Multinode
```
python tune_parallel.py --environment=conda run
```

## Scoring
```
python score.py --environment=conda
```

## Serving
```
serve run server:batch_preds
```