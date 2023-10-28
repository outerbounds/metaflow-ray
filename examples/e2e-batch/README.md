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