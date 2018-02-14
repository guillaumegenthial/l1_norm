# Learn L1 norm with a Neural Net

Uses [CS230 starter code package](https://cs230-stanford.github.io) for the pipeline.

## Quickstart

Create some data in `data/`

```
python build_data.py
```


Train

```
python train.py
```

Evaluate

```
python evaluate.py
```

Perform hyperparameter search over the learning rate


```
python search_hyperparams.py
```


## Thought Process

### Data

See `build_data.py`

We need to generate some data. Requirements:
- enough to train, not too much for speed issues
- train and dev sets ideally have the same distribution
- test set distribution ideally a bit different
- should be reproducible (fix random seed)
- randomize dimension of vectors and scale of the distribution
- saves data to files for future re-use of the model

Then, the file `model/input_fn.py` takes care of the input data pipeline to the Graph.

### Model

See `model/model_fn.py`

Obviously, the L1 norm of x_1, ...., x_n can be computed with the following graph

```
# Compute the absolute norm of every entry
out = tf.nn.relu(out) + tf.nn.relu(-out)

# Sum over the dimension
out = tf.reduce_sum(out, axis=-1)
```

It seems to respect the requirement (only ReLUs, +, - and sum), but we can also test if we can learn the right connection for the 2 dense layers ( * [1, -1] -> relu -> [1, 1] -> reduce_sum).

### Training and Hyperparameters

- Applying Dropout would not make sense as the architecture is as minimalist as possible. L2 regularization would introduce a bias.
- Relevant hyperparameters are batch_size, learning_rate, optimization method.
- Training loss is the L2 loss between the average per component of the predicted L1 norm and the gold L1 norm
- We take the average per component and per batch so that changing other hyperparameters does not impact the choice of learning_rate and batch_size


### Results and Observations

- Both models achieve 0 L2 loss between the two L1 norms (neural and gold).
- Some sensibility to the learning_rate, resolved with hyperparameter search and the use of Adam.
- Ability to generalize