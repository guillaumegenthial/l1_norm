"""Define the model."""

import tensorflow as tf


def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    x = inputs['x']

    out = tf.expand_dims(x, axis=-1)

    if params.model_version == "trainable":
        out = tf.layers.dense(out, 2, activation=tf.nn.relu, use_bias=False)
        out = tf.layers.dense(out, 1, activation=None, use_bias=False)
    else:
        # Cheat version where we only use relus and "+" and "-", same graph as above but hardcoded
        a = tf.get_variable('a', [1]) # fake variable just to be able to train
        out = tf.nn.relu(out) + tf.nn.relu(-out) + a*1e-16

    return out


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['y']
    lengths = inputs['lengths']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        outputs = build_model(mode, inputs, params)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    mask = tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32), axis=-1)
    sum_of_dims = tf.reduce_sum(lengths)
    predictions = tf.reduce_sum(tf.reduce_sum(outputs * mask, axis=-1), axis=-1)
    l2_loss = tf.reduce_mean(tf.square(predictions - labels))
    loss = l2_loss / tf.cast(sum_of_dims, tf.float32) # loss per component for indep to other hp

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'neg_l2_loss': tf.metrics.mean(-l2_loss),
            'accuracy': tf.metrics.mean(-loss) # select on best loss
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = -loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec