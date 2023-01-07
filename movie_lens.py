# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Training a deep NN on MovieLens with differentially private Adam optimizer."""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow.compat.v1 import estimator as tf_compat_v1_estimator
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from tensorflow_privacy.privacy.optimizers import dp_optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm
 
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').disabled = True


sampling_batch = 10000
microbatches = 10000
num_examples = 800167

#### Model
def nn_model_fn(features, labels, mode, params):

  """NN adapted from github.com/hexiangnan/neural_collaborative_filtering."""
  n_latent_factors_user = 10
  n_latent_factors_movie = 10
  n_latent_factors_mf = 5

  user_input = tf.reshape(features['user'], [-1, 1])
  item_input = tf.reshape(features['movie'], [-1, 1])

  # number of users: 6040; number of movies: 3706
  mf_embedding_user = tf.keras.layers.Embedding(
      6040, n_latent_factors_mf, input_length=1)
  mf_embedding_item = tf.keras.layers.Embedding(
      3706, n_latent_factors_mf, input_length=1)
  mlp_embedding_user = tf.keras.layers.Embedding(
      6040, n_latent_factors_user, input_length=1)
  mlp_embedding_item = tf.keras.layers.Embedding(
      3706, n_latent_factors_movie, input_length=1)

  # GMF part
  # Flatten the embedding vector as latent features in GMF
  mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
  mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
  # Element-wise multiply
  mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

  # MLP part
  # Flatten the embede, so returning NUMA node zerodding vector as latent features in MLP
  mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
  mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
  # Concatenation of two latent features
  mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

  predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

  logits = tf.keras.layers.Dense(5)(predict_vector)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf_estimator.ModeKeys.TRAIN:
    if params['dpsgd']:
      # Use DP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.compat.v1.train.OpThe notion of differentitimizer should be wrappable in differentially
      # private counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPAdamGaussianOptimizer(
          l2_norm_clip=params['l2_norm_clip'],
          noise_multiplier=params['noise_multiplier'],
          num_microbatches=microbatches,
          learning_rate=params['learning_rate'])
      opt_loss = vector_loss
    else:
      optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=params['learning_rate'])
      opt_loss = scalar_loss

    global_step = tf.compat.v1.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf_estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  if mode == tf_estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'rmse':
            tf.compat.v1.metrics.root_mean_squared_error(
                labels=tf.cast(labels, tf.float32),
                predictions=tf.tensordot(
                    a=tf.nn.softmax(logits, axis=1),
                    b=tf.constant(np.array([0, 1, 2, 3, 4]), dtype=tf.float32),
                    axes=1))
    }
    return tf_estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, eval_metric_ops=eval_metric_ops)
  return None

#### Dataloader
def load_movielens():
    """Loads MovieLens 1M as from https://grouplens.org/datasets/movielens/1m."""
    data = pd.read_csv(
        '/home/umair/projects/adversarial-robustness-toolbox/datasets/ml-1m/ratings.dat',
        sep='::',
        header=None,
        names=['userId', 'movieId', 'rating', 'timestamp'])
    n_users = len(set(data['userId']))
    n_movies = len(set(data['movieId']))
    print('number of movie: ', n_movies)
    print('number of user: ', n_users)

    # give unique dense movie index to movieId
    data['movieIndex'] = stats.rankdata(data['movieId'], method='dense') 
    # minus one to reduce tde, so returning NUMA node zerohe minimum value to 0, which is the start of col index

    print('number of ratings:', data.shape[0])
    print('percentage of sparsity:',
        (1 - data.shape[0] / n_users / n_movies) * 100, '%')

    train, test = train_test_split(data, test_size=0.2, random_state=100)

    return train.values - 1, test.values - 1, np.mean(train['rating'])

#### Main function (trainer)
def main(args):
    tf.compat.v1.logging.set_verbosity(3)
    # Load training and test data.
    train_data, test_data, _ = load_movielens()

    # Instantiate the tf.Estimator.
    ml_classifier = tf_estimator.Estimator(
        model_fn=nn_model_fn, model_dir=args['model_dir'], params=args)

    # Create tf.Estimator input functions for the training and test data.
    eval_input_fn = tf_compat_v1_estimator.inputs.numpy_input_fn(
        x={
            'user': test_data[:, 0],
            'movie': test_data[:, 4]
        },
        y=test_data[:, 2],
        num_epochs=1,
        shuffle=False)

    # Training loop.
    steps_per_epoch = num_examples // sampling_batch

    test_accuracy_list = []
    eps_list = []
    mu_list = []
    for epoch in tqdm(range(1, args['epochs'] + 1)):
        for _ in tqdm(range(steps_per_epoch)):
            whether = np.random.random_sample(num_examples) > (
                1 - sampling_batch / num_examples)
            subsampling = [i for i in np.arange(num_examples) if whether[i]]
            global microbatches
            microbatches = len(subsampling)

            train_input_fn = tf_compat_v1_estimator.inputs.numpy_input_fn(
                x={
                    'user': train_data[subsampling, 0],
                    'movie': train_data[subsampling, 4]
                },
                y=train_data[subsampling, 2],
                batch_size=len(subsampling),
                num_epochs=1,
                shuffle=True)
        
            # Train the model for one step.
            ml_classifier.train(input_fn=train_input_fn, steps=1)

        # Evaluate the model and print results
        eval_results = ml_classifier.evaluate(input_fn=eval_input_fn)
        test_accuracy = eval_results['rmse']
        test_accuracy_list.append(test_accuracy)
        print('Test RMSE after %d epochs is: %.3f' % (epoch, test_accuracy))

        # Compute the privacy budget expended so far.
        if args['dpsgd']:
            eps = compute_eps_poisson(epoch, args['noise_multiplier'], num_examples,
                                        sampling_batch, args["delta"])
            mu = compute_mu_poisson(epoch, args['noise_multiplier'], num_examples,
                                    sampling_batch)
            eps_list.append(eps)
            mu_list.append(mu)
            print('For delta=1e-6, the current epsilon is: %.2f' % eps)
            print('For delta=1e-6, the current mu is: %.2f' % mu)

            if mu > args['max_mu']:
                break
        else:
            print('Trained with vanilla non-private SGD optimizer')

    return test_accuracy_list, eps_list, mu_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('-mu', '--max_mu', type=int,  default=2, help='max value of mu (program terminates if mu gets bigger than this)')
    parser.add_argument('-lnc', '--l2_norm_clip', type=float, default=5, help='max L2 norm value')
    parser.add_argument('-nm', '--noise_multiplier', type=float, default=0.55, help='amount of noise to be added during training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('-d', '--delta', type=float, default=1e-6, help='delta value for eplison calculation')
    parser.add_argument('--model_dir', type=str, default=None, help='path top save model')
    parser.add_argument('--use_sgd', action='store_false', dest='dpsgd', 
                        default=True, help='if True valila sdg will be used for optimization otherwise DP-SQG will be used.')

    args = parser.parse_args()
    main(args.__dict__)

