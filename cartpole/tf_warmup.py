# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import q_learning


class SGDRegressor:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)
    self.lr = 0.1

  def partial_fit(self, X, Y):
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)



class FeatureTransformer:
  def __init__(self, env):
    # observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    # NOTE!! state samples are poor, b/c you get velocities --> infinity
    observation_examples = np.random.random((20000, 4))*2 - 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
            ])
    feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = feature_examples.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(feature_transformer.dimensions)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    result = np.stack([m.predict(X) for m in self.models]).T
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


def play_one(env, model, eps, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200

    # update the model
    next = model.predict(observation)
    # print(next.shape)
    assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward


class SGDRegressor:
  def __init__(self, D):
    print("Hello TensorFlow!")
    lr = 0.1

    # create inputs, targets, params
    # matmul doesn't like when w is 1-D
    # so we make it 2-D and then flatten the prediction
    self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # make prediction and cost
    Y_hat = tf.reshape( tf.matmul(self.X, self.w), [-1] )
    delta = self.Y - Y_hat
    cost = tf.reduce_sum(delta * delta)

    # ops we want to call later
    self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    self.predict_op = Y_hat
  
    # start the session and initialize params
    init = tf.global_variables_initializer()
    self.session = tf.InteractiveSession()
    self.session.run(init)

  def partial_fit(self, X, Y):
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})


if __name__ == '__main__':
  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
