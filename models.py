# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uses sonnet to make convnets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools32 import partial
import sonnet as snt
import tensorflow as tf


class SmallConvNet(object):
  """Implements simple convnet by wrapping sonnet.

    Use: output = SmallConvNet()(input)

    Attributes:
    layers: All the layers.
  """

  def __init__(self, train_batch_norm=True):
    self.layers = []
    self.layers.append(snt.Conv2D(16, [5, 5]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(
        partial(
            tf.nn.max_pool,
            ksize=[1, 5, 5, 1],
            strides=[1, 3, 3, 1],
            padding='SAME'))
    self.layers.append(snt.Conv2D(8, [5, 5]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(
        partial(
            tf.nn.max_pool,
            ksize=[1, 5, 5, 1],
            strides=[1, 5, 5, 1],
            padding='SAME'))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))

  def __call__(self, input_):
    for layer in self.layers:
      input_ = layer(input_)
    return input_


class CaffeNet(object):
  """Implements CaffeNet like network by wrapping sonnet.

  Inspired by:
  /third_party/caffe/examples/pycaffe/caffenet.py

  Use: output = CaffeNet()(input)

  Attributes:
  layers: All the layers.
  """

  def __init__(self, train_batch_norm=True):
    self.layers = []
    self.layers.append(
        snt.Conv2D(
            output_channels=96,
            kernel_shape=[11, 11],
            stride=[4, 4],
            padding='VALID'))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(
        partial(
            tf.nn.max_pool,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID'))
    self.layers.append(snt.Conv2D(output_channels=256, kernel_shape=[5, 5]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(
        partial(
            tf.nn.max_pool,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID'))
    self.layers.append(snt.Conv2D(output_channels=192, kernel_shape=[3, 3]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(snt.Conv2D(output_channels=192, kernel_shape=[3, 3]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)
    self.layers.append(snt.Conv2D(output_channels=192, kernel_shape=[3, 3]))
    self.layers.append(partial(snt.BatchNorm(), is_training=train_batch_norm))
    self.layers.append(tf.nn.relu)

  def __call__(self, input_):
    for layer in self.layers:
      input_ = layer(input_)
    return input_
