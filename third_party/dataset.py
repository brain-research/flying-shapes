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

"""An infinite dataset of flying shapes bouncing around on a black background.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class FlyingShapesDataHandler(object):
  """Data Handler that creates the Flying Shapes dataset on the fly."""

  def __init__(self, config=None, file_path=None, batch_size=None,
               seq_len=None):

    if config is None:
      self.seq_length_ = 10
      self.batch_size_ = 7
      self.image_size_ = 32
      self.num_digits_ = 3
      self.step_length_ = 0.5  # 1/initial_speed (i_s*cos(angle) = velocity)
      self.digit_size_ = 16
      self.frame_size_ = (self.image_size_ ** 2) * 3
      self.file_path_ = 'flying_shapes2.pkl'
    else:
      self.seq_length_ = config['num_frames']
      self.batch_size_ = config['batch_size']
      self.image_size_ = config['image_size']
      self.num_digits_ = config['num_digits']
      self.step_length_ = config['step_length']
      self.digit_size_ = config['digit_size']
      self.frame_size_ = (config['image_size'] ** 2) * 3
      self.file_path_ = config['file_path']
    if file_path is not None:
      self.file_path_ = file_path
    if batch_size is not None:
      self.batch_size_ = batch_size
    if seq_len is not None:
      self.seq_length_ = seq_len

    try:
      print (self.file_path_)
      dataset = pickle.load(open(self.file_path_))
    except Exception as e:
      raise e

    self.data_ = np.float32(dataset['images']).transpose(0, 3, 1, 2) / 255.
    self.labels_ = dataset['labels']
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetSeqLength(self):
    return self.seq_length_

  def GetRandomTrajectory(self, batch_size):
    """Generates a random trajectory, dealing with bounces off the 'walls'."""
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in xrange(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in xrange(batch_size):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int64)
    start_x = (canvas_size * start_x).astype(np.int64)

    return start_y, start_x

  def Overlap(self, a, b):
    return np.maximum(a, b)

  def GetLabelledBatch(self, label_key=None):
    """Return a labelled minibatch of data; a dict of image, bbox, label."""
    start_y, start_x = self.GetRandomTrajectory(
        self.batch_size_ * self.num_digits_)

    labels_by_index = [[] for label in self.labels_]
    for elem in self.labels_:
      labels_by_index[self.labels_[elem][0]] = [self.labels_[elem][1],
                                                self.labels_[elem][2], elem]

    if label_key == 'colour' or label_key == 'color':
      labels_size = len(set([l[0] for l in labels_by_index]))
    elif label_key == 'shape':
      labels_size = len(set([l[1] for l in labels_by_index]))
    else:  # unique classes for each colour+shape combination
      labels_size = len(labels_by_index)

    # minibatch data
    data = np.zeros((self.batch_size_, self.seq_length_,
                     3, self.image_size_, self.image_size_), dtype=np.float32)
    bboxes = np.zeros((self.batch_size_, self.seq_length_, 4))
    labels = np.zeros((self.batch_size_, self.seq_length_, labels_size))

    for j in xrange(self.batch_size_):
      for n in xrange(self.num_digits_):

        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :, :]
        labels_by_index = [[] for label in self.labels_]
        for elem in self.labels_:
          labels_by_index[self.labels_[elem][0]] = [self.labels_[elem][1],
                                                    self.labels_[elem][2], elem]

        if label_key == 'colour' or label_key == 'color':
          label = labels_by_index[ind][0]
        elif label_key == 'shape':
          label = labels_by_index[ind][1]
        else:  # unique classes for each colour+shape combination
          label = ind

        # generate video
        for i in xrange(self.seq_length_):
          top = start_y[i, j * self.num_digits_ + n]
          left = start_x[i, j * self.num_digits_ + n]
          bottom = top  + self.digit_size_
          right = left + self.digit_size_
          for c in [0, 1, 2]:
            data[j, i, c, top:bottom, left:right] = self.Overlap(
                data[j, i, c, top:bottom, left:right], digit_image[c])
          bboxes[j, i] = np.asarray([top, left, bottom, right])
          labels[j, i] = np.eye(labels_size)[label]

    # imagenet-style format
    return {'image': data.transpose((0, 1, 3, 4, 2)),
            'bbox': bboxes,
            'label': labels.transpose((1, 0, 2))}

  def GetUnlabelledBatch(self):
    """Return a minibatch of data as a dictionary in the style of imagenet."""
    start_y, start_x = self.GetRandomTrajectory(
        self.batch_size_ * self.num_digits_)

    # minibatch data
    data = np.zeros((self.batch_size_, self.seq_length_,
                     3, self.image_size_, self.image_size_), dtype=np.float32)
    bboxes = np.zeros((self.batch_size_, self.seq_length_, 4), dtype=np.float32)

    for j in xrange(self.batch_size_):
      for n in xrange(self.num_digits_):

        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :, :]

        # generate video
        for i in xrange(self.seq_length_):
          top = start_y[i, j * self.num_digits_ + n]
          left = start_x[i, j * self.num_digits_ + n]
          bottom = top  + self.digit_size_
          right = left + self.digit_size_
          for c in [0, 1, 2]:
            data[j, i, c, top:bottom, left:right] = self.Overlap(
                data[j, i, c, top:bottom, left:right], digit_image[c])
          bboxes[j, i] = np.asarray([top, left, bottom, right],
                                    dtype=np.float32)

    # imagenet-style format
    return {'image': data.transpose((0, 1, 3, 4, 2)), 'bbox': bboxes}

  def DisplayData(self, data, fig=1, case_id=0, output_file=None):
    """Display a sequence of frames with bboxes using matplotlib."""

    data = data[0]
    bboxes = data[1]

    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file = '%s_frames%s' % (name, ext)

    # get data
    data = data[case_id, :]
    bboxes = bboxes[case_id, :]
    data[data > 1.] = 1.
    data[data < 0.] = 0.
    data = data.reshape(-1, 3, self.image_size_, self.image_size_)
    data = data.transpose(0, 2, 3, 1)

    # create figure
    num_rows = 1
    plt.figure(2*fig, figsize=(20, 1))
    plt.clf()
    for i in xrange(self.seq_length_):
      figgy = plt.subplot(num_rows, self.seq_length_, i+1)
      figgy.add_patch(patches.Rectangle((bboxes[i][1], bboxes[i][0]),
                                        (bboxes[i][3] - bboxes[i][1]),
                                        (bboxes[i][2] - bboxes[i][0]),
                                        linewidth=2, edgecolor='b',
                                        facecolor='none'))
      plt.imshow(data[i])
      plt.axis('off')
    plt.draw()

    if output_file is not None:
      plt.savefig(output_file, bbox_inches='tight')
