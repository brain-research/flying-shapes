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


"""Utility functions mostly related to bounding boxes and tracking metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def bbox_iou(target, output):
  """Computes the intersection over union of two bounding boxes.

  Note this function is permissive wrt its input bounding box arguments. Both
  target and output are assumed to be of the form [y_value, x_value, y_value,
  x_value] and the function considers the independent permutations of the x and
  y values necessary to yield a valid bounding box. Additionally no validation
  or clipping is done to ensure input coordinates are in [0, 1].

  Args:
    target: A tensor of shape [batch, time, 4], the target bounding boxes.
    output: A tensor of shape [batch, time, 4], the output bounding boxes.

  Returns:
    A tensor of shape [batch*time], the intersection over union loss for each
    bbox.
  """
  cond_target = tf.reduce_all(tf.logical_not(tf.is_nan(target)))
  cond_output = tf.reduce_all(tf.logical_not(tf.is_nan(output)))
  assert_target = tf.Assert(cond_target, [target])
  assert_output = tf.Assert(cond_output, [output])

  with tf.control_dependencies([assert_target, assert_output]):
    target = tf.reshape(target, [-1, 4])
    output = tf.reshape(output, [-1, 4])

  out_xmin = tf.minimum(output[:, 1], output[:, 3])
  out_xmax = tf.maximum(output[:, 1], output[:, 3])
  out_ymin = tf.minimum(output[:, 0], output[:, 2])
  out_ymax = tf.maximum(output[:, 0], output[:, 2])

  tar_xmin = tf.minimum(target[:, 1], target[:, 3])
  tar_xmax = tf.maximum(target[:, 1], target[:, 3])
  tar_ymin = tf.minimum(target[:, 0], target[:, 2])
  tar_ymax = tf.maximum(target[:, 0], target[:, 2])

  out_area = (out_xmax - out_xmin) * (out_ymax - out_ymin)
  target_area = (tar_xmax - tar_xmin) * (tar_ymax - tar_ymin)

  y_intersection = tf.nn.relu(
      tf.minimum(out_ymax, tar_ymax) - tf.maximum(out_ymin, tar_ymin))
  x_intersection = tf.nn.relu(
      tf.minimum(out_xmax, tar_xmax) - tf.maximum(out_xmin, tar_xmin))

  intersection = y_intersection * x_intersection
  union = out_area + target_area - intersection + 1e-8

  iou = intersection / union

  return iou


def get_failures_and_robustness(ious, threshold, ignore_frames,
                                robustness_magic_number):
  """Compute fails and robustness.

  Robustness measures the average number of times the that the output fails
  to capture the target in the sequence. Failure is measured by IOU being below
  a certain threshold. Nonfailure count is sequence length - failure count.
  This is a pyfunc because tf.scan can't accumulate to a tuple (we would need
  to accumulate both ignore_frames and failure_count)

  Args:
    ious: A tensor of shape [seq_len]; the ious for each bbox
    threshold: The threshold value for IOU; under this value the tracking is
      considered "failed".
    ignore_frames: The number of frames to ignore (i.e. don't check for failure)
      after the frame in which tracking fails (based on VOT reset-based methods
      where the tracker is reset after a fail).
    robustness_magic_number: A magic number that Re3 uses in the robustness
       calculation; if this is 0 robustness is calculated as the average number
       of failures over the sequence; if anything else it is used in
       e^(robustness_magic_number*fails/seq_len) as in Re3.
       30 is the magic number used by Re3.

  Returns:
    Three ndarrays of shape [1]: first the nonfailure count, then failure count
    for that sequence, and then the robustness.
  """
  failure_count = 0
  nonfailure_count = 0
  ignore_countdown = 0
  for iou in ious:
    if ignore_countdown > 0:
      ignore_countdown -= 1
    else:
      if iou <= threshold:
        failure_count += 1
        ignore_countdown = ignore_frames
      else:
        nonfailure_count += 1

  fail_count = np.asarray([failure_count], dtype=np.int64)
  nonfail_count = np.asarray([nonfailure_count], dtype=np.int64)
  if robustness_magic_number == 0:
    robustness = fail_count.astype(np.float32) / len(ious)
  else:
    # This is how Re3 calculates robusness, but it is not bounded and it
    # has a magic number and other literature (e.g. VOT) describes
    # robustness as just average failure count; a more interpretable number.
    robustness = np.exp(
        -robustness_magic_number * fail_count.astype(np.float32) / len(ious))

  return nonfail_count, fail_count, 1-robustness
