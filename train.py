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

"""Try out models, losses, etc. on the Flying Shapes dataset.

   (i)  import dataset and check shapes
   (ii)  single-frame RNN/LSTM
   (iii) cropping and combining crop information
   (iv)  auxiliary tasks (e.g. representation loss, detection)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os

from cycler import cycler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import models
from third_party import dataset
import util


def main(argv):
  del argv  # Unused.
  #################
  # CONFIG
  #################

  experiment_name = None
  num_epochs = 100
  num_train_iter = 50
  test_every_n_epochs = 1
  batch_size = 5
  learning_rate = 1e-4
  seq_len = 10
  num_hidden = 100
  outputlayer_size = 4
  ssprob = 1  # scheduled sampling
  x_size = 32
  lstm_type = 'vanilla'  # 'conv'  #
  convnet_type = 'small'
  conv_out_size = 72
  join_layer = 'none'  # 'concat'  #
  ds_loss_type =  'cosine'  # None  # 'xent' #
  iou_loss = False  # True
  # bbox_loss_parameterization = 'coords'
  detection = None  # 'shape'
  device = 'gpu'
  dtype = tf.float32
  separate_lstms = False  # True

  config = locals() #dict(locals(), **FLAGS) #update locals with any flags passed by cmdln

  # make experimental directory
  if experiment_name is None:
    experiment_path = '_'.join([lstm_type, convnet_type, ds_loss_type,
                                join_layer, str(ssprob),
                                str(batch_size), str(seq_len),
                                str(num_epochs)])
  else:
    experiment_path = experiment_name

  i = 0
  while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
  experiment_path = experiment_path + "_" + str(i)
  os.mkdir(experiment_path)
  config['experiment_path'] = experiment_path
  print('Saving to ' + str(experiment_path))

  # write config file
  with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
    for key in sorted(config):
      f.write(key + '\t' + str(config[key]) + '\n')

  # open log file
  log_file = open(os.path.join(experiment_path, 'log.txt'), 'w')

  # if detection == 'shape':
  #   labels = []

  #################
  # SET UP GRAPH
  #################

  tf.reset_default_graph()  #

  with tf.device('/' + device + ':0'):
    #################
    # MODEL PARAMS
    #################

    if convnet_type == 'small':
      convnet = models.SmallConvNet()

    if separate_lstms:
      preframelstm_w = tf.get_variable('preframelstm_weight', [conv_out_size, num_hidden], dtype=dtype)
      preframelstm_b = tf.get_variable('prereplstm_bias', [num_hidden], dtype=dtype)

      prereplstm_w = tf.get_variable('prelstm_weight', [conv_out_size, num_hidden], dtype=dtype)
      prereplstm_b = tf.get_variable('prelstm_bias', [num_hidden], dtype=dtype)

      frame_lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
          tf.get_variable('initial_frame_c', [batch_size, num_hidden], dtype=dtype),
          tf.get_variable('initial_frame_h', [batch_size, num_hidden], dtype=dtype)
      )
      rep_lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
          tf.get_variable('initial_rep_c', [batch_size, num_hidden], dtype=dtype),
          tf.get_variable('initial_rep_h', [batch_size, num_hidden], dtype=dtype)
      )
      if lstm_type == 'conv':
        frame_cell = tf.contrib.rnn.ConvLSTMCell(num_hidden)
        rep_cell = tf.contrib.rnn.ConvLSTMCell(num_hidden)
      else:
        frame_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
        rep_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)

    else:
      prelstm_w = tf.get_variable('prelstm_weight', [conv_out_size, num_hidden], dtype=dtype)
      prelstm_b = tf.get_variable('prelstm_bias', [num_hidden], dtype=dtype)

      lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
          tf.get_variable('initial_c', [batch_size, num_hidden], dtype=dtype),
          tf.get_variable('initial_h', [batch_size, num_hidden], dtype=dtype)
      )
      if lstm_type == 'conv':
        cell = tf.contrib.rnn.ConvLSTMCell(num_hidden)
      else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)

    # params of output bbox linear layer
    bbox_w = tf.get_variable('bbox_weight', [num_hidden, outputlayer_size])
    bbox_b = tf.get_variable('bbox_bias', [outputlayer_size])

    if detection is not None:
      detection_w =  tf.get_variable('detection_weight', [num_hidden, labels_size], dtype=dtype)
      detection_b = tf.get_variable('detection_bias', [labels_size], dtype=dtype)

    #################
    # TRAIN GRAPH
    #################

    with tf.name_scope('train'):
      x_train = tf.placeholder(tf.float32, shape=[batch_size, seq_len, x_size, x_size, 3], name='x_train')
      bboxes = tf.placeholder(tf.float32, shape=[batch_size, seq_len, 4], name='bboxes')
      if separate_lstms:
        frame_state_t = frame_lstm_initial_state
        rep_state_t = rep_lstm_initial_state
      else:
        state_t = lstm_initial_state
      # manual unroll
      reps = []
      initialreps = []
      pred_bboxes = []
      target_bboxes = []
      with tf.variable_scope('observer'):
        for t in range(seq_len):
          if t > 0:
            tf.get_variable_scope().reuse_variables()

            # scheduled sampling
            r = tf.random_uniform([1])
            target_bbox = tf.cond(
                tf.reduce_all(r > ssprob),
                lambda: pred_bboxes[t - 1],
                lambda: bboxes[:, t, :])
          else:
            target_bbox = bboxes[:, t, :]
          input_t = x_train[:, t, :, :, :]
          bboxed_t = tf.image.crop_and_resize(
              input_t,
              target_bbox,
              tf.constant(range(batch_size), tf.int32),
              tf.constant([x_size, x_size], tf.int32)
          )
          conv_bboxed = tf.contrib.layers.flatten(convnet(bboxed_t))
          conv_frame = tf.contrib.layers.flatten(convnet(input_t))
          if join_layer == 'concat':
            lstm_in = conv_frame + conv_bboxed
          elif join_layer == 'none':
            lstm_in = conv_frame + conv_frame
          elif join_layer == 'crossconv':
            pass
          elif join_layer == 'film':
            pass
          else:
            print ("that's not a concat layer")

          # TODO: concat before LSTM layer, or separate LSTMs for frame and rep?
          pre_lstm = tf.nn.relu(tf.matmul(lstm_in, prelstm_w) + prelstm_b)
          output_t, state_t = cell(pre_lstm, state_t)
          if t == 0:
            initial_rep = state_t
          initialreps.append(initial_rep)
          reps.append(state_t)
          if detection is not None:
            logits = tf.matmul(output_t, detection_w) + detection_b
          pred_bbox_t = tf.nn.relu(tf.matmul(output_t, bbox_w) + bbox_b)
          pred_bboxes.append(pred_bbox_t)
          target_bboxes.append(target_bbox)

        pred_bboxes_tf = tf.stack(axis=1, values=pred_bboxes, name="preds_tf")
        target_bboxes_tf = tf.stack(axis=1, values=target_bboxes, name='targets_tf')
        reps_tf = tf.stack(axis=1, values=reps, name="reps_tf")
        initialreps_tf = tf.stack(axis=1, values=initialreps, name="initialreps_tf")

    # loss, metrics, optimizer
    bbox_loss = tf.losses.absolute_difference(target_bboxes_tf, pred_bboxes_tf)
    if ds_loss_type == 'cosine':
      rep_loss = tf.abs(tf.losses.cosine_distance(initialreps_tf, reps_tf, axis=0))
      loss = bbox_loss + rep_loss
    elif ds_loss_type == 'xent':
      rep_loss = tf.losses.sigmoid_cross_entropy(initialreps_tf, reps_tf)
      loss = bbox_loss + rep_loss
    else:
      rep_loss = tf.abs(tf.losses.cosine_distance(initialreps_tf, reps_tf, axis=0))
      loss = bbox_loss

    if detection is not None:
      detection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
      loss += detection_loss

    iou = util.bbox_iou(target_bboxes_tf, pred_bboxes_tf)
    mean_iou = tf.reduce_mean(iou)

    if iou_loss is not None:
      loss += mean_iou

    nonfail_count, fail_count, robustness = tf.py_func(
        util.get_failures_and_robustness, [iou, 0, 0, 0],
        [tf.int64, tf.int64, tf.float32],
        name='failure_and_robustness')

    #optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grad_and_vars = optimizer.compute_gradients(loss)
    optim = optimizer.apply_gradients(grad_and_vars)


    #################
    # TESTING GRAPH
    #################
#
#     with tf.name_scope('test'):
#       x_test = tf.placeholder(tf.float32, shape=[batch_size, seq_len, x_size, x_size, 3])
#       bboxes = tf.placeholder(tf.float32, shape=[batch_size, seq_len, 4])
#       state_t = lstm_initial_state
#       # manual unroll
#       reps = []
#       initialreps = []
#       pred_bboxes = []
#       target_bboxes = []
#       with tf.variable_scope('observer'):
#         for t in range(seq_len):
#           if t > 0:
#             tf.get_variable_scope().reuse_variables()
#
#             # scheduled sampling
#             r = tf.random_uniform([1])
#             target_bbox = tf.cond(
#                 tf.reduce_all(r > ssprob),
#                 lambda: pred_bboxes[t - 1],
#                 lambda: bboxes[:, t, :])
#           else:
#             target_bbox = bboxes[:, t, :]
#           input_t = x_train[:, t, :, :, :]
#           conv_out = tf.contrib.layers.flatten(convnet(input_t))
#           pre_lstm = tf.nn.relu(tf.matmul(conv_out, prelstm_w) + prelstm_b)
#           output_t, state_t = cell(pre_lstm, state_t)
#           if t == 0:
#             initial_rep = state_t
#           initialreps.append(initial_rep)
#           reps.append(state_t)
#           # if detection is not None:
#           #   logits = tf.matmul(output_t, detection_w) + detection_b
#           pred_bbox_t = tf.nn.relu(tf.matmul(output_t, bbox_w) + bbox_b)
#           pred_bboxes.append(pred_bbox_t)
#           target_bboxes.append(target_bbox)
#
#         pred_bboxes_tf = tf.stack(axis=1, values=pred_bboxes)
#         target_bboxes_tf = tf.stack(axis=1, values=target_bboxes)
#         reps_tf = tf.stack(axis=1, values=reps)
#         initialreps_tf = tf.stack(axis=1, values=initialreps)

    #################
    # RUN MAIN LOOP
    #################

    cf = tf.ConfigProto(allow_soft_placement=True)
    saver = tf.train.Saver()
    with tf.Session(config=cf) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_local_variables())

      data_source = dataset.FlyingShapesDataHandler(batch_size=batch_size,
                                                    seq_len=seq_len)

      # np_test_batch = data_source.GetUnlabelledBatch()
      # test_batch = {
      #     'test_image': np_batch['image'],#tf.convert_to_tensor(np_batch['image'], dtype=tf.float32),
      #     'test_bbox': np_batch['bbox']#tf.convert_to_tensor(np_batch['bbox'], dtype=tf.float32)
      # }

      train_loss = []
      train_rep_loss = []
      train_iou = []
      train_fail = []
      train_rob = []
      # test_loss = []
      # test_iou = []
      # test_fail = []
      # test_rob = []

      for n in range(num_epochs):
        np_batch = data_source.GetUnlabelledBatch()
        batch = {
          'image': np_batch['image'],#tf.convert_to_tensor(np_batch['image'], dtype=tf.float32),
          'bbox': np_batch['bbox']#tf.convert_to_tensor(np_batch['bbox'], dtype=tf.float32)
        }
        print ('Epoch ' + str(n) )
        for i in range(num_train_iter):
          # if True: #detection is None:
            # np_batch = data_source.GetUnlabelledBatch()
            # batch = {
            #     'image': np_batch['image'],#tf.convert_to_tensor(np_batch['image'], dtype=tf.float32),
            #     'bbox': np_batch['bbox']#tf.convert_to_tensor(np_batch['bbox'], dtype=tf.float32)
            # }
          # else:
          #   pass
          res = sess.run({
              'loss': loss,
              'rep_loss': rep_loss,
              'optim': optim,
              'iou': mean_iou,
              'fail_rate': fail_count,
              'robustness': robustness,
              'pred_bboxes': pred_bboxes_tf,
              'target_bboxes': target_bboxes_tf,
              'initial_rep': initial_rep
              },
              feed_dict={
                  x_train: batch['image'],
                  bboxes: batch['bbox']
              })
          #fig, ax = plt.subplots(1)
          #ax.imshow(im)
          #rect = patches.Rectangle(, linewidth=1

          train_loss.append(res['loss'])
          train_rep_loss.append(res['rep_loss'])
          train_iou.append(res['iou'])
          train_fail.append(res['fail_rate'])
          train_rob.append(res['robustness'])


          # PRINT TO LOG FILE AND STDERR

          log_str = ('Train ' + str(i) + ': ' +
                     str(res['rep_loss']) + '\t' +
                     str(res['loss']) + '\t' +
                     str(res['iou']) +  '\t' +
                     str(res['fail_rate'][0]) + '\t' +
                     str(res['robustness'][0]))

          print(log_str)
          log_file.write(log_str + '\n')

        # PLOT IMGS AND BBOXES
        num_rows = 1
        plt.figure(2, figsize=(20, 1))
        plt.clf()
        for i in xrange(seq_len):
          figgy = plt.subplot(num_rows, seq_len, i+1)
          gt = batch['bbox'][0][i]
          targetbox = res['target_bboxes'][0][i]
          predbox = res['pred_bboxes'][0][i]
          figgy.add_patch(patches.Rectangle((gt[1],
                                             gt[0]),
                                            (gt[3] - gt[1]),
                                            (gt[2] - gt[0]),
                                            linewidth=2,
                                            edgecolor='b',
                                            facecolor='none',
                                            alpha=0.6))
          figgy.add_patch(patches.Rectangle((targetbox[1],
                                             targetbox[0]),
                                            (targetbox[3] - targetbox[1]),
                                            (targetbox[2] - targetbox[0]),
                                            linewidth=2,
                                            edgecolor='y',
                                            facecolor='none',
                                            alpha=0.6))
          figgy.add_patch(patches.Rectangle((predbox[1],
                                             predbox[0]),
                                            (predbox[3] - predbox[1]),
                                            (predbox[2] - predbox[0]),
                                            linewidth=2,
                                            edgecolor='g',
                                            facecolor='none',
                                            alpha=0.6))
          plt.imshow(batch['image'][0][i])
          plt.axis('off')
        plt.savefig(os.path.join(experiment_path,'bboxes'+str(n)), bbox_inches='tight')
      saver.save(sess, os.path.join(experiment_path, 'saved_model'))
      #plt.draw()


        # print('bbox:', batch['bbox'])
        # print('target:', res['target_bboxes'])
        # print('pred:', res['pred_bboxes'])

#       if (n + 1) % test_every_n_epochs == 0:
#         np_test_batch = data_source.GetUnlabelledBatch()
#         test_batch = {
#             'image': tf.convert_to_tensor(np_batch['image'], dtype=tf.float32),
#             'bbox': tf.convert_to_tensor(np_batch['bbox'], dtype=tf.float32)
#         }
#         test_res = sess.run({
#             'test_loss': test_loss,
#             'test_iou': test_iou,
#             'test_fail_rate': test_fail_rate,
#             'test_robustness': test_robustness
#             },
#             feed_dict={
#                 x_test: test_batch['image'],
#                 bboxes_ttest: test_batch['bbox']
#             })
#         # test_loss.append(res['test_loss'])
#         # test_iou.append(res['test_iou'])
#         # test_fail.append(res['test_fail_rate'])
#         # test_rob.append(res['test_robustness'])
#
#         print('Test 'str(i) + ': ' +
#               str(res['test_loss']) + '\t' +
#               str(res['test_iou']) +  '\t' +
#               str(res['test_fail_rate'][0]) + '\t' +
#               str(res['test_robustness'][0])
#              )

    log_file.close()
    plt.close()

    plt.plot(train_loss, label='Train Loss')
    # plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_path,'loss.png'))
    plt.gcf().clear()

    plt.plot(train_rep_loss, label='Rep Loss')
    # plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_path,'rep_loss.png'))
    plt.gcf().clear()

    plt.plot(train_iou, label='Train IOU')
    # plt.plot(test_iou, label='Test IOU')
    plt.legend()
    plt.savefig(os.path.join(experiment_path,'iou.png'))
    plt.gcf().clear()

    plt.plot(train_fail, label='Train Failure Rate')
    # plt.plot(test_fail, label='Test Failure Rate')
    plt.legend()
    plt.savefig(os.path.join(experiment_path,'fails.png'))
    plt.gcf().clear()

    plt.plot(train_rob, label='Train Robustness')
    # plt.plot(test_rob, label='Test Robustness')
    plt.legend()
    plt.savefig(os.path.join(experiment_path,'robustness.png'))



if __name__ == '__main__':
  app.run(main)
