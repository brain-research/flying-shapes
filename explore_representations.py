"""Test out models on the Flying Shapes dataset.

   (i) import dataset and check shapes
   (ii) single-frame RNN/LSTM
   (iii) cropping and rep loss
   (iv) aux tasks (e.g. detection
   (v) metalearning loss


Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from google3.experimental.brain.tracking.flyingshapes.third_party import dataset

def main(argv):
  del argv  # Unused.
  #################
  # CONFIG
  #################

  experiment_name = None
  load_dataset_path = 'states_DATASET_vanilla_small_cosine_none_1_5_10_100_7.pkl'
  save_dataset = True
  num_epochs = 100
  batch_size = 5
  new_batch_size = 32
  dataset_size = 3200
  states_to_use = 'cell'  # 'states'
  timesteps_to_use = [9]
  num_batches = int(dataset_size / batch_size)
  learning_rate = 1e-3
  seq_len = 10
  num_hidden = 100
  device = 'gpu'
  dtype = tf.float32
  classification_layer = 'mlp2' # 'linear'  #  'mlp1'  #
  graph_to_restore = 'vanilla_small_cosine_none_1_5_10_100_7/saved_model.meta'

  config = locals()

  # make experimental directory
  if experiment_name is None:
    experiment_path = 'reps_' + '_'.join([classification_layer, str(batch_size), str(seq_len),
                                          str(learning_rate), str(num_epochs)])
  else:
    experiment_path = experiment_name

  i = 0
  while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
  experiment_path = experiment_path + "_" + str(i)
  os.mkdir(experiment_path)
  config['experiment_path'] = experiment_path
  print('Saving to ' + str(experiment_path))

  if save_dataset:
    save_dataset_path = 'states_DATASET_' + graph_to_restore.split('/')[0]

                                   # write config file
  with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
    for key in sorted(config):
      f.write(key + '\t' + str(config[key]) + '\n')

  # open log file
  log_file = open(os.path.join(experiment_path, 'log.txt'), 'w')


  #################
  # STATES DATASET
  #################

  if False: # load_dataset_path is not None:
    try:
      print ('Loading dataset from ' + load_dataset_path + '\n')
      new_dataset = pickle.load(open(load_dataset_path, 'rb'))
      train_set = new_dataset['train']
      val_set = new_dataset['val']
      test_set = new_dataset['test']
    except:
      print ("whups, that didn't work. Making dataset instead.")

  else:  # make dataset
    tf.reset_default_graph()  # TODO(teganm): is this necessary?

    with tf.device('/' + device + ':0'):

      cf = tf.ConfigProto(allow_soft_placement=True)
      with tf.Session(config=cf) as sess:

        #################
        # LOAD GRAPH
        #################

        saver = tf.train.import_meta_graph(graph_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        reps_tf = graph.get_tensor_by_name('train/observer/reps_tf_1:0')
        x_train = graph.get_tensor_by_name('train/Placeholder:0')
        bboxes = graph.get_tensor_by_name('train/Placeholder_1:0')
        # for op in graph.get_operations():
        #   if 'train' in str(op.name) :
        #     print (str(op.name) )
        #observer_state = graph.get_tensor_by_name('lstm/h:0')
        #observer_state = tf.stop_gradient(observer_state)
        #observer_state_shape = observer_state.get_shape().as_list()

        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_local_variables())

        #################
        # MAKE STATES DATASET
        #################

        data_source = dataset.FlyingShapesDataHandler(batch_size=batch_size,
                                                      seq_len=seq_len)
        new_dataset = {'state':[], 'label':[]}
        for i in range(num_batches):
          np_batch = data_source.GetLabelledBatch()
          batch = {
            'image': np_batch['image'],
            'bbox': np_batch['bbox'],
            'labels': np_batch['label']
          }
          res = sess.run({
              'lstm_states': reps_tf,
              },
              feed_dict={
                  x_train: batch['image'],
                  bboxes: batch['bbox']
              })
          if states_to_use == 'cell':
            new_dataset['state'].extend(res['lstm_states'][0].transpose((1,0,2)))
            new_dataset['label'].extend(batch['labels'].transpose((1,0,2))  )
          else:
            new_dataset['state'].append(res['lstm_states'][1])
            new_dataset['label'].append(batch['labels'])

      train_set = {'state': new_dataset['state'][:int(0.6*dataset_size)],
                   'label': new_dataset['label'][:int(0.6*dataset_size)]}
      dataset_states_shape = np.asarray(new_dataset['state']).shape
      print (dataset_states_shape)
      train_max = np.max(train_set['state'])
      train_min = np.min(train_set['state'] )
      print (train_max)
      print (train_min)
      normalized_states = (((new_dataset['state']-train_min) / (train_max-train_min))*2) - 1 # normalize -1 to 1
      train_set['state'] = normalized_states[:int(0.6*dataset_size)]

      val_set = {'state': normalized_states[int(0.6*dataset_size):int(0.8*dataset_size)],
                 'label': new_dataset['label'][int(0.6*dataset_size):int(0.8*dataset_size)]}
      test_set = {'state': normalized_states[int(0.8*dataset_size):],
                  'label': new_dataset['label'][int(0.8*dataset_size):]}

      train_set['state'] = np.asarray(train_set['state']).transpose(1,0,2)
      train_set['label'] = np.asarray(train_set['label']).transpose(1,0,2)
      val_set['state'] = np.asarray(val_set['state']).transpose(1,0,2)
      val_set['label'] = np.asarray(val_set['label']).transpose(1,0,2)
      test_set['state'] = np.asarray(test_set['state']).transpose(1,0,2)
      test_set['label'] = np.asarray(test_set['label']).transpose(1,0,2)

      if save_dataset_path:
        print ('Saving created dataset into ' + save_dataset_path + '\n')
        with open (save_dataset_path, 'wb') as f:
          pickle.dump({'train':train_set, 'test':test_set, 'val':val_set}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)

      print (train_set['state'].shape)


    #################
    # MAKE CLASSIFIER
    #################

    states_shape = train_set['state'][0][0].shape
    state_dim = states_shape[-1]
    labels_shape = train_set['label'][0][0].shape
    label_dim = labels_shape[-1]

    state = tf.placeholder(dtype, shape=[None, state_dim])
    label = tf.placeholder(dtype, shape=[None, label_dim])

    out = state

    if classification_layer == 'mlp1' or classification_layer == 'mlp2':
      w_m1 = tf.get_variable("w_m1", [state_dim, num_hidden], dtype=dtype)
      b_m1 = tf.get_variable("b_m1", [num_hidden], dtype=dtype)
      out  = tf.nn.relu(tf.matmul(state, w_m1) + b_m1)
      state_dim = num_hidden
    if classification_layer == 'mlp2':
      w_m2 = tf.get_variable("w_m2", [num_hidden, num_hidden], dtype=dtype)
      b_m2 = tf.get_variable("b_m2", [num_hidden], dtype=dtype)
      out  = tf.nn.relu(tf.matmul(out, w_m2) + b_m2)

    #Linear layer and softmax
    w_linear = tf.get_variable("w_linear", [state_dim, label_dim], dtype=dtype)
    b_linear = tf.get_variable("b_linear", [label_dim], dtype=dtype)
    logits   = tf.matmul(out, w_linear) + b_linear

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #################
    # TRAIN/VAL CLASSIFIER
    #################

    with tf.device('/' + device + ':0'):
      cf = tf.ConfigProto(allow_soft_placement=True)
      with tf.Session(config=cf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())


        train_loss = [[] for i in timesteps_to_use]
        train_acc = [[] for i in timesteps_to_use]
        val_loss = [[] for i in timesteps_to_use]
        val_acc = [[] for i in timesteps_to_use]
        test_loss = [[] for i in timesteps_to_use]
        test_acc = [[] for i in timesteps_to_use]

        for j,t in enumerate(timesteps_to_use):
          for i in range(num_epochs):
            log_str = ('Timestep ' +str(t) + ' Epoch '+ str(i) + ': ')
            print (log_str)
            log_file.write(log_str+'\n')

#             for n in range(int(len(train_set['state'])/new_batch_size) - new_batch_size):
#             #for n, train_ex in enumerate(train_set['state']):
#               start_idx = n*new_batch_size # 1*32: 32+32
#               end_idx = (n*new_batch_size) + new_batch_size
#
#               print (np.asarray(train_set['state'][t][start_idx:end_idx]).shape)
#
#               res = sess.run({'loss': loss,
#                               'acc': accuracy,
#                               'optim': optimizer},
#                              feed_dict={
#                                  state: train_set['state'][t][start_idx:end_idx],
#                                  label: train_set['label'][0][start_idx:end_idx]
#                              })
#               train_loss[t].append(res['loss'])
#               train_acc[t].append(res['acc'])
#               train_str = '  Timestep ' + str(t) + ' Train acc: '+str(res['acc']) + '\t loss:' + str(res['loss'])
#               print (train_str)
#               log_file.write(train_str +'\n')
            #for n, train_ex in enumerate(train_set['state'][t]):
            num_train_batches = int(len(train_set['state'][0])/new_batch_size)
            print (num_train_batches)
            for n in range(num_train_batches):
              start_idx = n*new_batch_size # 1*32: 32+32
              end_idx = (n*new_batch_size) + new_batch_size
              batch = {'state': train_set['state'][t][start_idx:end_idx],
                       'label': train_set['label'][t][start_idx:end_idx]
              }
              res = sess.run({'loss': loss,
                              'acc': accuracy,
                              'optim': optimizer},
                             feed_dict={
                                 state: batch['state'], #[train_ex],  # [train_set['state'][t][0]], #
                                 label: batch['label'] #[train_set['label'][0][n]] # [train_set['label'][0][0]] #
                             })
              train_loss[j].append(res['loss'])
              train_acc[j].append(res['acc'])
              train_str = '  Timestep ' + str(t) + ' Train acc: '+str(res['acc']) + '\t loss:' + str(res['loss'])
              print (train_str)
              log_file.write(train_str +'\n')

            for n, val_ex in enumerate(val_set['state'][t]):
              res = sess.run({'loss': loss,
                              'acc': accuracy},
                             feed_dict={
                                 state: [val_ex],
                                 label: [val_set['label'][0][n]]
                             })
              val_loss[j].append(res['loss'])
              val_acc[j].append(res['acc'])
              val_str = '  Timestep ' + str(t) + ' Val acc: '+str(res['acc']) + '\t loss:' + str(res['loss'])
              print (val_str)
              log_file.write(val_str +'\n')

            for n, test_ex in enumerate(test_set['state'][t]):
              res = sess.run({'loss': loss,
                              'acc': accuracy},
                             feed_dict={
                                 state: [test_ex],
                                 label: [test_set['label'][0][n]]
                             })
              test_loss[j].append(res['loss'])
              test_acc[j].append(res['acc'])
              test_str = '  Timestep ' + str(t) + ' Test acc: '+str(res['acc']) + '\t loss:' + str(res['loss'])
              log_file.write(test_str +'\n')

          plt.gcf().clear()
          plt.plot(train_loss[j], label=('Training'))
          plt.plot(val_loss[j], label=('Validation'))
          plt.title('Timestep '+str(t))
          plt.xlabel('Iterations')
          plt.ylabel('Loss')
          plt.legend()
          plt.savefig(os.path.join(experiment_path,'timestep_'+str(t)+'_loss.png'))
          plt.gcf().clear()

          plt.plot(train_acc[j], label=('Training'))
          plt.plot(val_acc[j], label=('Validation'))
          plt.title('Timestep '+str(t))
          plt.xlabel('Iterations')
          plt.ylabel('Accuracy')
          plt.legend()
          plt.savefig(os.path.join(experiment_path,'timestep_'+str(t)+'_acc.png'))
          plt.gcf().clear()

if __name__ == '__main__':
  app.run(main)
