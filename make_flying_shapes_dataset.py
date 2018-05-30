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


"""Convert the flying_shapes png images to a dictionary of images and labels.
"""
import cPickle as pickle
import glob
import numpy as np
from PIL import Image

file_list = glob.glob('flying_shapes/*.png')
x = np.array([np.array(Image.open(f).resize((20, 20))) for f in file_list])
x = x[:, 2:18, 2:18, :3]  # cut borders and alpha channel off

colour_switch = {
    'B': 0,
    'dashed': 1,
    'empty': 2,
    'G': 3,
    'O': 4,
    'P': 5,
    'R': 6,
    'W': 7,
    'Y': 8,
}
shape_switch = {
    'asterisk': 0,
    'circle': 1,
    'cookie': 2,
    'flower': 3,
    'half': 4,
    'pentagon': 5,
    'plus': 6,
    'square': 7,
    'T': 8,
    'triangle': 9,
}

labels = {}
for i, f in enumerate([ff.split('/')[1] for ff in file_list]):
  label_arr = [i]
  for c in colour_switch:
    if f.startswith(c):
      label_arr.append(colour_switch[c])
  for s in shape_switch:
    if s in f:
      label_arr.append(shape_switch[s])
  labels[f] = label_arr
  assert len(label_arr) == 3

dataset = {'images': x, 'labels': labels}
pickle.dump(dataset, open('flying_shapes2.pkl', 'wb'))
