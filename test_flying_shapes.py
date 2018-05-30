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

"""Test create and print a batch from the flying shapes dataset.
"""
from third_party import dataset


def test_flying_shapes():
  """Wrapper for flying_shapes.py data generator."""

  config = {}
  config['seq_length'] = 10
  config['batch_size'] = 2
  config['image_size'] = 600
  config['num_digits'] = 3
  config['step_length'] = 0.5
  config['digit_size'] = 180
  config['frame_size'] = (config['image_size']**2) * 3
  config['file_path'] = 'flying_shapes.npy'

  data_generator = dataset.FlyingShapesDataHandler(config)
  x, bboxes = data_generator.GetUnlabelledBatch()
  data_generator.DisplayData(x, bboxes)

  x2, bboxes2 = data_generator.GetLabelledBatch()
  data_generator.DisplayData(x2, bboxes2)
