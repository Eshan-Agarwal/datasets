# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Lint as: python3
"""Video visualizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from IPython import display
from PIL import Image
import numpy as np


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import features as features_lib
from tensorflow_datasets.core import lazy_imports_lib
from tensorflow_datasets.core.visualization import visualizer


def _display_gif(gif_fn, ds:tf.data.Dataset,
                 num_examples:int, width:int,
                 height:int) -> None:
  """Display Video as GIF.

  Args:
    gif_fn: Function which will write GIF's.
    ds: `tf.data.Dataset`. The tf.data.Dataset object to visualize. Examples
      should not be batched. Examples will be consumed in order until
      (rows * cols) are read or the dataset is consumed.
    num_examples: Number of examples to display from video dataset.
    width: width of frame for display GIF.
    height: height of frame for display GIF.
  """
  examples = list(dataset_utils.as_numpy(ds.take(num_examples)))
  for i, ex in enumerate(examples):
    gif_fn(ex)
    with open('sample.gif','rb') as f:
      display.display(display.Image(data=f.read(),format="png",
                                    width=width,height=height))
    tf.io.gfile.remove("sample.gif")


def _write_video_gif(video:np.ndarray, fps:int) -> None:
  """Process the Video and write it as GIF.
  Args:
    video: numpy array of video as image sequence.
    fps: frame per second for display GIF. Default to 10.
  """
  if len(video.shape) != 4:
    raise ValueError(
        'Video dimension should be 4.')
  _,_,_,c = video.shape
  if(c==1):
    video = video.reshape(video.shape[:3])

  # Write GIF using PIL
  video_seq = [Image.fromarray(v_frame).quantize() for v_frame in video]
  video_seq[0].save("sample.gif", save_all=True, append_images=video_seq[1:]\
                    ,loop=0,duration=int((1000)/fps))


class VideoGridVisualizer(visualizer.Visualizer):
  """Visualizer for video datasets."""

  def match(self, ds_info:tfds.core.DatasetInfo) -> bool:
    """See base class.
    Args:
      ds_info: `tfds.core.DatasetInfo` object of the dataset to visualize.

    Returns:
      bool: True if the visualizer can be applied to the dataset.
    """
    # Supervised required a single image key
    video_keys = visualizer.extract_keys(ds_info.features, features_lib.Video)
    return len(video_keys) > 0

  def show(
      self,
      ds_info:tfds.core.DatasetInfo,
      ds:tf.data.Dataset,
      num_examples:int=5,
      video_key:str=None,
      width:int=150,
      height:int=150,
      fps:int=10,
  ) -> None:
    """Display the dataset.

    Args:
      ds_info: `tfds.core.DatasetInfo` object of the dataset to visualize.
      ds: `tf.data.Dataset`. The tf.data.Dataset object to visualize. Examples
        should not be batched. Examples will be consumed in order until
        (rows * cols) are read or the dataset is consumed.
      num_examples: Number of examples to display from video dataset. Default is 5
      video_key: `string`, name of the feature that contains the video. If not
         set, the system will try to auto-detect it.
      width: width of frame for display GIF.Default to 150.
      height: height of frame for display GIF.Default to 150.
      fps: frame per second for display GIF. Default to 10.
    """
    # Extract the video key automatically.
    if not video_key:
      video_keys = visualizer.extract_keys(ds_info.features, features_lib.Video)
      video_key = video_keys[0]

    def make_cell_fn(ex:np.ndarray) -> None:
      if not isinstance(ex, dict):
        raise ValueError(
            '{} requires examples as `dict`, with the same '
            'structure as `ds_info.features`. It is currently not compatible '
            'with `as_supervised=True`. Received: {}'.format(
                type(self).__name__, type(ex)))

      _write_video_gif(ex[video_key], fps)
    # Display GIF
    _display_gif(make_cell_fn, ds, num_examples, width, height)
