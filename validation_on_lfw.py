from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import lfw

def main(args):
  batch_size = 100
  with tf.Grapy().as_default():
    with tf.Session() as sess:
      pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

      paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
      
      print('Loading feature extraction model')
      facenet.load_model('./pre-trained/20170512-110547')
      # Get input and output tensors
      images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
      embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
      phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
      embedding_size = embeddings.get_shape()[1]
      print('done')

  for i in range(batch_size):
