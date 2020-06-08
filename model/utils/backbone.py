import glob, os
import tensorflow as tf
from model.utils import label_map_util

def set_model(model):
  # Set model
  model_name = model

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  path_to_ckpt = model_name + '/frozen_inference_graph.pb'
  print(path_to_ckpt)

  # List of the strings that is used to add correct label for each box.
  path_to_labels = os.path.join('model/data', 'labelmap.pbtxt')

  num_classes = 8

  # Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Loading label map
  label_map = label_map_util.load_labelmap(path_to_labels)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  return detection_graph, category_index
