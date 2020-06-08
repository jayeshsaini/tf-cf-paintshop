import tensorflow as tf
import cv2
import numpy as np
from model.utils import visualization_utils as vis_util

def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):     
         
    counting_mode = "..."
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

        

            input_frame = cv2.imread(input_video)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

    # insert information text to video frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Visualization of the results of a detection.        
    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                              1,
                                                                                              is_color_recognition_enabled,
                                                                                              np.squeeze(boxes),
                                                                                              np.squeeze(classes).astype(np.int32),
                                                                                              np.squeeze(scores),
                                                                                              category_index,
                                                                                              use_normalized_coordinates=True,
                                                                                              line_thickness=4,
                                                                                              min_score_thresh=.8)
        
    #  To print results with labelled image
    print ("\nFound Following objects in image:\n")
    print(counting_mode)
                        
    # Added to save the labelled image
    img_name = "./model/output_images/image.jpg"
    cv2.imwrite(img_name, input_frame)
    print("Image Saved")
            
    return counting_mode       

def single_image_target_counting(input_video, detection_graph, category_index, is_color_recognition_enabled,targeted_object, fps, width, height):     
         
    counting_mode = "..."

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

        
            input_frame = cv2.imread(input_video)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

    # insert information text to video frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Visualization of the results of a detection.        
    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_target_image_array(1,input_frame,
                                                                                              1,
                                                                                              is_color_recognition_enabled,
                                                                                              np.squeeze(boxes),
                                                                                              np.squeeze(classes).astype(np.int32),
                                                                                              np.squeeze(scores),
                                                                                              category_index,
                                                                                              targeted_objects=targeted_object,
                                                                                              use_normalized_coordinates=True,
                                                                                              line_thickness=4,
                                                                                              min_score_thresh=.8)
        
    #  To print results with labelled image
    print ("\nFound Following objects in image:\n")
    print(counting_mode)
        
    # Added to save the labelled image
    img_name = "./model/output_images/wrong_image.jpg"
    cv2.imwrite(img_name, input_frame)
    print("Image Saved")
            
    return counting_mode, input_frame