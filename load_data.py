import tensorflow as tf
import pdb

"""Function to preprocess a single image"""
def preprocess_image(image, shape):
  (y, x) = shape
  # We want all images to be of the same size
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, y, x)
  #Convert [0, 255] to [0, 1]
  rescaled_image = tf.cast(cropped_image, tf.float32)/255
  mean, var = tf.nn.moments(rescaled_image, axes=[0, 1])
  rescaled_image = (rescaled_image-mean)/tf.sqrt(var)
  return rescaled_image

"""Function to load in a single image"""
def read_image(filename_queue, imgShape):
  # Read an entire image file at once
  image_reader = tf.WholeFileReader()
  filename, image_file = image_reader.read(filename_queue)
  # If the image has 1 channel (grayscale) it will broadcast to 3
  image = tf.image.decode_image(image_file, channels=3)
  cropped_image = preprocess_image(image, [imgShape[0], imgShape[1]])
  return [filename, cropped_image]

def createQueueReader(file_location, batch_size, imgShape, num_read_threads=1, min_after_dequeue=0, seed=1234):
    capacity = min_after_dequeue + (num_read_threads + 1) * batch_size
    shuffle_images = False

    filenames = tf.constant([string.strip()
      for string
      in open(file_location, "r").readlines()])

    ## OPTION 1 - Using TF Helper Functions
    fi_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_images, seed=seed, capacity=capacity)
    data_list = [read_image(fi_queue, imgShape) for _ in range(num_read_threads)]
    filename_batch, image_batch = tf.train.batch_join(data_list, batch_size=batch_size, capacity=capacity, shapes=[[], imgShape])

    # Must initialize local variables as well as global to init epoch counter
    # in tf.train.string_input_producer
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    return(filename_batch, image_batch)

if __name__ == "__main__":
    file_location = "/media/tbell/datasets/natural_images.txt"
    batch_size = 1
    imgShape = [256, 256, 3]
    #num_read_threads = 1
    #min_after_dequeue = 0
    #seed = 1234
    (filename_batch, image_batch) = createQueueReader(file_location, batch_size, imgShape=imgShape)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    file_idx = 0
    with tf.Session() as sess:
      sess.run(init_op)
      # Coordinator manages threads, checks for stopping requests
      coord = tf.train.Coordinator()
      enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
      #file_list = sess.run(filenames)
      #print(file_list)
      #print("\n-----\n")
      with coord.stop_on_exception():
        while not coord.should_stop():
          try:
            fname, data = sess.run([filename_batch, image_batch])
            pdb.set_trace()
            print(str(file_idx)+" "+str(fname)+"\t"+str(data.shape)+"\n")
            file_idx+=1
          except tf.errors.OutOfRangeError:
            coord.request_stop()
      coord.request_stop()
      coord.join(enqueue_threads)

