import tensorflow as tf

"""Function to preprocess a single image"""
def preprocess_image(image, shape):
  (y, x) = shape
  # We want all images to be of the same size
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, y, x)
  return cropped_image

"""Function to load in a single image"""
def read_image(filename_queue):
  # Read an entire image file at once
  image_reader = tf.WholeFileReader()
  filename, image_file = image_reader.read(filename_queue)
  # If the image has 1 channel (grayscale) it will broadcast to 3
  image = tf.image.decode_image(image_file, channels=3)
  cropped_image = preprocess_image(image)
  return [filename, cropped_image]

file_location = "/media/tbell/datasets/natural_images.txt"
num_epochs = 1
batch_size = 1
num_read_threads = 1
min_after_dequeue = 0
seed = 1234
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size
shuffle_images = False

filenames = tf.constant([string.strip()
  for string
  in open(file_location, "r").readlines()])

## OPTION 1 - Using TF Helper Functions
fi_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle_images, seed=seed, capacity=capacity)
data_list = [read_image(fi_queue) for _ in range(num_read_threads)]
filename_batch, image_batch = tf.train.batch_join(data_list, batch_size=batch_size, capacity=capacity, shapes=[[], [256, 256, 3]])

# Must initialize local variables as well as global to init epoch counter
# in tf.train.string_input_producer
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

file_idx = 0
with tf.Session() as sess:
  sess.run(init_op)
  # Coordinator manages threads, checks for stopping requests
  coord = tf.train.Coordinator()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
  file_list = sess.run(filenames)
  print(file_list)
  print("\n-----\n")
  with coord.stop_on_exception():
    while not coord.should_stop():
      try:
        fname, data = sess.run([filename_batch, image_batch])
        print(str(file_idx)+" "+str(fname)+"\t"+str(data.shape)+"\n")
        file_idx+=1
      except tf.errors.OutOfRangeError:
        coord.request_stop()
  coord.request_stop()
  coord.join(enqueue_threads)
