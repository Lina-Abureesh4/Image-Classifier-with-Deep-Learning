import tensorflow as tf

def process_image(image):
    """
    The process_image function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array 
    with shape (224, 224, 3)
    :param image: an image that needs preprocessing
    :type image: numpy array
    :return: a preprocessed image normalized with pixel values in the range 0-1 and resized to shape (224, 224, 3)
    rtype: numpy array
    """
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255

    return image