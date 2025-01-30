import tensorflow as tf
print("TensorFlow version :", tf.__version__)
print("GPU disponible :", tf.config.list_physical_devices('GPU'))
print("CUDA support :", tf.test.is_built_with_cuda())
print("GPU détecté :", tf.test.gpu_device_name())
