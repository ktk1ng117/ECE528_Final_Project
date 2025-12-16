import tensorflow as tf

keras_model_path = "efficientnet/model_EfficientNetb0.keras"
tflite_model_path = "efficientnet/efficientnet_model_quantized_converted.tflite"

loaded_model = tf.keras.models.load_model(keras_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)