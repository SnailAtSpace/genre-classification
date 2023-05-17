import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from keras import layers

tf.config.run_functions_eagerly(True)
batch_size = 22050 * 5
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
dtype_used = tf.float16

model = keras.models.load_model("/home/snail/wife")

my_cool_track = tf.reshape(tfio.audio.AudioIOTensor("/home/snail/genres/reggae/reggae.00001.wav").to_tensor(), (1, -1, 1))
result = tf.zeros([1, 10])
for i in range(int(my_cool_track.shape[1] / batch_size)):
    batch = my_cool_track[:, i * batch_size: (i + 1) * batch_size, :]
    output = model(batch, training=False)  # what a mess
    result += output
    # print(output)
result = genres[tf.math.argmax(tf.nn.softmax(tf.squeeze(result)))]
print(result)