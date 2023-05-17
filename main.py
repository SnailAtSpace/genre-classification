import tensorflow as tf
import tensorflow.python.framework.errors_impl
import tensorflow_io as tfio
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow import python as tfpy

tf.config.run_functions_eagerly(True)
batch_size = 22050 * 1
file_size = 661500
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
dtype_used = tf.float16


class CCNNModel(keras.Model):
    def train_step(self, data):
        audio_tensor, target = data
        audio = audio_tensor  # 8*file_size*1, need to convert to (file_size/batch_size)*8*batch_size*1
        with tf.GradientTape() as tape:
            result = tf.zeros([audio_tensor.shape[0], 10])
            for i in range(int(file_size / batch_size)):
                batch = audio[:, i * batch_size: (i + 1) * batch_size, :]
                output = self(batch, training=True)
                result += output
            result = tf.nn.softmax(result)
            loss = self.compiled_loss(target, result, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(target, result)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        audio_tensor, target = data
        audio = audio_tensor
        result = tf.zeros([audio_tensor.shape[0], 10])
        for i in range(int(file_size / batch_size)):
            output = self(audio[:, i * batch_size:(i + 1) * batch_size, :], training=False)  # what a mess
            result += output
            loss = self.compiled_loss(target, result, regularization_losses=self.losses)
        result /= tf.nn.softmax(result)
        self.compiled_metrics.update_state(target, result)
        return {m.name: m.result() for m in self.metrics}


def make_raw_model():
    modelinput = keras.Input(shape=(batch_size, 1))
    modelinput = layers.regularization.GaussianNoise(0.5)(modelinput)
    x = layers.Conv1D(8, 441, 7, activation=tf.nn.relu, padding="same")(modelinput)
    x = layers.MaxPool1D(7, 7)(x)
    x = layers.Conv1D(16, 294, 5, activation=tf.nn.relu, padding="same")(x)
    x = layers.MaxPool1D(5, 5)(x)
    x = layers.Conv1D(32, 21, 2, activation=tf.nn.relu, padding="same")(x)
    x = layers.GlobalMaxPool1D()(x)
    output = layers.Dense(10, activation=tf.nn.leaky_relu)(x)
    return modelinput, output


def make_compiled_model():
    i, o = make_raw_model()
    model = CCNNModel(inputs=i, outputs=o, name="model")
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=1e-6, beta_2=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


def unify(inp: tf.Tensor, size):
    inp.shape.assert_has_rank(1)
    if inp.shape.as_list()[0] >= size:
        return inp[:size]
    else:
        return tf.pad(inp, [[0, size - inp.shape.as_list()[0]]])


def get_audio_files_as_dataset() -> tf.data.Dataset:
    prefix = "/home/snail/genres/"
    files = tf.data.Dataset.from_tensor_slices((tf.zeros([1, file_size, 1], dtype=dtype_used), [[0]]))
    for g in range(10):
        for i in range(2, 100):
            label = genres[g]
            if label == "jazz" and i == 54:
                continue
            t = tfio.audio.AudioIOTensor(prefix + f"{label}/{label}.{i:05d}.wav").to_tensor()
            # (<samples>,1)->(<samples>,)->(file_size,)
            t = tf.cast(t, dtype_used) / 32768
            t = tf.reshape(unify(tf.squeeze(t), file_size), (file_size, 1))
            file = tf.data.Dataset.from_tensors(t)
            label = tf.data.Dataset.from_tensors(tf.constant([g]))
            files = files.concatenate(tf.data.Dataset.zip((file, label)))
    return files.skip(1)  # this is stupid


def get_eval_samples():
    prefix = "/home/snail/genres/"
    files = tf.convert_to_tensor(tf.zeros([1, file_size, 1], dtype=dtype_used))
    for g in range(10):
        for i in range(2):
            label = genres[g]
            t = tfio.audio.AudioIOTensor(prefix + f"{label}/{label}.{i:05d}.wav").to_tensor()
            t = tf.reshape(unify(tf.squeeze(tf.cast(t, dtype_used) / 32768), file_size), (1, file_size, 1))
            files = tf.concat((files, t), 0)
    return files[1:], tf.reshape(tf.repeat(tf.convert_to_tensor(range(10)), repeats=[2]*10), (20, 1))


audio_dataset = get_audio_files_as_dataset()
audio_dataset = audio_dataset.shuffle(len(audio_dataset)).batch(32)
print("dataset length is:", len(audio_dataset))
model = make_compiled_model()
print("ALRIGHT IT'S TRAINING TIME HERE WE GO")
history = model.fit(
    audio_dataset,
    epochs=100,
    validation_data=get_eval_samples()
)

model.save("model")
