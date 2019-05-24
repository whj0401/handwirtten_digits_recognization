import tensorflow as tf
from tensorflow._api.v1.keras import layers
import read_data

x_train, y_train = read_data.read_train_data()
x_test, y_test = read_data.read_test_data()
num, H, W, _ = x_train.shape

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='valid',
                        data_format='channels_last',
                        activation='relu',
                        use_bias=True,
                        input_shape=(H, W, 1)
                        ),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same',
                        data_format='channels_last',
                        activation='relu',
                        use_bias=True,
                        input_shape=(H-2, W-2, 16)
                        ),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        data_format='channels_last',
                        activation='relu',
                        use_bias=True,
                        input_shape=(H-2, W-2, 32)
                        ),
    tf.keras.layers.Flatten(input_shape=(H-2, W-2, 64)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(0.00005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

model.save('recognizing_model.h5')
