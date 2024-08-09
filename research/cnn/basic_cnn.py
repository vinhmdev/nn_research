import numpy as np
import keras

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class BasicCNN:

    def __init__(self):
        self.cnn = keras.models.Sequential()
        self.training_set = keras.preprocessing.image_dataset_from_directory(
            directory='dataset/training_set/',
            image_size=(64, 64),
            batch_size=32,
            label_mode='categorical',
            # label_mode='binary',
        )
        self.test_set = keras.preprocessing.image_dataset_from_directory(
            directory='dataset/test_set/',
            image_size=(64, 64),
            batch_size=32,
            label_mode='categorical',
            # label_mode='binary',
        )

    def create_cnn(self, save_to: str = 'cnn.keras'):
        self.cnn = keras.models.Sequential()

        self.cnn.add(layer=keras.layers.RandomFlip(mode='horizontal'))
        self.cnn.add(layer=keras.layers.RandomRotation(factor=0.1))
        self.cnn.add(layer=keras.layers.RandomBrightness(factor=0.1))
        self.cnn.add(layer=keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2))
        self.cnn.add(layer=keras.layers.RandomCrop(height=60, width=60, input_shape=(64, 64, 3)))

        self.cnn.add(layer=keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3), ))
        self.cnn.add(layer=keras.layers.MaxPool2D(pool_size=2, strides=2, ))
        self.cnn.add(layer=keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(layer=keras.layers.MaxPool2D(pool_size=2, strides=2, ))

        self.cnn.add(layer=keras.layers.Flatten())

        self.cnn.add(layer=keras.layers.Dense(units=128, activation='relu'))

        # self.cnn.add(layer=keras.layers.Dense(units=1, activation='sigmoid'))
        # self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.cnn.add(layer=keras.layers.Dense(units=len(self.training_set.class_names), activation='softmax'))
        self.cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=25)

        keras.models.save_model(filepath=save_to, model=self.cnn, )

    def load_cnn(self, filepath: str = 'cnn.keras'):
        self.cnn = keras.models.load_model(filepath=filepath)

    def predict(self, path: str):
        image_test = keras.preprocessing.image.load_img(
            path=path,
            target_size=(64, 64)
        )
        image_test = keras.preprocessing.image.img_to_array(img=image_test)
        image_test = np.expand_dims(image_test, axis=0)

        result = self.cnn.predict(image_test)
        result = np.array(result).reshape((-1))

        infos = zip(self.training_set.class_names, result)

        return dict(infos), result


# TESTING <==================================================>
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

basic_cnn = BasicCNN()


# basic_cnn.create_cnn()  # run it if re-create cnn.keras file


def test(path: str):
    basic_cnn.load_cnn()
    infos, result = basic_cnn.predict(path=path)

    im = mpimg.imread(path)

    print(f'cat_or_dog_1: {infos=}')

    text = []
    for i, (k, v) in enumerate(infos.items()):
        print(i, k, v)
        text.append(f'=> {k}: {v * 100} %')
    t = plt.text(
        0,
        0,
        '\n'.join(text),
    )
    t.set_bbox(dict(facecolor='white', edgecolor='white', linewidth=1))

    plt.imshow(im)
    plt.show()


test('dataset/single_prediction/cat_or_dog_1.jpg')
test('dataset/single_prediction/cat_or_dog_2.jpg')
