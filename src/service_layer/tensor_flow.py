import base64
import pathlib
from io import BytesIO

import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.lite.python.schema_py_generated import np
from struct import unpack
import os

class TensorFlow:

    dataset_url = "https://dmcorrales.com/backup.tgz"
    data_dir = tf.keras.utils.get_file('dataset', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    batch_size = 32
    img_height = 180
    img_width = 180

    def init(self, image_64: str):
        print(self.data_dir)
        image_count = len(list(self.data_dir.glob('*/*.png')))
        train_ds = self.train_dataset()
        class_names = train_ds.class_names
        val_ds = self.val_dataset()
        print(class_names)
        self.plot_figures(train_ds, class_names)
        self.performance(train_ds, val_ds)
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        self.normalize_data(train_ds, normalization_layer)
        model = self.create_model()
        self.train(model, train_ds, val_ds)
        return self.predict(model, class_names, image_64)

    def train_dataset(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        return train_ds

    def val_dataset(self):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        return val_ds

    def plot_figures(self, train_ds, class_names):
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
                plt.show()

    def performance(self, train_ds, val_ds):
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def normalize_data(self, train_ds, normalization_layer):
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))

    def create_model(self):
        num_classes = 5

        model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def train(self, model, train_ds, val_ds):
        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        model.summary()

    def predict(self, model, class_names, image_64):
        sunflower_url = image_64.encode("ascii")
        decoded = base64.decodebytes(sunflower_url)

        imgdata = base64.b64decode(image_64)
        filename = '/home/dcorrales/Descargas/some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)

        img = keras.preprocessing.image.load_img(
            filename , target_size=(self.img_height, self.img_width)
        )

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        return self._get_price_by_type(class_names[np.argmax(score)])

    def _get_price_by_type(self, type_product: str):
        product_list = {
            "banana": {"name": "Banano", "price": 500, "description": "Delicioso para comer con galletas y leche", "stock": 5},
            "choclitos": {"name": "Choclitos", "price": 1800, "description": "Para pasar un buen rato con los amigos", "stock": 160},
            "nevera": {"name": "Nevera", "price": 900000, "description": "¡Manten tus productos a salvo!", "stock": 160},
            "poker": {"name": "Cerveza Poker", "price": 1600, "description": "Descubre el sabor y frescura de Cerveza Poker, la mejor compañía para compartir momentos inolvidables entre amigos ", "stock": 5},
            "silla": {"name": "Silla", "price": 100000,
                      "description": "Sientate cómodo", "stock": 5}
        }

        return product_list.get(type_product)

