import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TensorFlow:
    PATH = os.path.join('/content/drive/My Drive/Deep_Learning_Colab/Covid19TensorFlowClassifier', 'ImagesDataset')
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    epochs = 8

    def __init__(self):
        total_train, total_val = self.load_image_dataset()
        train_data_gen, batch_size, sample_training_images, val_data_gen = self.image_pre_processing()
        self.plot_images(sample_training_images[:5])
        model = self.setup_convolutional_neural_network()
        self.train_convolutional_neural_network(model, batch_size, train_data_gen, total_train, val_data_gen, total_val)

    def load_image_dataset(self) -> tuple:
        train_butterfly_dir = os.path.join(self.train_dir, 'covid')
        train_owl_dir = os.path.join(self.train_dir, 'nocovid')
        validation_butterfly_dir = os.path.join(self.validation_dir, 'covid')
        validation_owl_dir = os.path.join(self.validation_dir, 'nocovid')

        num_butterfly_tr = len(os.listdir(train_butterfly_dir))
        num_owl_tr = len(os.listdir(train_owl_dir))

        num_butterfly_val = len(os.listdir(validation_butterfly_dir))
        num_owl_val = len(os.listdir(validation_owl_dir))

        total_train = num_butterfly_tr + num_owl_tr
        total_val = num_butterfly_val + num_owl_val

        print('total training covid images:', num_butterfly_tr)
        print('total training no-covid images:', num_owl_tr)

        print('total validation covid images:', num_butterfly_val)
        print('total validation no-covid images:', num_owl_val)
        print("--")
        print("Total training images:", total_train)
        print("Total validation images:", total_val)
        return total_train, total_val

    def image_pre_processing(self) -> tuple:
        batch_size = 16  # Warning: batch_size < train_data
        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=self.train_dir,
                                                                   shuffle=True,
                                                                   target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                   class_mode='binary')
        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                      directory=self.validation_dir,
                                                                      target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                      class_mode='binary')
        sample_training_images, _ = next(train_data_gen)
        return train_data_gen, batch_size, sample_training_images, val_data_gen

    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(15, 15))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def setup_convolutional_neural_network(self):
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()
        return model

    def train_convolutional_neural_network(self, model, batch_size, train_data_gen, total_train, val_data_gen,
                                           total_val):
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=self.epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(self.epochs)
