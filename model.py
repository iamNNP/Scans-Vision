import numpy as np
from emnist import extract_training_samples, extract_test_samples
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2


# TEST MODEL №2 (VAL ACC 87%, TEST ACC 85%)
# train_images, train_labels = extract_training_samples('byclass')
# test_images, test_labels = extract_test_samples('byclass')

# max_samples = 60000
# train_images, train_labels = train_images[:max_samples], train_labels[:max_samples]
# max_samples = 10000
# test_images, test_labels = test_images[:max_samples], test_labels[:max_samples]

# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=62)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=62)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Dropout(0.25),

#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Dropout(0.25),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(62, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.1)

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test accuracy: {test_acc:.3f}')

# model.save('emnist_model_v2.h5')



# TEST MODEL №3 (VAL ACC 89%, TEST ACC 83%)
# train_images, train_labels = extract_training_samples('byclass')
# test_images, test_labels = extract_test_samples('byclass')

# num_classes = 62

# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# num_train_samples = 60000
# train_images, train_labels = train_images[:num_train_samples], train_labels[:num_train_samples]
# test_images, test_labels = test_images[:num_train_samples], test_labels[:num_train_samples]

# model.fit(train_images, train_labels, epochs=15, batch_size=64)

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test accuracy: {test_acc:.3f}')

# model.save('emnist_model_v3.h5')



# train_images, train_labels = extract_training_samples('letters')
# test_images, test_labels = extract_test_samples('letters')

# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# num_classes = 52
# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test accuracy: {test_acc:.3f}')

# model.save('letters_model.h5')



# train_images, train_labels = extract_training_samples('digits')
# test_images, test_labels = extract_test_samples('digits')

# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# model.save('digits_model.h5')
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test accuracy: {test_acc:.3f}')


class NNModel:
    __model_types = {'digits': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                   'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']}

    def __init__(self, model_type, model_file_path):
        self.__validate_model_type(model_type)
        self.__validate_model_file(model_file_path)
        self.__model_type = model_type
        self.__symbols = self.__model_types[model_type]
        self.__model_file_path = model_file_path


    @classmethod
    def __validate_model_type(cls, model_type):
        if model_type not in cls.__model_types:
            raise AttributeError('Incorrect model type. Only "digits" and "letters" model types are available')
        
    
    @staticmethod
    def __validate_model_file(model_file_path):
        if not(isinstance(model_file_path, str) and model_file_path.endswith('.h5')):
            raise AttributeError('Incorrect model file path. File path string must end with .h5 extension')


    @staticmethod
    def img_preprocessing(img):
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array.astype('float32') / 255

        return img_array


    def recognize(self, img_array):
        model = load_model(self.__model_file_path)
        predicts_arr = model(img_array)
        symbol_index = np.argmax(predicts_arr)
        accuracy = round(np.max(predicts_arr), 5)
        
        if self.__model_type == 'digits':
            symbol = self.__symbols[symbol_index]
        else:
            symbol = self.__symbols[symbol_index-1]

        return symbol, accuracy


    # @property
    # def model_type(self):
    #     return self.__model_type

    # @model_type.setter
    # def model_type(self, model_type):
    #     self.__model_type = model_type
    
    # @property
    # def model_file_path(self):
    #     return self.__model_file_path

    # @model_file_path.setter
    # def model_file_path(self, model_file_path):
    #     self.__model_file_path = model_file_path

    

    # def recognize_letter(img_array, model):
    #     letters_model = load_model(model)
    #     letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    #     predicts_arr = letters_model(img_array)
    #     letter_index = np.argmax(predicts_arr)
    #     accuracy = round(np.max(predicts_arr), 5)
    #     letter = letters[letter_index-1]

    #     return letter, accuracy


    # def recognize_digit(img_array, model):
    #     digits_model = load_model(model)
    #     digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    #     predicts_arr = digits_model(img_array)
    #     digit_index = np.argmax(predicts_arr)
    #     accuracy = round(np.max(predicts_arr), 5)
    #     digit = digits[digit_index]

    #     return digit, accuracy
