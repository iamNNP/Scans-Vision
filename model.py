import numpy as np
from emnist import extract_training_samples, extract_test_samples
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2


class NNModel:
    __models_dic = {'digits': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                   'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']}

    def __init__(self, model_type, model_file_path):
        self.model_type = model_type
        self._symbols = self.__models_dic[model_type]
        self.model_file_path = model_file_path

    @property
    def model_type(self):
        return self._model_type
    
    @model_type.setter
    def model_type(self, model_type):
        if not(model_type in self.__models_dic):
            raise ValueError('Model type can be only "digits" or "letters".')
        self._model_type = model_type

    @property
    def symbols(self):
        return self._symbols
    
    @property
    def model_file_path(self):
        return self._model_file_path
    
    @model_file_path.setter
    def model_file_path(self, model_file_path):
        if not(isinstance(model_file_path, str)):
            raise ValueError('Model file path type can only be str.')
        self._model_file_path = model_file_path
    

    @staticmethod
    def img_preprocessing(img):
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array.astype('float32') / 255

        return img_array


    def recognize(self, img_array):
        model = load_model(self.model_file_path)
        predicts_arr = model(img_array)
        symbol_index = np.argmax(predicts_arr)
        accuracy = round(np.max(predicts_arr), 5)
        
        if self.model_type == 'digits':
            symbol = self._symbols[symbol_index]
        else:
            symbol = self._symbols[symbol_index-1]

        return symbol, accuracy