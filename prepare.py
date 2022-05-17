

from tensorflow.keras.models import load_model
import numpy as np
import pickle

MODEL_NAME = 'model_writers.h5'
TOKENIZER_NAME = 'tokenizer.pickle'
WIN_SIZE   = 1000                         # Длина отрезка текста (окна) в словах
WIN_HOP    = 100                          # Шаг окна разбиения текста на векторы

def process(text_train, text_test = None):
    model = load_model(MODEL_NAME)
    with open(TOKENIZER_NAME, 'rb') as handle:
        tokenizer = pickle.load(handle)

    seq_train = tokenizer.texts_to_sequences(text_train) 

    if text_test:                           
        seq_test = tokenizer.texts_to_sequences(text_test)
    else:
        seq_test = None                                 

    x_val =  vectorize_sequence(seq_train, WIN_SIZE, WIN_HOP)

    return model.predict(x_val)

def vectorize_sequence(seq_list, win_size, hop):
    class_count = len(seq_list)

    x, y = [], []

    for cls in range(class_count):
        vectors = split_sequence(seq_list[cls], win_size, hop)

        x += vectors

        y += [utils.to_categorical(cls, class_count)] * len(vectors)

    return np.array(x)
