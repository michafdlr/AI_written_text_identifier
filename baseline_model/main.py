import pandas as pd
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from dotenv import load_dotenv

load_dotenv()
np.random.seed(42)
tf.random.set_seed(42)

def get_data_path():
    path_data = Path(os.getenv("PATH_DATA"))

    return path_data

def load_data(path_data, n_rows_train=200_000, n_rows_val_test=10_000):

    final_data = {}

    for split, n_rows in zip(["train", "valid", "test"], [n_rows_train, n_rows_val_test, n_rows_val_test]):
        data = pd.read_csv(path_data / f"xl-1542M-k40.{split}.csv", usecols=["text"], nrows=n_rows // 2)
        data["AI"] = 1
        file_path = path_data / f"webtext.{split}.csv"
        temp = pd.read_csv(file_path, usecols=["text"], nrows=n_rows // 2)
        temp["AI"] = 0
        final_data[split] = pd.concat([temp, data]).sample(frac=1).reset_index(drop=True)

    return final_data

def clean_text(text):

    def remove_punctuation(text):
        return ''.join([l for l in text if not l in string.punctuation])

    def lower(text):
        return text.lower()

    def remove_numbers(text):
        return ''.join([char for char in text if not char.isdigit()])

    def tokenize(text):
        return word_tokenize(text)

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))

        return [word for word in text if not word in stop_words]

    def lemmatize_text(text):

        for pos in ["v", "n", "a", "r", "s"]:
            text = [WordNetLemmatizer().lemmatize(word, pos=pos) for word in text]

        return ' '.join(text)

    text = remove_numbers(lower(remove_punctuation(text)))
    tokens = tokenize(text)
    tokens_clean = remove_stopwords(tokens)
    text_clean = lemmatize_text(tokens_clean)

    return text_clean.strip()
def prepare_data_lstm(X_train, X_val, X_test):

    max_words = 2000
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train.text)
    X_train_seq = tokenizer.texts_to_sequences(X_train.text)
    X_val_seq = tokenizer.texts_to_sequences(X_val.text)
    X_test_seq = tokenizer.texts_to_sequences(X_test.text)
    max_len = 100
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding="post")
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

    return X_train_padded, X_val_padded, X_test_padded

def train_lstm(X_train_padded, y_train, X_val_padded, y_val):
    num_classes = 2
    inputs = Input(shape=(100,))
    x = Embedding(input_dim=2000, output_dim=32)(inputs)
    x = LSTM(32)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train_padded, y_train,
              batch_size=16, epochs=5,
              validation_data=(X_val_padded, y_val),
              callbacks=[early_stopping])

    return model

def evaluate_lstm(model, X_test_padded, y_test):
    accuracy = model.evaluate(X_test_padded, y_test)[1]
    print("LSTM accuracy:", accuracy)

def plot_learning_curve_tf(model, X_train_padded, y_train, X_val_padded, y_val):
    train_scores = []
    val_scores = []
    train_sizes = np.linspace(0.1, 1, 5)

    for size in train_sizes:
        slice_size = int(X_train_padded.shape[0] * size)
        X_train_slice = X_train_padded[:slice_size, :]
        y_train_slice = y_train[:slice_size]
        model.fit(X_train_slice, y_train_slice, batch_size=16, epochs=5, verbose=0)
        train_score = model.evaluate(X_train_slice, y_train_slice, verbose=0)[1]
        val_score = model.evaluate(X_val_padded, y_val, verbose=0)[1]
        train_scores.append(train_score)
        val_scores.append(val_score)

    plt.plot(train_sizes, train_scores, label="Training Score")
    plt.plot(train_sizes, val_scores, label="Validation Score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    path_data = get_data_path()
    data_dict = load_data(path_data)
    data_train = data_dict["train"]
    data_val = data_dict["valid"]
    data_test = data_dict["test"]
    data_train["text"] = data_train["text"].apply(clean_text)
    data_val["text"] = data_val["text"].apply(clean_text)
    data_test["text"] = data_test["text"].apply(clean_text)
    X_train = data_train[["text"]]
    X_val = data_val[["text"]]
    X_test = data_test[["text"]]
    y_train = data_train["AI"]
    y_val = data_val["AI"]
    y_test = data_test["AI"]
    X_train_padded, X_val_padded, X_test_padded = prepare_data_lstm(X_train, X_val, X_test)

    lstm_model = train_lstm(X_train_padded, y_train, X_val_padded, y_val)
    evaluate_lstm(lstm_model, X_test_padded, y_test)
    plot_learning_curve_tf(lstm_model, X_train_padded, y_train, X_val_padded, y_val)

if __name__ == "__main__":
    main()
