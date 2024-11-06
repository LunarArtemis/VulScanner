from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import pickle
import argparse

warnings.filterwarnings("ignore")

class BLSTM:
    def __init__(self, data, name="", batch_size=64, epochs=4):
        self.data = data
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs

        # Preprocess data
        self._preprocess_data()

        # Build and compile model
        self.model = self._build_model()

    def _preprocess_data(self):
        # Preprocess labels
        label_encoder = LabelEncoder()
        self.data['label'] = label_encoder.fit_transform(self.data['label'])
        self.num_classes = len(self.data['label'].unique())  # Number of classes

        # Tokenize code snippets
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.data['code'])
        sequences = tokenizer.texts_to_sequences(self.data['code'])
        self.X = pad_sequences(sequences)

        # Convert labels to one-hot encoding
        self.y = np.eye(self.num_classes)[self.data['label'].values]

        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Save tokenizer and label encoder
        with open('tokenizer.pkl', 'wb') as file:
            pickle.dump(tokenizer, file)
        with open('label_encoder.pkl', 'wb') as file:
            pickle.dump(label_encoder, file)

        # Check data shapes
        print(f'Vocabulary size: {len(tokenizer.word_index) + 1}')
        print('Padded sequences shape:', self.X.shape)
        print(f'Number of classes: {self.num_classes}')
        print(f'X_train shape: {self.X_train.shape}')
        print(f'X_test shape: {self.X_test.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'y_test shape: {self.y_test.shape}')

    def _build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.5))
        model.add(LSTM(300))
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=Adamax(learning_rate=0.002),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        # Fit model
        self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        self.model.save_weights(self.name + "_model.h5")

    def test(self):
        # Load model weights
        self.model.load_weights(self.name + "_model.h5")
        results = self.model.evaluate(self.X_test, self.y_test)
        print("Test loss:", results[0])
        print("Test accuracy:", results[1])

        # Predict and evaluate
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
        print('Classification Report:\n', classification_report(y_true, y_pred))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test a BLSTM model for vulnerability detection.",
        epilog="Example usage:\n"
               "  python3 vulchk.py -d your_dataset.csv -m train\n"
               "  python3 vulchk.py -d your_dataset.csv -m test\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-d', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('-n', type=str, default="blstm_model", help='Name of the model for saving weights.')
    parser.add_argument('-bs', type=int, default=64, help='Batch size for training.')
    parser.add_argument('-e', type=int, default=4, help='Number of epochs for training.')
    parser.add_argument('-m', type=str, choices=['train', 'test'], required=True, help='Mode to run: train or test.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load the dataset
    data = pd.read_csv(args.data_path)

    # Create an instance of the BLSTM class
    blstm = BLSTM(data, name=args.model_name, batch_size=args.batch_size, epochs=args.epochs)

    # Train or test the model based on the provided mode
    if args.mode == 'train':
        blstm.train()
    elif args.mode == 'test':
        blstm.test()