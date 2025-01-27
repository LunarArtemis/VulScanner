import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch

class BLSTM:
    def __init__(self, data, name="", batch_size=64, epochs=20, max_length=128):
        self.data = data
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length

        # Load CodeBERT tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.embedding_model = AutoModel.from_pretrained("microsoft/codebert-base")

        # Preprocess data
        self._preprocess_data()

        # Build and compile model
        self.model = self._build_model()

    def _preprocess_data(self):
        self.data['label'] = self.data['label'].astype(int)
        
        embeddings = []
        for code in self.data['code']:
            inputs = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)

        self.X = np.array(embeddings).squeeze()
        self.y = self.data['label'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=12
        )

    def _build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
    
        model.compile(optimizer=Adam(learning_rate=0.02), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    
        return model

    def train(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.name + "_best_model.weights.keras", save_best_only=True, monitor='val_loss')
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping, model_checkpoint]
        )

        self._plot_learning_curve(history)

    def _plot_learning_curve(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.ylim(0, 1)  # Set y-axis limit for accuracy
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.ylim(0, 2)  # Set y-axis limit for loss
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

    def test(self):
        self.model.load_weights(self.name + "_best_model.weights.keras")
        results = self.model.evaluate(self.X_test, self.y_test)
        print("Test loss:", results[0])
        print("Test accuracy:", results[1])
        print("Test AUC:", results[2])

    def predict_code(self, file_path):
        with open(file_path, 'r') as f:
            code_lines = f.readlines()

        vulnerable_lines = []
        predictions = []

        for idx, line in enumerate(code_lines):
            line = line.strip()
            if not line:
                continue
            inputs = self.tokenizer([line], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings = np.expand_dims(embeddings, axis=1)

            prediction = self.model.predict(embeddings)
            predicted_class = 1 if prediction > 0.5 else 0

            predictions.append(predicted_class)
            if predicted_class == 1:
                vulnerable_lines.append(idx)

        result = "Vulnerable" if 1 in predictions else "Non-Vulnerable"
        vulnerable_code_lines = [f"{idx + 1}: {code_lines[i].strip()}" for i in vulnerable_lines]

        return result, vulnerable_code_lines
