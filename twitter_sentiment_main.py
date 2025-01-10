import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import dump

data = pd.read_csv('Twitter_Data.csv')

# Convert 'clean_text' column to strings
data['selected_text'] = data['selected_text'].astype(str)
data['selected_text'] = data['selected_text'].str.replace('[^a-zA-Z\s]', '').str.lower()


# splitting the data into independent and dependent attributes

X = data['selected_text']
y = data['sentiment']

unique_sentiments = y.unique()

y = y.replace({'negative':0, 'positive':2,'neutral':1})

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


print(f"X train shape {X_train.shape}")
print(f"y train shape {y_train.shape}")
print(f"X test shape {X_test.shape}")
print(f"y test shape {y_test.shape}")


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')


label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)



# One-hot encode labels
num_classes = len(unique_sentiments)
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)


model = tf.keras.Sequential([
    Embedding(input_dim= 5000, output_dim = 100),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')

])

model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_pad, y_train_onehot, epochs=5,batch_size=32, validation_data=(X_test_pad, y_test_onehot))

model.save('sentiment_model.h5')
dump(tokenizer, 'tokenizer.joblib')
