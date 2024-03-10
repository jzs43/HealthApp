import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional
from sklearn.model_selection import train_test_split

sentences = ['I need tips for improving my sleep quality.',
 "I'm looking for support groups for transgender individuals.",
 'Can you tell me more about birth control options?',
 'I have concerns about unprotected sex.',
 'I have questions about hormone replacement therapy.',
 'I need advice on gender-affirming surgeries.',
 "I'm curious about healthy eating habits.",
 'Can you tell me more about birth control options?',
 'Can you provide resources for dealing with anxiety?',
 'I need tips for improving my sleep quality.',
 'I need tips for improving my sleep quality.',
 "I'm looking for support groups for transgender individuals.",
 'Can you provide resources for dealing with anxiety?',
 "I've been feeling really down lately.",
 'Can you recommend some good exercises for back pain?',
 'Can you provide information on changing my legal gender?',
 "I'm struggling with stress and need some advice.",
 'Can you provide resources for dealing with anxiety?',
 "I'm experiencing symptoms after a sexual encounter.",
 "I'm struggling with stress and need some advice.",
 'I think I might be experiencing symptoms of depression.',
 'I need advice on gender-affirming surgeries.',
 'I need tips for improving my sleep quality.',
 'Can you provide resources for dealing with anxiety?',
 'I need advice on gender-affirming surgeries.',
 'Can you provide information on changing my legal gender?',
 'I have questions about hormone replacement therapy.',
 'Can you provide information on changing my legal gender?',
 'Can you provide information on changing my legal gender?',
 'I need tips for improving my sleep quality.',
 "I'm looking for information on STDs.",
 'Can you recommend some good exercises for back pain?',
 'I need advice on gender-affirming surgeries.',
 'Can you provide information on changing my legal gender?',
 'I think I might be experiencing symptoms of depression.',
 'Can you recommend some good exercises for back pain?',
 'I have concerns about unprotected sex.',
 'I think I might be experiencing symptoms of depression.',
 "I'm looking for support groups for transgender individuals.",
 "I'm experiencing symptoms after a sexual encounter.",
 "I'm curious about healthy eating habits.",
 "I'm looking for support groups for transgender individuals.",
 "I'm looking for support groups for transgender individuals.",
 'I need advice on gender-affirming surgeries.',
 'I think I might be experiencing symptoms of depression.',
 'I need advice on gender-affirming surgeries.',
 "I'm interested in learning more about meditation.",
 'Can you provide resources for dealing with anxiety?',
 "I'm curious about healthy eating habits.",
 'I think I might be experiencing symptoms of depression.',
 'Can you recommend some good exercises for back pain?',
 'Can you recommend some good exercises for back pain?',
 'I need advice on gender-affirming surgeries.',
 "I'm curious about healthy eating habits.",
 'I have questions about hormone replacement therapy.',
 "I'm struggling with stress and need some advice.",
 'I have questions about hormone replacement therapy.',
 'I need advice on gender-affirming surgeries.',
 "I'm looking for support groups for transgender individuals.",
 'Can you recommend some good exercises for back pain?',
 'I think I might be experiencing symptoms of depression.',
 'I have concerns about unprotected sex.',
 'Can you recommend some good exercises for back pain?',
 'I have concerns about unprotected sex.',
 'Can you recommend some good exercises for back pain?',
 "I'm looking for information on STDs.",
 'I need advice on gender-affirming surgeries.',
 'Can you provide information on changing my legal gender?',
 "I've been feeling really down lately.",
 'I need tips for improving my sleep quality.',
 "I'm looking for support groups for transgender individuals.",
 'Can you tell me more about birth control options?',
 "I'm interested in learning more about meditation.",
 'Can you provide resources for dealing with anxiety?',
 'Can you tell me more about birth control options?',
 "I'm struggling with stress and need some advice.",
 "I'm looking for information on STDs.",
 "I've been feeling really down lately.",
 "I'm interested in learning more about meditation.",
 "I've been feeling really down lately.",
 'Can you provide resources for dealing with anxiety?',
 "I'm experiencing symptoms after a sexual encounter.",
 'I need advice on gender-affirming surgeries.',
 'Can you provide information on changing my legal gender?',
 'I have questions about hormone replacement therapy.',
 "I'm interested in learning more about meditation.",
 "I'm looking for information on STDs.",
 'I need advice on gender-affirming surgeries.',
 "I'm interested in learning more about meditation.",
 "I'm curious about healthy eating habits.",
 'I need advice on gender-affirming surgeries.',
 'I have questions about hormone replacement therapy.',
 'I have concerns about unprotected sex.',
 'I think I might be experiencing symptoms of depression.',
 'Can you recommend some good exercises for back pain?',
 "I'm interested in learning more about meditation.",
 'I need tips for improving my sleep quality.',
 'Can you provide resources for dealing with anxiety?',
 "I'm looking for support groups for transgender individuals.",
 'I need advice on gender-affirming surgeries.']

# labels = [random.randint(0, 3) for _ in range(100)]

# Initialize labels array
labels = []

# Define keyword sets for each category
mental_health_keywords = {'sleep', 'anxiety', 'depression', 'stress', 'down', 'meditation', 'mental'}
sexual_health_keywords = {'birth control', 'unprotected sex', 'STDs', 'sexual encounter'}
queer_gender_inquiry_keywords = {'transgender', 'gender-affirming', 'changing my legal gender', 'hormone replacement'}

# Classify each sentence
for sentence in sentences:
    sentence_lower = sentence.lower()
    
    if any(keyword in sentence_lower for keyword in mental_health_keywords):
        labels.append(0)  # Mental Health
    elif any(keyword in sentence_lower for keyword in sexual_health_keywords):
        labels.append(1)  # Sexual Health
    elif any(keyword in sentence_lower for keyword in queer_gender_inquiry_keywords):
        labels.append(2)  # Queer Gender Inquiry
    else:
        labels.append(3)  # None of them

labels = np.array(labels)
print("labels: ", labels)

# Preprocess data
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=120, truncating='post', padding='post')

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Model architecture
model = Sequential([
    Embedding(5000, 32, input_length=120),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

test_sequences = tokenizer.texts_to_sequences(sentences)
test_padded = pad_sequences(test_sequences, maxlen=120, truncating='post', padding='post')
predictions = model.predict(test_padded)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluate the model
# results = model.evaluate(X_test, y_test)
# print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

for i, sentence in enumerate(sentences):
    print(f"Sentence: '{sentence}'")
    print(f"Belongs to category: {predicted_classes[i]}")