# ===================================================================================================
# Build and train a classifier for the BBC-text dataset.
# Fix a multiclass classification problem.
#
# The dataset used originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Target accuracy and validation_accuracy > 91%
# ===================================================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np



def solution():
    #bbc = pd.read_csv('Link to csv datasets')
    # print(bbc)

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters 
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # Splitting datasets
    train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(bbc['text'], bbc['category'], random_state=0, shuffle=False)
    labels = bbc['category']
    # print(labels)

    # # Fit tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    # training
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
    # validation
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    # training
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    # validation
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        # Modify model here. Don't change the last layer. (6 classes)
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using multiclass loss function
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(train_padded,
              training_label_seq,
              epochs=10,
              validation_data=(validation_padded, validation_label_seq)
              )

    return model


if __name__ == '__main__':
    # The code below is to run and save model as a .h5 file.
    # You can change the model format according to your needs
    model = solution()
    model.save("model.h5")
