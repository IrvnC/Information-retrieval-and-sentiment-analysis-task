# =====================================================================================================
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Target accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution():
    data_url = #'link to datasets'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    with open('sarcasm.json', 'r') as f:
        datastore=json.load(f)

    # DO NOT CHANGE THIS CODE
    # Make sure used all of these parameters
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    for i in datastore:
        sentences.append(i['headline'])
        labels.append(i['is_sarcastic'])

    training_sentences = sentences[:training_size]
    validation_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    validation_labels = labels[training_size:]


    # Fit tokenizer with training data
    tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)# YOUR CODE HERE
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)


    training_label_seq = np.array(training_labels)
    validation_label_seq = np.array(validation_labels)

    model = tf.keras.Sequential([
        # Modify model here. Don't change the last layer.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Make sure you are using binary loss function
    model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, baseline=0.80)
    #add callbacks in model.fit when it need to avoid overvitting

    model.fit(train_padded,
              training_label_seq,
              validation_data=(validation_padded, validation_label_seq),
              epochs=20
              )

    return model


if __name__ == '__main__':
    # The code below is to run and save model as a .h5 file.
    # You can change the model format according to your needs
    model = solution()
    model.save("model.h5")
