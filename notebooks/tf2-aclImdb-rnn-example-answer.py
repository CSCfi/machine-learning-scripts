ex1_inputs = keras.Input(shape=(1,), dtype=tf.string)

x = vectorization_layer(ex1_inputs)
x = layers.Embedding(input_dim=nb_words, 
                     output_dim=embedding_dims)(x)
x = layers.Dropout(0.2)(x)

x = layers.LSTM(lstm_units, return_sequences=True)(x)
x = layers.LSTM(lstm_units)(x)

## With bidirectional layers:
#x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
#x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

ex1_outputs = layers.Dense(1, activation='sigmoid')(x)
