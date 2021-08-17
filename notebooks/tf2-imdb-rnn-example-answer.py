ex1_inputs = keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(input_dim=nb_words, 
                     output_dim=embedding_dims)(ex1_inputs)
x = layers.Dropout(0.2)(x)

x = layers.LSTM(lstm_units, return_sequences=True)(x)
x = layers.LSTM(lstm_units)(x)

ex1_outputs = layers.Dense(1, activation='sigmoid')(x)