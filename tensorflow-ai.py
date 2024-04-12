import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from data import data  # Import data from data.py


# Tokenize the data
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, oov_token='<OOV>')
tokenizer.fit_on_texts([input_text for input_text, _ in data] + [target_text for _, target_text in data])

# Add special tokens to the word index
tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

# Print the word index to verify
print(tokenizer.word_index)

# Reverse the word index to map token indices back to words
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

input_sequences = tokenizer.texts_to_sequences([input_text for input_text, _ in data])
target_sequences = tokenizer.texts_to_sequences([target_text for _, target_text in data])

# Pad the sequences
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

# Define vocab_size
vocab_size = len(tokenizer.word_index) + 1

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')
target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=vocab_size)

# Define the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_units = 256

encoder_inputs = tf.keras.Input(shape=(None,), name='encoder_input')
encoder_embeddings = Embedding(vocab_size, embedding_dim, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embeddings)

decoder_inputs = tf.keras.Input(shape=(None,), name='decoder_input')
decoder_embeddings = Embedding(vocab_size, embedding_dim, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=[state_h, state_c])

decoder_dense = Dense(vocab_size, activation='softmax', name='dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_sequences, target_sequences[:, :-1]], target_sequences[:, 1:], epochs=100, batch_size=64)


# Test the model
def sample_with_temperature(logits, temperature=1.0):
    logits = np.asarray(logits).astype('float64')
    logits = np.log(logits) / temperature
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    choices = range(len(logits))
    return np.random.choice(choices, p=probs)


def make_inference(input_text, temperature=1.0):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')

    # Predict on the encoder LSTM layer to get the initial states
    _, state_h_enc, state_c_enc = model.get_layer('encoder_lstm')(model.get_layer('encoder_embedding')(input_seq))
    states_value = [state_h_enc, state_c_enc]

    # Initialize the decoder input with the start token
    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    output_sequence = []

    for _ in range(max_target_length):
        # Predict on the decoder LSTM layer
        decoder_output, state_h_dec, state_c_dec = model.get_layer('decoder_lstm')(
            model.get_layer('decoder_embedding')(decoder_input), initial_state=states_value)
        states_value = [state_h_dec, state_c_dec]

        # Predict on the dense layer
        output = model.get_layer('dense')(decoder_output)

        # Sample the next token with temperature
        sampled_token_index = sample_with_temperature(output[0, -1], temperature)
        output_sequence.append(sampled_token_index)

        # If the end token is predicted, break the loop
        if sampled_token_index == tokenizer.word_index['<end>']:
            break

        # Update the decoder input for the next time step
        decoder_input = tf.expand_dims([sampled_token_index], 0)

    # Convert the token indices back to text
    return tokenizer.sequences_to_texts([output_sequence])[0]


# Chat loop
while True:
    user_input = input("USER: ")
    if user_input.lower() == 'quit':
        print("SINEWAVE-AI: Goodbye!")
        break
    response = make_inference(user_input)
    print(f"SINEWAVE-AI: {response}")
