import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense  # Dense layer for output projection

# Disable eager execution for TF 1.x style session graph execution
tf.compat.v1.disable_eager_execution()

# ====== Parameters ======
vocab_size = 50000           # Vocabulary size for both source and target languages
num_units = 128              # Number of LSTM units (hidden size)
batch_size = 16              # Batch size for training
source_sequence_length = 40  # Max length of source sentences (time steps)
target_sequence_length = 60  # Max length of target sentences (time steps)
decoder_type = 'basic'       # Decoder type: 'basic' or 'attention'
sentences_to_read = 50000    # Number of sentence pairs to read from files
src_max_sent_length = 41     # Max length for source sentences (with start token)
tgt_max_sent_length = 61     # Max length for target sentences (with start token)

# ====== Load Vocabulary Files ======
# Load source (German) vocabulary file: each line is one word
src_dictionary = {}
with open('vocab.50K.de.txt', encoding='utf-8') as f:
    for line in f:
        # Strip newline and map word to an increasing integer ID
        src_dictionary[line.strip()] = len(src_dictionary)
# Create reverse dictionary (id -> word) for decoding if needed
src_reverse_dictionary = {v: k for k, v in src_dictionary.items()}

# Load target (English) vocabulary file
tgt_dictionary = {}
with open('vocab.50K.en.txt', encoding='utf-8') as f:
    for line in f:
        tgt_dictionary[line.strip()] = len(tgt_dictionary)
tgt_reverse_dictionary = {v: k for k, v in tgt_dictionary.items()}

print('Source vocab size:', len(src_dictionary))
print('Target vocab size:', len(tgt_dictionary))

# ====== Load Sentence Data ======
source_sent = []  # List to hold all source sentences
target_sent = []  # List to hold all target sentences

# Read source sentences from file, skipping first 50 lines (header or metadata)
with open('train.de', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 50:
            continue
        source_sent.append(line.strip())
        if len(source_sent) >= sentences_to_read:
            break

# Similarly read target sentences
with open('train.en', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 50:
            continue
        target_sent.append(line.strip())
        if len(target_sent) >= sentences_to_read:
            break

# Sanity check: number of source and target sentences must be equal
assert len(source_sent) == len(target_sent), \
    f'Source and Target sentences mismatch: {len(source_sent)} vs {len(target_sent)}'

print(f'Read {len(source_sent)} sentence pairs.')

# ====== Tokenization Helper Function ======
def split_to_tokens(sent, is_source=True):
    """
    Preprocess a sentence by adding spaces around commas and periods,
    splitting into tokens, and replacing unknown tokens with empty string.
    """
    sent = sent.replace(',', ' ,').replace('.', ' .').replace('\n', ' ')
    tokens = sent.split(' ')
    dictionary = src_dictionary if is_source else tgt_dictionary
    # Replace unknown tokens with '' (empty string)
    tokens = [tok if tok in dictionary else '' for tok in tokens]
    return tokens

# ====== Sentence to Indices Conversion ======
def sentence_to_indices(sentence_tokens, dictionary, max_len, add_start_token=True):
    """
    Convert tokens into list of integer indices based on vocabulary.
    Add start token at beginning if add_start_token=True.
    Pad with empty token index or truncate to max_len.
    """
    indices = []
    if add_start_token:
        # Assuming '' token in dictionary used for padding or unknown token, index 0
        indices.append(dictionary.get('', 0))
    for tok in sentence_tokens:
        if tok in dictionary:
            indices.append(dictionary[tok])
        else:
            indices.append(dictionary.get('', 0))
    # Pad with 0s if shorter than max_len
    if len(indices) < max_len:
        indices += [dictionary.get('', 0)] * (max_len - len(indices))
    else:
        # Truncate if longer than max_len
        indices = indices[:max_len]
    return indices

# ====== Prepare Training Data ======
train_inputs = []
train_outputs = []
train_inp_lengths = []
train_out_lengths = []

for src_sent, tgt_sent in zip(source_sent, target_sent):
    # Tokenize source and target sentences
    src_tokens = split_to_tokens(src_sent, is_source=True)
    tgt_tokens = split_to_tokens(tgt_sent, is_source=False)

    # Convert tokens to indices
    # Reverse source tokens to improve learning for sequence to sequence
    src_indices = sentence_to_indices(src_tokens[::-1], src_dictionary, src_max_sent_length)
    tgt_indices = sentence_to_indices(tgt_tokens, tgt_dictionary, tgt_max_sent_length)

    # Add to training lists
    train_inputs.append(src_indices)
    train_outputs.append(tgt_indices)

    # Keep track of actual sentence lengths (+1 for start token)
    train_inp_lengths.append(min(len(src_tokens) + 1, src_max_sent_length))
    train_out_lengths.append(min(len(tgt_tokens) + 1, tgt_max_sent_length))

# Convert to numpy arrays (for easy feeding to TF graph)
train_inputs = np.array(train_inputs, dtype=np.int32)
train_outputs = np.array(train_outputs, dtype=np.int32)
train_inp_lengths = np.array(train_inp_lengths, dtype=np.int32)
train_out_lengths = np.array(train_out_lengths, dtype=np.int32)

print('Prepared training data shapes:')
print('Inputs:', train_inputs.shape)
print('Outputs:', train_outputs.shape)

# ====== Load Pretrained Embeddings ======
# Load pre-trained embedding matrices (shape: vocab_size x embedding_dim)
encoder_emb_layer = tf.convert_to_tensor(np.load('de-embeddings.npy'), dtype=tf.float32)
decoder_emb_layer = tf.convert_to_tensor(np.load('en-embeddings.npy'), dtype=tf.float32)

# ====== Define TensorFlow Placeholders ======
# Input placeholders for encoder inputs at each time step (time-major)
enc_train_inputs = [tf.compat.v1.placeholder(tf.int32, [batch_size], name=f'enc_train_inputs_{i}')
                    for i in range(source_sequence_length)]

# Decoder inputs for training (teacher forcing) at each time step
dec_train_inputs = [tf.compat.v1.placeholder(tf.int32, [batch_size], name=f'dec_train_inputs_{i}')
                    for i in range(target_sequence_length)]

# Decoder expected output labels at each time step (targets shifted by one)
dec_train_labels = [tf.compat.v1.placeholder(tf.int32, [batch_size], name=f'dec_train_labels_{i}')
                    for i in range(target_sequence_length)]

# Masks to ignore padding in loss computation (1.0 for real tokens, 0.0 for padding)
dec_label_masks = [tf.compat.v1.placeholder(tf.float32, [batch_size], name=f'dec_label_masks_{i}')
                   for i in range(target_sequence_length)]

# Placeholder for actual input lengths of encoder and decoder sequences (for dynamic_rnn)
enc_train_inp_lengths = tf.compat.v1.placeholder(tf.int32, [batch_size], name='train_input_lengths')
dec_train_inp_lengths = tf.compat.v1.placeholder(tf.int32, [batch_size], name='train_output_lengths')

# ====== Embedding Lookup for Encoder and Decoder ======
# For each time step, look up embeddings for input token indices
encoder_emb_inp = tf.stack([tf.nn.embedding_lookup(encoder_emb_layer, inp) for inp in enc_train_inputs])
decoder_emb_inp = tf.stack([tf.nn.embedding_lookup(decoder_emb_layer, inp) for inp in dec_train_inputs])

# ====== Encoder RNN ======
encoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units)
# Initialize encoder LSTM hidden state with zeros
initial_state = encoder_cell.zero_state(batch_size, tf.float32)

# Run dynamic RNN over the encoder input embeddings (time major)
encoder_outputs, encoder_state = tf.compat.v1.nn.dynamic_rnn(
    encoder_cell,
    encoder_emb_inp,
    initial_state=initial_state,
    sequence_length=enc_train_inp_lengths,
    time_major=True,
    dtype=tf.float32
)

# ====== Decoder RNN with Optional Attention ======
decoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units)

# Final projection layer to map decoder outputs to vocab logits
projection_layer = Dense(units=vocab_size, use_bias=True)

# Training helper feeds ground truth inputs at each time step for teacher forcing
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp,
    [tgt_max_sent_length - 1] * batch_size,  # sequence lengths for each example in batch
    time_major=True
)

if decoder_type == 'basic':
    # Basic decoder without attention
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer
    )
elif decoder_type == 'attention':
    # Attention mechanism setup
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, encoder_outputs,
        memory_sequence_length=enc_train_inp_lengths
    )
    # Wrap decoder cell with attention
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=num_units
    )
    decoder = tf.contrib.seq2seq.BasicDecoder(
        attn_cell, helper, attn_cell.zero_state(batch_size, tf.float32),
        output_layer=projection_layer
    )
else:
    raise ValueError('Unknown decoder type')

# Decode the sequences dynamically (time-major output)
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder,
    output_time_major=True,
    swap_memory=True
)

# Extract logits from decoder outputs
logits = outputs.rnn_output

# ====== Loss Computation ======
# Compute cross-entropy loss per time step and per batch example
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.stack(dec_train_labels),
    logits=logits
)

# Mask padding tokens in the loss, then average over batch and time
loss = tf.reduce_sum(crossent * tf.stack(dec_label_masks)) / (batch_size * target_sequence_length)

# ====== Optimizer ======
train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

# ====== TensorFlow Session to Initialize and Run ======
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print("Model initialized. Ready to train.")

    # Training loop (example outline):
    # for epoch in range(num_epochs):
    #     for batch in batches:
    #         feed_dict = {
    #             enc_train_inputs[t]: batch_encoder_inputs[:, t] for t in range(source_sequence_length)
    #             dec_train_inputs[t]: batch_decoder_inputs[:, t] for t in range(target_sequence_length)
    #             dec_train_labels[t]: batch_decoder_labels[:, t] for t in range(target_sequence_length)
    #             dec_label_masks[t]: batch_masks[:, t] for t in range(target_sequence_length)
    #             enc_train_inp_lengths: batch_encoder_lengths
    #             dec_train_inp_lengths: batch_decoder_lengths
    #         }
    #         sess.run(train_op, feed_dict=feed_dict)
