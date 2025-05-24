# Import PyTorch and relevant modules
import torch
import torch.nn as nn
from torch.autograd import Variable  # For wrapping tensors for autograd
import torch.optim as optim  # Optimizers like Adam, SGD
import torch.nn.functional as F  # Functional interface (activation etc.)
import random  # To shuffle training data
import numpy as np  # For numerical operations if needed
from copy import deepcopy  # To make deep copies of lists

# Flatten helper function: converts list of lists into a single list
flatten = lambda l: [item for sublist in l for item in sublist]

# Check if CUDA (GPU) is available and select device
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.set_device(0)  # Use GPU 0 for computation

# Define tensor types depending on device availability for convenience
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

random.seed(1024)  # Set random seed for reproducibility

# Function to generate batches from the training data
def getBatch(batch_size, train_data):
    random.shuffle(train_data)  # Shuffle entire dataset for randomness
    sindex = 0  # Start index of batch
    eindex = batch_size  # End index of batch
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]  # Slice a batch
        sindex = eindex  # Move start to next batch start
        eindex += batch_size  # Move end to next batch end
        yield batch  # Yield this batch (generator)
    if sindex < len(train_data):
        yield train_data[sindex:]  # Yield remaining samples as last batch

# Function to pad sequences in a batch to same length for model input
def pad_to_batch(batch, w_to_ix):
    fact, q, a = list(zip(*batch))  # Separate facts, questions, answers
    
    max_fact = max([len(f) for f in fact])  # Max number of facts in any example
    max_len = max([f.size(1) for f in flatten(fact)])  # Max length of each fact (in words)
    max_q = max([qq.size(1) for qq in q])  # Max length of questions
    max_a = max([aa.size(1) for aa in a])  # Max length of answers

    facts, fact_masks, q_p, a_p = [], [], [], []  # Lists to store padded data

    # Pad facts to uniform length
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            # If fact shorter than max length, pad with 'PAD' token (index of '')
            if fact[i][j].size(1) < max_len:
                pad_len = max_len - fact[i][j].size(1)
                pad_tensor = Variable(LongTensor([w_to_ix['']] * pad_len)).view(1, -1)
                fact_p_t.append(torch.cat([fact[i][j], pad_tensor], 1))
            else:
                fact_p_t.append(fact[i][j])
        # Pad facts to max_fact count by adding empty facts if needed
        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(LongTensor([w_to_ix['']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)

        # Create masks where padding exists (mask=True for pad tokens)
        fact_masks.append(torch.cat([
            Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False)
            for t in fact_p_t]).view(fact_p_t.size(0), -1))

        # Pad questions similarly
        if q[i].size(1) < max_q:
            pad_len = max_q - q[i].size(1)
            pad_tensor = Variable(LongTensor([w_to_ix['']] * pad_len)).view(1, -1)
            q_p.append(torch.cat([q[i], pad_tensor], 1))
        else:
            q_p.append(q[i])

        # Pad answers similarly
        if a[i].size(1) < max_a:
            pad_len = max_a - a[i].size(1)
            pad_tensor = Variable(LongTensor([w_to_ix['']] * pad_len)).view(1, -1)
            a_p.append(torch.cat([a[i], pad_tensor], 1))
        else:
            a_p.append(a[i])

    questions = torch.cat(q_p)  # Concatenate padded questions into batch tensor
    answers = torch.cat(a_p)  # Concatenate padded answers

    # Create mask for padded questions (True where pad token)
    question_masks = torch.cat([
        Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False)
        for t in questions]).view(questions.size(0), -1)

    return facts, fact_masks, questions, question_masks, answers

# Prepare sequence: Convert list of words into tensor of word indices
def prepare_sequence(seq, to_index):
    # Map each word to its index, if not found map to '' token index
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index[""], seq))
    return Variable(LongTensor(idxs))  # Wrap indices in PyTorch Variable

# Load raw training data from file and clean lines
data = open('qa5_three-arg-relations_train.txt').readlines()
data = [d[:-1] for d in data]  # Remove trailing newline from each line

train_data = []
fact = []
qa = []
for d in data:
    index = d.split(' ')[0]  # Get the story line number from line
    if index == '1':  # If story index is 1, reset facts and QAs (new story)
        fact = []
        qa = []
    if '?' in d:  # If line contains a question
        temp = d.split('\t')  # Split question-answer pairs by tab
        ques = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']  # Extract question tokens
        ans = temp[1].split() + ['']  # Extract answer tokens + empty token
        temp_s = deepcopy(fact)  # Deepcopy facts so they don't change later
        train_data.append([temp_s, ques, ans])  # Append facts, question, answer to train_data
    else:
        fact.append(d.replace('.', '').split(' ')[1:] + [''])  # For facts, strip and split words

# Separate facts, questions, and answers to create vocabulary
fact, q, a = list(zip(*train_data))

# Create a set of all unique words in facts, questions, and answers for vocab
vocab = list(set(flatten(flatten(fact)) + flatten(q) + flatten(a)))

# Create word-to-index dictionary with special tokens mapped first
word_to_index = {'': 0, '': 1, '': 2, '': 3}  # You should fill actual special tokens here

# Add all vocab words to word_to_index dictionary
for vo in vocab:
    if word_to_index.get(vo) is None:
        word_to_index[vo] = len(word_to_index)

# Create index-to-word reverse dictionary for decoding
index_to_word = {v: k for k, v in word_to_index.items()}

# Convert all text data to tensor sequences of indices using vocab dictionary
for s in train_data:
    for i, fact in enumerate(s[0]):
        s[0][i] = prepare_sequence(fact, word_to_index).view(1, -1)  # Convert facts
    s[1] = prepare_sequence(s[1], word_to_index).view(1, -1)  # Convert questions
    s[2] = prepare_sequence(s[2], word_to_index).view(1, -1)  # Convert answers
