# -------------- Text generation ----------------
# ---------- Character Level Encoding -----------
# --------------    Vanilla RNN     -------------

# hidden(t) = tanh ( Weight(t) * hidden(t-1) + Weight(input) * Input(t) )
# output(t) = Weight(output) * hidden(t)

# ------------ Implementation Steps -----------
# 1 creating vocab
# 2 padding
# 3 splitting into input/labels
# 4 one-hot encoding
# 5 model defining
# 6 model training
# 7 model evaluation

import torch
from torch import nn
import numpy as np

# 1 ---------------------------------------------
text = ["hey how are you", "good i am fine", "have a nice day"]
chars = set(" ".join(text)) # joining the documents and getting unique characters i.e. vocabulary
int2char = dict(enumerate(chars)) # dictionary : mapping of character on an integer

char2int = {} # dictionary: mapping of integers on a character for reversing the encoding
for k,v in int2char.items():
    char2int.update({v:k})

print(char2int)
# 2 ---------------------------------------------
# padding w.r.t the longest string
longest = max(text, key=len)
max_len = len(longest)

for i in range(len(text)):      # all documents in text
    while len(text[i])<max_len: # append whitespace at the end of the doc till it reaches max_len
        text[i] += " "
print("After padding text: ",text)
# 3 ---------------------------------------------
# input: last character removed
# label/ground truth: first character removed (one time-step ahead of input data)
input_seq = []
label_seq = []

for i in range(len(text)):
    input_seq.append(text[i][:-1])
    label_seq.append(text[i][1:])
print("Input: ",input_seq)
print("Label: ",label_seq)

# 4 ---------------------------------------------
# First integer encoding then one-hot-encoding
# integer encoding the input and label
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    label_seq[i] = [char2int[character] for character in label_seq[i]]
print(input_seq)
print(label_seq)

# dict_size :   dictionary length - determines the length of one-hot vectors
# seq_len   :   length of sequences being fed to the model which is fixed: maxlength-1 as last char got removed
# batch_size:   the number of sentences in one batch

dict_size = len(char2int) # 17
seq_len = max_len-1       # 14
batch_size = len(text)    # 3

def hot_encoder(input_sequence, batch_size, seq_len, dict_size):
    """ creates arrays of zeroes for each character in a sequence
    and replaces the corresponding character-index with 1 i.e. makes
    it "hot".
    """
    # multi-dim array of zeros with desired output size
    # shape = 3,14,17
    # every doc is 14 char long. Each char is a 17 items long one-hot vector as dict_size is 17
    onehot_data = np.zeros(shape=(batch_size, seq_len, dict_size))

    # replacing the 0 with 1 for each character to make its one-hot vector
    for batch_item in range(batch_size):
        for sequence_item in range(seq_len):
            onehot_data[batch_item, sequence_item, input_sequence[batch_item][sequence_item] ] = 1

    print("Shape of one-hot encoded data: ", onehot_data.shape)
    return onehot_data

input_seq = hot_encoder(input_seq,batch_size, seq_len, dict_size)
print("Input data after being shaped and one-hot:\n",input_seq)

# From np array Tensor
input_seq = torch.from_numpy(input_seq)
label_seq = torch.Tensor(label_seq)
print(label_seq)

# 5 ---------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



