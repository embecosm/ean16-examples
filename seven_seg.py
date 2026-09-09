#!/usr/bin/env python3

# Basic PyTorch model to recognize 7-segement display characters

# Copyright (C) 2025 Embecosm Limited
#
# Contributor: Jeremy Bennett <jeremy.bennett@embecosm.com>

# SPDX-License-Identifier: GPL-3.0-or-later

"""
A simple 3 layer neural network model to recognize 7-segment displays and
categorize as a digit/symbol value or "invalid".

The input to the model is a vector of 7 bits representing the display as
follows

   - 0 -
   |   |
   1   2
   |   |
   + 3 +
   |   |
   4   5
   |   |
   + 6 +

Digit/symbol encodings

   Binary  Decimal  Meaning
  1110111      119     0
  0100100       36     1
  1011101       93     2
  1101101      109     3
  0101110       46     4
  1101011      107     5
  1111011      123     6
  0100101       37     7
  1111111      127     8
  1101111      111     9
  0001000        8     -
  0000000        0   blank

The outputs from the model is a vector of 13 bits, with the bits
representing:

 0 - 9: Digits 0-9
 10:    Minus sign
 11:    Blank
 12:    Invalid

"""

from random import randrange
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# The baseline data vectors of good and bad displays. 7-bit numbers with 1
# where that segment is lit.
goodDig = [0b1110111, 0b0100100, 0b1011101, 0b1101101, 0b0101110, 0b1101011,
           0b1111011, 0b0100101, 0b1111111, 0b1101111, 0b0001000, 0b0000000, ]
badDig =  [0b0000001, 0b0000010, 0b0000011, 0b0000100, 0b0000101, 0b0000110,
           0b0000111, 0b0001001, 0b0001010, 0b0001011, 0b0001100, 0b0001101,
           0b0001110, 0b0001111, 0b0010000, 0b0010001, 0b0010010, 0b0010011,
           0b0010100, 0b0010101, 0b0010110, 0b0010111, 0b0011000, 0b0011001,
           0b0011010, 0b0011011, 0b0011100, 0b0011101, 0b0011110, 0b0011111,
           0b0100000, 0b0100001, 0b0100010, 0b0100011, 0b0100110, 0b0100111,
           0b0101000, 0b0101001, 0b0101010, 0b0101011, 0b0101100, 0b0101101,
           0b0101111, 0b0110000, 0b0110001, 0b0110010, 0b0110011, 0b0110100,
           0b0110101, 0b0110110, 0b0110111, 0b0111000, 0b0111001, 0b0111010,
           0b0111011, 0b0111100, 0b0111101, 0b0111110, 0b0111111, 0b1000000,
           0b1000001, 0b1000010, 0b1000011, 0b1000100, 0b1000101, 0b1000110,
           0b1000111, 0b1001000, 0b1001001, 0b1001010, 0b1001011, 0b1001100,
           0b1001101, 0b1001110, 0b1001111, 0b1010000, 0b1010001, 0b1010010,
           0b1010011, 0b1010100, 0b1010101, 0b1010110, 0b1010111, 0b1011000,
           0b1011001, 0b1011010, 0b1011011, 0b1011100, 0b1011110, 0b1011111,
           0b1100000, 0b1100001, 0b1100010, 0b1100011, 0b1100100, 0b1100101,
           0b1100110, 0b1100111, 0b1101000, 0b1101001, 0b1101010, 0b1101100,
           0b1101110, 0b1110000, 0b1110001, 0b1110010, 0b1110011, 0b1110100,
           0b1110101, 0b1110110, 0b1111000, 0b1111001, 0b1111010, 0b1111100,
           0b1111101, 0b1111110,]

# Symbolic representation of the output categories 0-12
cats = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ' ', '?']

# A map of all the segment combinations and their corresponding
# output category.
allDig = {0b0000000: 11, 0b0000001: 12, 0b0000010: 12, 0b0000011: 12,
          0b0000100: 12, 0b0000101: 12, 0b0000110: 12, 0b0000111: 12,
          0b0001000: 10, 0b0001001: 12, 0b0001010: 12, 0b0001011: 12,
          0b0001100: 12, 0b0001101: 12, 0b0001110: 12, 0b0001111: 12,
          0b0010000: 12, 0b0010001: 12, 0b0010010: 12, 0b0010011: 12,
          0b0010100: 12, 0b0010101: 12, 0b0010110: 12, 0b0010111: 12,
          0b0011000: 12, 0b0011001: 12, 0b0011010: 12, 0b0011011: 12,
          0b0011100: 12, 0b0011101: 12, 0b0011110: 12, 0b0011111: 12,
          0b0100000: 12, 0b0100001: 12, 0b0100010: 12, 0b0100011: 12,
          0b0100100:  1, 0b0100101:  7, 0b0100110: 12, 0b0100111: 12,
          0b0101000: 12, 0b0101001: 12, 0b0101010: 12, 0b0101011: 12,
          0b0101100: 12, 0b0101101: 12, 0b0101110:  4, 0b0101111: 12,
          0b0110000: 12, 0b0110001: 12, 0b0110010: 12, 0b0110011: 12,
          0b0110100: 12, 0b0110101: 12, 0b0110110: 12, 0b0110111: 12,
          0b0111000: 12, 0b0111001: 12, 0b0111010: 12, 0b0111011: 12,
          0b0111100: 12, 0b0111101: 12, 0b0111110: 12, 0b0111111: 12,
          0b1000000: 12, 0b1000001: 12, 0b1000010: 12, 0b1000011: 12,
          0b1000100: 12, 0b1000101: 12, 0b1000110: 12, 0b1000111: 12,
          0b1001000: 12, 0b1001001: 12, 0b1001010: 12, 0b1001011: 12,
          0b1001100: 12, 0b1001101: 12, 0b1001110: 12, 0b1001111: 12,
          0b1010000: 12, 0b1010001: 12, 0b1010010: 12, 0b1010011: 12,
          0b1010100: 12, 0b1010101: 12, 0b1010110: 12, 0b1010111: 12,
          0b1011000: 12, 0b1011001: 12, 0b1011010: 12, 0b1011011: 12,
          0b1011100: 12, 0b1011101:  2, 0b1011110: 12, 0b1011111: 12,
          0b1100000: 12, 0b1100001: 12, 0b1100010: 12, 0b1100011: 12,
          0b1100100: 12, 0b1100101: 12, 0b1100110: 12, 0b1100111: 12,
          0b1101000: 12, 0b1101001: 12, 0b1101010: 12, 0b1101011:  5,
          0b1101100: 12, 0b1101101:  3, 0b1101110: 12, 0b1101111:  9,
          0b1110000: 12, 0b1110001: 12, 0b1110010: 12, 0b1110011: 12,
          0b1110100: 12, 0b1110101: 12, 0b1110110: 12, 0b1110111:  0,
          0b1111000: 12, 0b1111001: 12, 0b1111010: 12, 0b1111011:  6,
          0b1111100: 12, 0b1111101: 12, 0b1111110: 12, 0b1111111:  8,}

NUM_GOOD = len(goodDig)
NUM_BAD = len(badDig)
NUM_ALL = NUM_GOOD + NUM_BAD

class NN7Seg(nn.Module):
    """
    A simple 3 layer model, two linear, with a non-linear inner layer
    suffices.  We make the size of the inner layer configurable.
    """
    def __init__(self, inner_layer_size):
        """
        Create the layers for the neural network.
        """
        super().__init__()
        self.layer1 = nn.Linear(7, inner_layer_size)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(inner_layer_size, 13)

    def forward(self, x):
        """
        Compute the output vector from the input vector.
        """
        y = self.layer1(x)
        z = self.layer2(y)
        return self.layer3(z)

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Do one epoch of training.  Return the final loss.
    """
    # Put the model into training mode
    model.train()
    # Train on dataset (batch) at a time
    for _, (invec, outvec) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(invec)
        loss = loss_fn(pred, outvec)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def test_loop(dataloader, model, loss_fn):
    """
    Test all the possible inputs to determine how well trained we are.  Return
    a tuple of accuracy and loss
    """
    # Put the model in inference mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for invec, outvec in dataloader:
            pred = model(invec)
            test_loss += loss_fn(pred, outvec).item()
            correct += (pred.argmax(1) == outvec.argmax(1)).type(torch.int).sum().item()

    test_loss /= num_batches
    correct /= size
    return (100*correct, test_loss)

def make_invec(idx):
    """
    Convert a bitstring representation of an 7-segment display to a vector.
    """
    res = []
    for b in range(0, 7):
        res.append((idx >> b) & 1)
    return res

def make_outvec(idx):
    """
    Convert a numeric output value to a vector, with the bit corresponding to
    that value set to 1.
    """
    res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    res[idx] = 1
    return res

class Custom7SegTrainDataset(Dataset):
    """
    The dataset for training.  This dataset gives an even balance between the
    frequency of each valid input and invalid inputs, even though the great
    majority of possible inputs are invalid.
    """
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        It is important that the 13 possible outputs are sampled evenly from
        the possible inputs. All but one output correspond to 1 possible
        input.  The last input corresponds to 116 possible inputs.
        """
        idx_mod = idx % (NUM_GOOD * 2)  # Half the cases will be bad digits
        if idx_mod < NUM_GOOD:
            # Good digit, input is the index
            invec = make_invec(goodDig[idx_mod])
            outvec = make_outvec(idx_mod)
        else:
            # Must be a bad digit, select one at random
            idx_bad = randrange(0, NUM_BAD)
            invec = make_invec(badDig[idx_bad])
            outvec = make_outvec(NUM_GOOD)

        return torch.tensor(invec, dtype=torch.float), \
            torch.tensor(outvec, dtype=torch.float)


class Custom7SegTestDataset(Dataset):
    """
    The dataset for testing.  This dataset yields each possible input, of
    which 12 correspond to valid digits/symbols and 116 to invalid
    digits/symbols.
    """
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        We just map the input to one of the 128 possible combinations
        """
        idx_mod = idx % (NUM_ALL)
        invec = make_invec(idx_mod)
        outvec = make_outvec(allDig[idx_mod])
        return torch.tensor(invec, dtype=torch.float), \
            torch.tensor(outvec, dtype=torch.float)


def train_model(model, learning_rate, batch_size, num_training_examples,
                num_testing_examples):
    """
    Train the model according to the supplied parameters.
    """
    # Training parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set up data and dataloaders for training and testing
    train_data = Custom7SegTrainDataset(num_training_examples)
    test_data = Custom7SegTestDataset(num_testing_examples)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=1)

    # Multiple epochs of training until we get >99.9% accuracy.
    for t in range(50):
        train_loss = train_loop(train_data_loader, model, loss_fn, optimizer)
        acc, test_loss = test_loop(test_data_loader, model, loss_fn)
        print(f'Epoch {t:3d}, training loss {train_loss:8.5f}, ' +
              f'testing loss {test_loss:8.5f}, test accuracy {acc:6.2f}%')
        if acc > 99.9:
            return

def display_7seg(val):
    """
    Provide a 5 line list which displays a character.
    """
    val_vec = make_invec(val)
    ch0 = '-' if val_vec[0] == 1 else ' '
    ch1 = '|' if val_vec[1] == 1 else ' '
    ch2 = '|' if val_vec[2] == 1 else ' '
    ch3 = '-' if val_vec[3] == 1 else ' '
    ch4 = '|' if val_vec[4] == 1 else ' '
    ch5 = '|' if val_vec[5] == 1 else ' '
    ch6 = '-' if val_vec[6] == 1 else ' '

    res = []
    res.append(f' {ch0} ')
    res.append(f'{ch1} {ch2}')
    res.append(f' {ch3} ')
    res.append(f'{ch4} {ch5}')
    res.append(f' {ch6} ')

    return res

def showcase(model):
    """
    Showcase the model, using some simple examples.
    """
    # Put the model in inference mode
    model.eval()
    # Set of 14 test values to be displayed in two rows of 7.
    demos = goodDig
    demos.append(28)    # Known bad display.
    demos.append(42)    # Known bad display.
    lines = ['', '', '', '', '',]
    for i, val in enumerate(demos):
        # Predict the answer
        invec = torch.tensor(make_invec(val), dtype=torch.float)
        disp = display_7seg(val)
        resval = model(invec).argmax(0)
        res = cats[resval]
        # Create the basic shape
        lines[0] = f'{lines[0]}{disp[0]}'
        lines[1] = f'{lines[1]}{disp[1]}'
        lines[2] = f'{lines[2]}{disp[2]} = {res}'
        lines[3] = f'{lines[3]}{disp[3]}'
        lines[4] = f'{lines[4]}{disp[4]}'
        # For the last one of 7, print it and clear, otherwise pad.
        if (i % 7) == 6:
            for l in lines:
                print(l)
            print()
            lines = ['', '', '', '', '',]
        else:
            lines[0] = f'{lines[0]}      |  '
            lines[1] = f'{lines[1]}      |  '
            lines[2] = f'{lines[2]}  |  '
            lines[3] = f'{lines[3]}      |  '
            lines[4] = f'{lines[4]}      |  '

def main():
    """
    Main program learning, testing and showcasing
    """
    # Create the model
    inner_layer_size = 26
    mynn = NN7Seg(inner_layer_size)

    # General parameters
    learning_rate = 1e0
    batch_size = 32
    num_training_examples = 1000
    num_testing_examples = 128

    # Train the model
    print(f'Number of training examples: {num_training_examples}, ' +
          f'learning rate {learning_rate:0.5f}, ' +
          f'inner layer size {inner_layer_size}')
    train_model(model=mynn, learning_rate=learning_rate,
                batch_size=batch_size,
                num_training_examples=num_training_examples,
                num_testing_examples=num_testing_examples)
    print("Training done!")

    # Showcase the model
    showcase(mynn)

# Make sure we have new enough python.  This is will predate logging being set
# up, so just print any error message.
def check_python_version(major, minor):
    """Check the python version is at least {major}.{minor}."""
    if ((sys.version_info[0] < major)
        or ((sys.version_info[0] == major) and (sys.version_info[1] < minor))):
        print(f'ERROR: Requires Python {major}.{minor} or later',
              file=sys.stderr)
        sys.exit(1)

# Make sure we have new enough Python and only run if this is the main package
check_python_version(3, 10)
if __name__ == '__main__':
    sys.exit(main())
