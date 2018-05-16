import sys
sys.path.append("/Users/ketran/tools/penne")
from penne import *
from penne import lm
from penne import recurrent, conv
import numpy
import random
import time

source_file = "data/kyoto-train.align.ja"
source_order = "data/kyoto-train.align"

vocab_size = 30000  # short list of words

# make data
data = lm.read_data(source_file)
target = lm.read_data(source_order)

vocab = lm.make_vocab(data, vocab_size)
numberizer = lm.Numberizer(vocab)

embedding_dims = 128
hidden_dims = 128
# create word-embeddings
embeddings = [parameter(numpy.random.normal(0., 0.01, (embedding_dims,))) for w in vocab]

# create an encoder
enc = recurrent.GRU(embedding_dims, hidden_dims)

frnn_forward = recurrent.GRU(embedding_dims, hidden_dims)
frnn_backward = recurrent.GRU(embedding_dims, hidden_dims)
flinear = Layer(2*hidden_dims, hidden_dims, f=None, bias=None)
def encode_birnn(fids):
    fvectors = map(lambda x: embeddings[x], fids)
    states_forward = frnn_forward.transduce(fvectors)
    states_backward = reversed(frnn_backward.transduce(reversed(fvectors)))
    states = []
    for sf, sb in zip(states_forward, states_backward):
        states.append(concatenate([sf, sb]))

    return flinear(stack(states))

# make a decoder
dec = recurrent.GRU(hidden_dims, hidden_dims)
def encode(fids):
    """
    arg: a list of words,
    return: a vector that encode the whole sequence of words
    """
    fvectors = map(lambda x: embeddings[x], fids)
    enc.start()
    for v in fvectors:
        enc.step(v)
    return enc.h


def decode_loss(words, target):
    loss = constant(0.)
    assert(len(words) == len(target))
    fids = map(numberizer.numberize, words) # get index of words in vocabulary
    # map the string into integer index of permutation
    perm = map(int, target[:-1])
    # map permutation to context index
    """
    Even though we want to predict the permutation, but the number itself is nonsense
    We want to have more information from those number. In this simplest implementation,
    we use the word at perm[i] as the context
    """
    out_layer = encode_birnn(fids)
    # initialize decoder with encoder output
    dec.start(encode(fids))
    fvectors = map(lambda x: embeddings[x], fids)
    for i,vec in enumerate(fvectors[:-1]):
        h = dec.step(vec)  # compute hidden state of GRU
        o = logsoftmax(dot(out_layer, h))  # compute output
        loss -= o[perm[i]]  # compute loss
    return loss


# Training
trainer = Adagrad(learning_rate=0.1)

for epoch in xrange(100):
    train_loss = 0
    num_sents = 0
    num_words = 0
    for words, target in zip(data, target):
        num_sents += 1
        num_words += len(words) - 1
        loss = decode_loss(words, target)
        train_loss += trainer.receive(loss)
        if num_sents % 50 == 0:
            print "sent: %s\tloss: %s" % (num_sents, numpy.exp(train_loss/num_words))
    print "loss of epoch %s:\t%s" % (epoch, numpy.exp(train_loss/num_words))


