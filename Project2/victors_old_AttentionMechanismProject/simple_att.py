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

# use another set of parameter for predicting permutation
# this is non-shared parameter setting of the output layer
out_embeddings = [parameter(numpy.random.normal(0., 0.01, (hidden_dims,))) for w in vocab]

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
    fids = map(numberizer.numberize, words[:-1]) # get index of words in vocabulary
    # map the string into integer index of permutation
    perm = [numberizer.numberize("<s>")] + map(int, target[:-1])# + [numberizer.numberize("</s>")]
    # map permutation to context index
    """
    Even though we want to predict the permutation, but the number itself is nonsense
    We want to have more information from those number. In this simplest implementation,
    we use the word at perm[i] as the context
    """
    word_out = map(lambda x: out_embeddings[x], fids)  # do not predict for end of sentence </s>
    # turn words to vector embeddings
    out_layer = stack(word_out)
    # initialize decoder with encoder output
    dec.start(encode(fids))
    fvectors = map(lambda x: embeddings[x], perm)
    #print perm, len(perm), len(word_out)
    for i,vec in enumerate(fvectors[:-1]):
        h = dec.step(vec)  # compute hidden state of GRU
        o = logsoftmax(dot(out_layer, h))  # compute output
        loss -= o[perm[i+1]]  # compute loss
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


