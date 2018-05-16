import sys
sys.path.append("C:/Users/victo/Documents/Honours2/penne")
from penne import *
from penne import lm
from penne import recurrent, conv
import random
import time
import numpy

vocab_size = 30000

train = lm.read_data("data/kyoto-train.cln.ja")
target = lm.read_data("data/kyoto-train.align")
vocab = lm.make_vocab(train,vocab_size)
numberizer = lm.Numberizer(vocab)

#the other file has to be used, but does not have aligned data yet
valid_data = lm.read_data("data/kyoto-train.cln.ja")
valid_target = lm.read_data("data/kyoto-train.align")

embedding_dims = 128
hidden_dims = 128

embeddings = [parameter(numpy.random.normal(0.,0.01,(embedding_dims,))) for w in vocab]

#encoder
enc = recurrent.GRU(embedding_dims,hidden_dims)

def encode(fids):
    fvectors = map(lambda x: embeddings[x], fids)
    enc.start()
    for v in fvectors:
        enc.step(v)
    return enc.h

#decoder

out_embeddings = [parameter(numpy.random.normal(0., 0.01, (hidden_dims,))) for w in vocab]
dec = recurrent.GRU(embedding_dims,hidden_dims)

def decode(tar, enc_state, fids):
    loss = constant(0.)
    perm = [numberizer.numberize("<s>")] + map(int, tar[:-1])

    word_out = map(lambda x: out_embeddings[x], fids)
    out_layer = stack(word_out)

    dec.start(enc_state)
    fvectors = map(lambda x: embeddings[x], perm)
    for i,v in enumerate(fvectors[:-1]):
        h = dec.step(v)
        o = logsoftmax(dot(out_layer, h))
        loss -= o[perm[i+1]]
    return loss

#training
trainer = Adagrad(learning_rate=0.1)
last_valid_loss = 0
for epoch in xrange(100):
    train_loss = 0
    num_sents = 0
    num_words = 0
    for sent,tar in zip(train,target):
        num_sents += 1
        num_words += len(sent)
        assert(len(sent) == len(tar))
        fids = map(numberizer.numberize, sent[:-1]) #index of words
        state = encode(fids)
        loss = decode(tar, state, fids)
        train_loss += trainer.receive(loss)
        if num_sents % 100 == 0:
            print "sent: %s\tloss: %s" % (num_sents, numpy.exp(train_loss/num_words))
    print "\nloss of epoch %s:\t%s\n" % (epoch, numpy.exp(train_loss/num_words))

    valid_loss = 0
    valid_num_sent = 0
    valid_num_words = 0
    for sent,tar in  zip(valid_data,valid_target):
        valid_num_sents += 1
        valid_num_words += len(sent)
        assert(len(sent) == len(tar))
        fids = map(numberizer.numberize, sent[:-1]) #index of words
        state = encode(fids)
        loss = decode(tar, state, fids)
        valid_loss += compute_values(loss)[loss]
        if valid_num_sents % 100 == 0:
            print "valid sent: %s\t valid loss: %s" % (valid_num_sents, numpy.exp(valid_loss/valid_num_words))
    if last_valid_loss != 0 and valid_loss > last_valid_loss:
        print "Stopped training at epoch: %s\n epoch train loss: %s\tvalid loss: %s" % (epoch, numpy.exp(train_loss/num_words),numpy.exp(valid_loss/valid_num_words))
        break

save_model(open("data/trained_network.model","w"))