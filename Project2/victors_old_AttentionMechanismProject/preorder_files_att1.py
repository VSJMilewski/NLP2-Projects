import sys
sys.path.append("/home/milewski/tools/penne")
from penne import *
from penne import lm
from penne import recurrent, conv
import random
import time
import numpy
import scipy
import scipy.stats

"""
To load in the trained model, the model also has to be in the code.
So the encoder and the decoder are defined before loading the model
"""
vocab_size = 30000

train = lm.read_data("data/low/kyoto-train.cln.ja")
target = lm.read_data("data/align/kyoto-train.cln.align.ja")
vocab = lm.make_vocab(train,vocab_size)
numberizer = lm.Numberizer(vocab)

valid_data = lm.read_data("data/low/kyoto-dev.ja")
valid_target = lm.read_data("data/align/kyoto-dev.align.ja")

test = lm.read_data("data/low/kyoto-test.ja")
test_target = lm.read_data("data/align/kyoto-test.align.ja")

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


#the trained model is loaded
load_model(open("data/trained/trained_network_att1.npy"))

####
class Hypothesis(object):
    """use for decoding"""
    def __init__(self, perm, score, state):
        self.perm = perm
        self.score = score
        self.state = state

    def update(self):
        return


def decode(sent, beam_size):
    """Beam search decoder"""
    """bin: keep track of hypothesis, bin[i] contains all the hypotheses of length i"""
    bin = [] # to keep track of hypothesis

    # first of all, we need to encode the whole source sentence
    # since this is done only one time
    # map word to its id
    fids = map(numberizer.numberize, sent[:-1])
    enc_state = encode(fids)
    """
        enc_state is a computational graph
        it's very expensive to use it during decoding, since the graph will be recomputed each time
        and during decoding, we don't update parameter, so we would like to fix enc_state
        This is done nicely by compute the actual value of enc_state
    """
    enc_state = constant(compute_value(enc_state))  # now it's a constant, accessing time is constant

    """ the output layer also can be precomputed """
    word_out = map(lambda x: out_embeddings[x], fids)
    out_layer = stack(word_out)
    out_layer = constant(compute_value(out_layer))  ## turn it to a precomputed big matrix
    """ now we start the decoder with encoder state """
    dec.start(enc_state)
    state0 = dec.step(embeddings[numberizer.numberize("<s>")])
    state0_const = constant(compute_value(state0))
    # create the first hypothesis
    h0 = Hypothesis([-1], 0.0, state0_const)
    # put h0 to bin
    stacks = []
    stacks.append(h0)
    bin.append(stacks)  # note that we put h0 in to a stack first

    # now we can loop through number of source words
    n = len(fids)
    for i in range(n):
        prev_stack = bin[i]
        curr_stack = []
        for hypo in prev_stack:
            # expand it
            # first, reset the decoder state
            dec.start(hypo.state)
            # update decoder state with the previous generated word
            last_id = hypo.perm[-1]  # here it's just the position
            # we get the actual word
            word_id = fids[last_id]
            # then turn it to a vector
            last_inpt = out_embeddings[word_id]
            # now, update decode
            new_state = dec.step(last_inpt)
            #compute the log output
            log_prob = logsoftmax(dot(out_layer, new_state))
            # actually compute it
            log_prob = compute_value(log_prob)
            # get out top beam_size log prob
            """Add your code here"""
            for j,prob in enumerate(log_prob):
                perm = list(hypo.perm)
                #if not covered yet, extend the permutation
                if(j not in perm):
                    perm.append(j)
                    new_prob = hypo.score + prob
                    new_h = Hypothesis(perm,new_prob,new_state)
                    curr_stack.append(new_h)

        #if there more then 100 permutations, take the top 100
        if len(curr_stack) > beam_size:
            ordered_scores = []
            #get the scores from the hypothesis
            for h in curr_stack:
                ordered_scores.append(h.score)
            #get the indexs from 100 highest scores
            locations = sorted(range(len(ordered_scores)), key=lambda i: ordered_scores[i])[-beam_size:]
            #put top 100 in the bin
            stacks = [curr_stack[i] for i in locations]
            bin.append(stacks)
        else:
            bin.append(curr_stack)

    ordered_scores = []
    last_stack = bin[-1]
    #get the scores from the hypothesis
    for h in last_stack:
        ordered_scores.append(h.score)

    #return the hypothesis with highest score
    return last_stack[ordered_scores.index(max(ordered_scores))].perm[1:]

###########
#create preordered sentences
###########

original_test = open('data/low/kyoto-test.ja','r')
preordered_test = open('data/preordered/kyoto-test.ordered.att1.ja','w')

for sent in test:
    original = original_test.readline()
    orig_words = []
    for word in original.split():
        orig_words.append(word)
    hyp = decode(sent,100)
    for perm in hyp:
        preordered_test.write(orig_words[perm]+" ")
    preordered_test.write("\n")

original_test.close()
preordered_test.close()

original_dev = open('data/low/kyoto-dev.ja','r')
preordered_dev = open('data/preordered/kyoto-dev.ordered.att1.ja','w')

for sent in valid_data:
    original = original_dev.readline()
    orig_words = []
    for word in original.split():
        orig_words.append(word)
    hyp = decode(sent,100)
    for perm in hyp:
        preordered_dev.write(orig_words[perm]+" ")
    preordered_dev.write("\n")

original_dev.close()
preordered_dev.close()

tune = lm.read_data("data/low/kyoto-tune.ja")
original_tune = open('data/low/kyoto-dev.ja','r')
preordered_tune = open('data/preordered/kyoto-dev.ordered.att1.ja','w')

for sent in tune:
    original = original_tune.readline()
    orig_words = []
    for word in original.split():
        orig_words.append(word)
    hyp = decode(sent,100)
    for perm in hyp:
        preordered_tune.write(orig_words[perm]+" ")
    preordered_tune.write("\n")

original_tune.close()
preordered_tune.close()