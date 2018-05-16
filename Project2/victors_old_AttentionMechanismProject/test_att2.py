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

source_file_train = "data/low/kyoto-train.cln.ja"
source_order_train = "data/align/kyoto-train.cln.align.ja"
source_file_valid = "data/low/kyoto-dev.ja"
source_order_valid = "data/align/kyoto-dev.align.ja"
source_file_test = "data/low/kyoto-test.ja"
source_order_test = "data/align/kyoto-test.align.ja"

vocab_size = 30000  # short list of words

# make data
data = lm.read_data(source_file_train)
target = lm.read_data(source_order_train)

valid_data = lm.read_data(source_file_valid)
valid_target = lm.read_data(source_order_valid)

test_data = lm.read_data(source_file_test)
test_target = lm.read_data(source_order_test)

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

#the trained model is loaded
load_model(open("data/trained/trained_network_att2.npy"))

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
    out_layer = encode_birnn(fids)
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
            last_inpt = embeddings[word_id]
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

def hammingDistance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(bool(ord(ch1) - ord(ch2)) for ch1, ch2 in zip(s1, s2))

#testing
correct_test = 0.
num_sents = 0.
num_words = 0.
kendall = 0.
num_sents_without = 0.
kendall_without = 0.
hamming = 0

for sent,tar in zip(test_data,test_target):
    num_sents += 1
    num_words += len(sent)
    assert(len(sent) == len(tar))
    hyp = decode(sent,100)
    hyp = map(str,hyp)
    tar = tar[:-1]
    kend = 1
    if len(hyp) > 1:
        kend = scipy.stats.stats.kendalltau(hyp,tar)[0]
        kendall_without += kend
        num_sents_without += 1
    kendall += kend
    print kend
    #hamming += hammingDistance(hyp,tar)
    #print "correct order: [%s]" % ",".join(map(str,tar))
    #print "created order: [%s]" % ",".join(map(str,hyp))
    #print "\n###############\n"
    for h,w in zip(hyp,tar):
        if h == w:
            correct_test += 1
print "=========================\nRESULTS\n========================="
print "Total Kendall Tau: "
print kendall
print "Number of sentences: "
print num_sents
final_kendall = kendall/num_sents
print "Kendall Tau average:"
print final_kendall
print "\nNumber of sentences larger then 1:"
print num_sents_without
print "Kendall Tau without sentences of length 1:"
print kendall_without
final_kendall_without = kendall_without/num_sents_without
print "Kendall Tau without 1 average:"
print final_kendall_without
print "Correct words:\t%d" % (correct_test)
print "Total number of words:\t%d" % (num_words)
#print "Hamming accuracy:\t%d%%" % ((hamming/num_words)*100)
print "accuracy of test: "
print ((correct_test/num_words)*100)
