import numpy as np
from penne import *
from penne import recurrent
import random

def generate_sequence(length):
	seq = [length]
	for x in xrange(length):
		seq.append(random.choice([0,1]))
	seq.append(2)
	return seq

def make_network(digits):
	loss = constant(0.)
	r.start()
	for i in xrange(len(digits)-1):
		h = r.step(digits[i])
		o= output_layer(h)
		loss -= o[digits[i+1]]
	return loss

def print_seqs(seqs):
	for seq in seqs:
		print seq

if __name__ == "__main__":
	#Generate (pseudo-)random binary sequences
	sequences =	[generate_sequence(y) for y in xrange(20, 31)]

	#Set up the RNN
	nh = 100
	r = recurrent.LSTM(-256, nh)
	output_layer =  Layer(nh, 3, f=logsoftmax)
	trainer = Adagrad(learning_rate=0.1)

	for epoch in xrange(100):
		l = 0.

		#Train model
		for seq in sequences:
			loss = make_network(seq)
			l += trainer.receive(loss)
		print "Loss: {}".format(l)
		print "This is the outputsequences of epoch {}".format(epoch)

		#Regenerate sequences
		for i in xrange(20,31):
			c = i
			r.start()
			values = {}
			outputseq = [i]
			for t in xrange(31):
				h = r.step(c)
				o = output_layer(h)
				values = compute_values(o, values)
				c = np.argmax(np.random.multinomial(1, np.exp(values[o])))
				if c == 2: break
				outputseq.append(c)
			outputseq.append(2)
			print outputseq
		print ""
	print "These were the sentences that had to be copied:"
	print_seqs(sequences)