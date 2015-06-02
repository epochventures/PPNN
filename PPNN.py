"""PPNN.py: Neural Network implementation in Pure Python - (PPNN) Pure Python Neural Network.

	The project is aimed at those who want to use the very good book on Neural Networks
	that has been used by many called 'Practical Neural Network Recipies in C++'
	ISBN: 978-0124790407, by Timothy Masters. This is a simple python implemention 
	that tries to tie in with the book but expressly for python without recourse to 
	using modules such as NUMPY or SCIPY, not that there is anything wrong in using these 
	fast modules, but here simply stay faithful to the implementations within the book.

	As an example (like the book) we illustrate a working manifestation of the XOR example
	and also provide the training example which are generated on the fly and then used to
	train the network. After training the network you are free to test this network.

	In PPNN networks exist as python dictionaries, which means transporting them is very easy
	as is examing each and every weight. For example a working neural network that 
	performs that function of XOR looks like this:

	{'output': {0: {0: -8.30648077083447, 1: 3.6378400405424904, 2: 14.006753208219276, 3: 1.895120416489752}}, 
	'hidden': {0: {0: -10.513505383255971, 1: 10.913534176278933, 2: 0.24543391390819042}, 
	1: {0: 2.4869136576399664, 1: -0.3450042945080538, 2: -4.722548494099791}, 
	2: {0: -3.685140869118192, 1: 4.433265048621298, 2: -3.1641081562711477}}}

	Caveats: This code is not designed to be super quick but as a way to get up and running
	as quickly as possible with python. I'm sure there are better ways to do many of the functions
	but I'm sure this will be updated with time. The point is to be able to fiddle and play
	with Neural Networks without treating them as some kind of magical black box, but to be
	able to delve inside, tinker and play.



"""

__author__      = "Brett Donovan"
__copyright__   = "Copyright 2013"



import os, re, sys, math, random, time, datetime
#from os import walk
from copy import deepcopy

### Activation Functions

def grad_activation(x):
	fx = activation(x)
	return (fx * (1.0 - fx))

def activation(x):
	op = 1.0 / (1.0 + math.exp(-x))	
	return op

### Build Initial Weights and Structure

def buildNN(Ninput, Nhidden, Noutput, w_scale):
	random.seed()
	w_dict = {}
	w_dict['hidden'] = {}
	w_dict['output'] = {}

	for j in xrange(0, Nhidden):
		w_dict['hidden'][j] = {}
		for i in xrange(0, Ninput+1): # add in extra bias weight
			w_dict['hidden'][j][i] = w_scale * random.uniform(-1, 1)
	for k in xrange(0, Noutput):
		w_dict['output'][k] = {}
		for j in xrange(0, Nhidden+1): # add in extra bias weight
			w_dict['output'][k][j] = w_scale * random.uniform(-1, 1)	
	#print w_dict	
	return w_dict

def dotprod(A, B): # dictionaries
	dp = 0.0
	if len(A) != len(B):
		print "Major error, vectors don't match..."
		print len(A), len(B)
	for a in A.keys(): ## loop over keys
		dp+=A[a]*B[a]
	return dp		



def computeFF(w_dict, input_dict):
	
	#tmp_input = deepcopy(input_dict)
	act = {}
	act['hidden'] = {}
	act['output'] = {}
	bias_i = len(w_dict['hidden'][0]) - 1 ## Add in pure bias here
	#tmp_input[bias_i] = 1.0 ## FULL ACTIVATION
	input_dict[bias_i] = 1.0
	for j in w_dict['hidden']:
		dp = dotprod(w_dict['hidden'][j], input_dict)
		act['hidden'][j] = activation(dp)
	bias_h = len(w_dict['output'][0]) - 1
	act['hidden'][bias_h] = 1.0
	for k in w_dict['output']:
		dp = dotprod(w_dict['output'][k], act['hidden'])
		act['output'][k] = activation(dp)
	act['input'] = input_dict
	del act['hidden'][bias_h]
	del input_dict[bias_i]
	return act


def computeMSE(target, actual):
	tot = 0.0
	for i in target:
		tot += (target[i] - actual[i])**2.0
	#print tot, len(target)
	return (tot / float(len(target)))
	

# GRADIENT FUNCTIONS

def grad_output(act, target, grad_dict, w_dict):
	bias_h = len(w_dict['hidden'])
	#print "bias_h", bias_h
	deltas = {}
	for k in act['output']:
		delta = (target[k] - act['output'][k]) * grad_activation(act['output'][k]) 
		deltas[k] = delta
		for j in w_dict['hidden']:
			
			grad_dict['output'][k][j] += delta * act['hidden'][j] 
		grad_dict['output'][k][bias_h] += delta
	return grad_dict, deltas


def grad_hidden(act, target, grad_dict, w_dict, deltas_o):

	bias_i = len(act['input'])
	bias_h = len(act['hidden'])
	act['input'][bias_i] = 1.0
	act['hidden'][bias_h] = 1.0

	delta = {}
	#w_dict['hidden'].keys()[0]
	for j in w_dict['hidden'][0]: 
		
		sum_d = 0.0
		for k in w_dict['output']:
			sum_d += deltas_o[k] * w_dict['output'][k][j]

		delta[j] = sum_d * grad_activation(act['hidden'][j]) 
		

	for j in w_dict['hidden']: 

		for i in w_dict['hidden'][0]:
			#print j, i
			grad_dict['hidden'][j][i] += delta[j] * act['input'][i] 			
		
		
		#grad_dict['hidden'][j][bias_i] += delta[j]
	
	del act['input'][bias_i]
	del act['hidden'][bias_h]
	return grad_dict


def InitializeGrad(w_dict):
	
	grad = {}
	grad['output'] = {}
	grad['hidden'] = {}
	for k in w_dict['output']:
		grad['output'][k] = {}
		for j in w_dict['output'][k]:
			grad['output'][k][j]  = 0.0

	for j in w_dict['hidden']:
		grad['hidden'][j] = {}
		for i in w_dict['hidden'][j]:
			grad['hidden'][j][i]  = 0.0
	
	return grad
	

def UpdateWeights(w_dict, grad, pgrad, rate, mom):
	## do output first
	max_grad = 0
	for k in w_dict['output']:
		for j in w_dict['output'][k]:
			corr = rate * grad['output'][k][j] + mom * pgrad['output'][k][j]
			
			w_dict['output'][k][j] += corr
			pgrad['output'][k][j] = corr
			if abs(grad['output'][k][j]) > max_grad:
				max_grad = grad['output'][k][j]

	for j in w_dict['hidden']:
		for i in w_dict['hidden'][j]:
			corr = rate * grad['hidden'][j][i] + mom * pgrad['hidden'][j][i]
			
			w_dict['hidden'][j][i] += corr
			pgrad['hidden'][j][i] = corr

			if abs(grad['hidden'][j][i]) > max_grad:
				max_grad = grad['hidden'][j][i]
	return w_dict, abs(max_grad)

	
def SumGrad(start, end, grad):
	grad_sum = {}
	grad_sum['output'] = {}
	for k in grad[start]['output']:
		grad_sum['output'][k] = {}
		for j in grad[start]['output'][k]:
			grad_sum['output'][k][j] = sum([float(grad[pres]['output'][k][j]) for pres in xrange(start, end)])
			
	grad_sum['hidden'] = {}
	for j in grad[start]['hidden']:
		grad_sum['hidden'][j] = {}
		for i in grad[start]['hidden'][j]:
			grad_sum['hidden'][j][i] = sum([float(grad[pres]['hidden'][j][i]) for pres in xrange(start, end)])
	
	return grad_sum

def printNN(w_dict):
	# Print out our NN in a more pretty format
	op_str = ""
	for k in xrange(0,len(w_dict['output'])):
		for j in xrange(0,len(w_dict['output'][k])):
		 	op_str += "["+str(k)+","+str(j)+ "]: " + str("{:.3f}".format(float(w_dict['output'][k][j]))) + " "
	print "OP: " + op_str
	op_str = ""
	for j in xrange(0,len(w_dict['hidden'])):
		for i in xrange(0,len(w_dict['hidden'][j])):
		 	op_str += "["+str(j)+","+str(i)+ "]: " + str("{:.3f}".format(float(w_dict['hidden'][j][i]))) + " "
	print "HD: " + op_str


def randomize(value, scale):
	if value <= 0:
		value += scale
	else:
		value -= scale
	return value	


def no_presentations(Npresentations, Nepochs):
	epoch_dict = {}
	current = Npresentations
	average_per_epoch = Npresentations / Nepochs
	print "Average presentations per Epoch: ", average_per_epoch
	for epoch in xrange(0, Nepochs):
		Nexamples = random.randint(1, average_per_epoch)
		
		if current - Nexamples > 0:
			current -= Nexamples
		else:
			Nexamples = current
			current = 0
		epoch_dict[epoch] = Nexamples
	return epoch_dict
	
def SplitEpoch(presentations, NoSplits):
	
	rand_lst = []
	tot_lst = []
	
	for no in xrange(0, NoSplits-1):
		rand = random.randint(0, presentations/NoSplits)
		if sum(rand_lst) + rand > presentations:
			rand_lst.append(0)
		else:
			rand_lst.append(rand)
	rand_lst.append(presentations - sum(rand_lst))
	random.shuffle(rand_lst)
	total = 0
	tot_lst.append(total)
	for i in xrange(1, NoSplits+1):
		total = sum(rand_lst[0:i])
		tot_lst.append(total)

	return tot_lst



def build_xor_train(Npresentations, Nepochs):
	scale = 0.1
	train_dict = {}
	epoch_dict = no_presentations(Npresentations, Nepochs)
	for epoch in epoch_dict:
		pres_per_epoch = epoch_dict[epoch]
		train_dict[epoch] = {}

		split_epoch = SplitEpoch(pres_per_epoch, 4)
		for i in xrange(0, len(split_epoch)-1):
			
			for pres in xrange(split_epoch[i], split_epoch[i+1]):
				a = random.randint(0,1)
				b = random.randint(0,1)
				c = a ^ b
				#print a, b, c
				train_dict[epoch][pres] = {}
				train_dict[epoch][pres]['input'] = {}
				train_dict[epoch][pres]['output'] = {}
				train_dict[epoch][pres]['input'][0] = randomize(a,scale)
				train_dict[epoch][pres]['input'][1] = randomize(b,scale)
				train_dict[epoch][pres]['output'][0] = randomize(c,scale)
			
	return train_dict		


def find_grad(epoch_train_dict, w_dict, rate, mom, grad=None):
	
	error = 0.0
	
	
	if grad == None:
		grad = {}
		grad = InitializeGrad(w_dict)
		print "Initalized gradients..."
		
	mse_dict = {}
	mse_dict[0] = {}
	count = 0
	for pres in epoch_train_dict:
		pgrad = deepcopy(grad)		
		count+=1
		act_dict = {}
		act_dict = computeFF(w_dict, epoch_train_dict[pres]['input'])
		grad, deltas_o = grad_output(act_dict, epoch_train_dict[pres]['output'], grad, w_dict)
		grad = grad_hidden(act_dict, epoch_train_dict[pres]['output'], grad, w_dict, deltas_o)


		mse_dict[pres] = computeMSE(epoch_train_dict[pres]['output'], act_dict['output'])
	if count > 0:
		#print "Update Weights..."
		w_dict, max_grad = UpdateWeights(w_dict, pgrad, grad, rate, mom)
	
	if len(mse_dict) > 0:
		error = sum(mse_dict.values())/float(len(mse_dict))
	
	return float(error), grad, w_dict




def evaluate_XOR(w_dict):
	#net = {}
	net = computeFF(w_dict, {0:0.0, 1:0.0})
	print {0:0.0, 1:0.0}, net['output']
	#net = {}	
	net = computeFF(w_dict, {0:0.0, 1:1.0})
	print {0:0.0, 1:1.0}, net['output']
	#net = {}
	net = computeFF(w_dict, {0:1.0, 1:0.0})
	print {0:1.0, 1:0.0}, net['output']
	#net = {}
	net = computeFF(w_dict, {0:1.0, 1:1.0})
	print {0:1.0, 1:1.0}, net['output']



if __name__ == "__main__":

	#### Neural Network Structure Parameters for three layers ####
	
	Ninput = 2
	Nhidden = 2
	Noutput = 1
	Weight = 4 #Range of weights for each node, between -Weight and Weight

	##############################################################


	output = False
	Npresentations = 4000000
	Nepochs = 100000
	grad_tol = 0.001
	error_tol = 0.0001
	rate = 0.02
	mom = 0.2
	fraction_epoch = 0.10

	min_epoch = fraction_epoch * Nepochs
	batch_update = False # update after iteration

	xor_train = build_xor_train(Npresentations, Nepochs)
	w_dict = buildNN(Ninput, Nhidden, Noutput, Weight)
	

	w_init = deepcopy(w_dict)
	if output:
		print "Initial Evaluation"
		evaluate_XOR(w_dict)
		print "============================"
	grad = None
	total_pres = 0
	exit_type = 'NO'
	Nepoch_exit = fraction_epoch*Nepochs
	Nepoch_exit = min_epoch
	with open("error.txt", 'w+') as fout:    		
		for epoch in xor_train:
			if len(xor_train[epoch]) > 0:
				if output:
					print "Epoch: ", epoch, " Presentations: ", len(xor_train[epoch]), " Epoch: ", epoch, " of: ", len(xor_train)
					
					print "========BEFORE========="	
					printNN(w_dict)
					print "========BEFORE========="		
				error, grad, w_dict = find_grad(xor_train[epoch], w_dict, rate, mom, grad)
				print "Error: ", error
				total_pres += len(xor_train[epoch])
				if output:
					print "========AFTER========="	
					printNN(w_dict)
					print "========AFTER========="	
					print "Error: ", error
				
				fout.write(str(total_pres) + " " + str(error)+"\n")
				if error <= error_tol and epoch > Nepoch_exit:
					print "E: ", error, 
					print total_pres
					print "==================================================================="
					print "Error Tolerance Reached"
					print "==================================================================="
					exit_type = 'MSE'
					break
		if exit_type == 'MSE':
			evaluate_XOR(w_dict)
	
	print "Initial"
	evaluate_XOR(w_init)
	print "Final"
	evaluate_XOR(w_dict)

	print w_dict

	print time.ctime()
