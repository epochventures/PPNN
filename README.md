# PPNN
Pure Python Neural Network



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
