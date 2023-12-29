# Tester for the perceptron classifier on training data 

import numpy as np
import sys

def main():
	test_me = True
	if len(sys.argv) < 1:
		print("Usage: python test-me.py <hyp-file>")
		exit()

	X = np.loadtxt('a2-test-data.txt')
	Y = np.loadtxt('a2-test-label.txt')
	Weight = np.loadtxt(sys.argv[1], dtype=complex)

	if len(X[0]) < len(Weight):
		Bias = Weight[-1]
		Weight = Weight[:-1]
	else:
		Bias = 0.0

	num_errors = 0
	if test_me:
		print("Testing Perceptron ...") 
		for i in range(len(X)): 
			X[i] = X[i]/np.linalg.norm(X[i])
			if Y[i] * (np.dot(Weight, X[i]) + Bias) < 0:
				num_errors = num_errors + 1

		print("Number of errors = ", num_errors)

######################################################################################

if __name__ == "__main__":
    main()

	

