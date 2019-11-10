import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from cycler import cycler

#---------#
#functions
#---------#

#calculates true and false positives and negatives
#returns a tuple consiting of the number of each and a list of the guesses
def testAccuracy(data, weights):
	height = data['height']
	weight = data['weight']
	sex = data['sex']
	guessList = []
	
	truePos = 0
	trueNeg = 0
	falsePos = 0
	falseNeg = 0
	
	for index, value in enumerate(height):
		guess = guessSex(data, index, weights)
		guessList.append(guess)
		
		#if actual value is male
		if sex[index] == 0:
			if guess == sex[index]:
				truePos += 1
			else:
				falseNeg += 1
		elif sex[index] == 1:
			if guess == sex[index]:
				trueNeg += 1
			else:
				falsePos += 1
	
	return (truePos, trueNeg, falsePos, falseNeg, guessList)
	
#guesses the sex based on height and weight
#returns 0 if male, 1 if female
def guessSex(data, index, weights):
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	height = data['height']
	weight = data['weight']
	sex = data['sex']
	
	threshold = a * height[index] + b * weight[index] + c
	
	#print(a, b)
	# #if less than threshold, expect female
	# if weight[index] <= threshold:
		# return 1
	# #else if greater than threshold, expect male
	# else:
		# return 0
		
	if threshold < 0:
		return 1
	else:
		return 0
		
def learning(data, w, learningConst):
	# male = data[0]
	# female = data[1]
	
	# totalError = 0
	
	# k = 0.1
	
	# for i in range(max(len(male.index), len(female.index))):
		# if i < len(male.index):
			# guess = guessSex(male, i, w)
			# lrn = learningConst * (male['sex'][i] - guess)
			
			# totalError += (male['sex'][i] - guess) ** 2
			
			# newWeight = (male['height'][i] * lrn, male['weight'][i] * lrn, guess * lrn)
			
			# w = np.subtract(w, newWeight)
		# if i < len(female.index):
			# guess = guessSex(female, i, w)
			# lrn = learningConst * (female['sex'][i] - guess)
			
			# totalError += (female['sex'][i] - guess) ** 2
			
			# newWeight = (female['height'][i] * lrn, female['weight'][i] * lrn, guess * lrn)
			
			# w = np.subtract(w, newWeight)
	
	# return (w, totalError)
	
	
	height = data['height']
	weight = data['weight']
	sex = data['sex']
	totalError = 0
	
	k = 0.1
	
	#FOR SOFT USE 1 / (1 + math.e ** (-k * net))
	#SOFT: alpha * error * input
	
	#print(w[0], w[1])
	
	
	for index, value in enumerate(height):
		output = guessSex(data, index, w)
		
		lrn = learningConst * (sex[index] - output)
		
		#print(lrn, learningConst * (sex[index] - output), learningConst)
		
		totalError += (sex[index] - output) ** 2
		
		newW = (height[index] * lrn, weight[index] * lrn, output * lrn)
		
		if lrn != 0:
			print(w, newW, sex[index], output)
		
		w = np.subtract(w, newW)
		
		# if lrn != 0:
			# print(learningConst, (sex[index] - output), lrn, w, newW)
		
		# newD = tuple(j * lrn for j in w)
		# # print(index, w, output, dw, newD)
		# w = np.subtract(w, newD)
	
	return (w, totalError)

def repeatLearning(data, weights, learningConst, name, color = 0):
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	line_boundary_x = np.amax([abs(np.amin(data['height'])), abs(np.amax(data['height']))])
	line_boundary_y = np.amax([abs(np.amin(data['weight'])), abs(np.amax(data['weight']))])
	
	x = np.arange(-line_boundary_x, line_boundary_x)
	y = -(a / b) * x - c
	
	plt.plot(x, y, label = name)
	
	accuracy = testAccuracy(data, weights)
	
	# truePos = accuracy[0]
	# trueNeg = accuracy[1]
	# falsePos = accuracy[2]
	# falseNeg = accuracy[3]
	# guessList = accuracy[4]
	
	# truePosRate = truePos / (truePos + falseNeg)
	# trueNegRate = trueNeg / (trueNeg + falsePos)
	# falsePosRate = falsePos / (trueNeg + falsePos)
	# falseNegRate = falseNeg / (truePos + falseNeg)
	
	# accRate = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
	# errRate = 1 - accRate
	
	# print('----', color)
	# print('True Positive:', truePos, truePosRate)
	# print('True Negative:', trueNeg, trueNegRate)
	# print('False Positive:', falsePos, falsePosRate)
	# print('False Negative:', falseNeg, falseNegRate)
	# print('Total Error:',  errRate)
	
def runData(filename, title, errThresh):
	df = pd.read_csv(filename)
	title = title
	
	df.columns = ['height', 'weight', 'sex']
	
	df['height'] = (df['height'] - df['height'].mean()) / np.std(df['height'])
	df['weight'] = (df['weight'] - df['weight'].mean()) / np.std(df['weight'])
	
	dataM = df[df['sex'] == 0].sample(frac = 1)
	dataF = df[df['sex'] == 1].sample(frac = 1)
	
	dataM = dataM.reset_index(drop = True)
	dataF = dataF.reset_index(drop = True)
	
	#makes the male and female entries alternate
	data = pd.concat([dataM, dataF]).sort_index(kind = 'merge')
	
	#initialize random weights
	#formula: ax + by + c > 0
	a = rand.uniform(-0.5, 0.5)
	b = rand.uniform(-0.5, 0.5)
	c = rand.uniform(-0.5, 0.5)

	weights = (a, b, c)

	sns.lmplot(x = 'height', y = 'weight', data = df, hue = 'sex', palette = 'GnBu', fit_reg = False).fig.suptitle(title)
	
	print('=====', title, '=====')
	plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])))

	learningConst = 0.3
	
	for i in range(10):
		repeatLearning(df, weights, learningConst, i)
		result = learning(df, weights, learningConst)
		weights = result[0]
		totErr = result[1]
		print(i, totErr)
		if totErr < errThresh:
			break
	

	print()
	
	line_boundary_x = np.amax([abs(np.amin(df['height'])), abs(np.amax(df['height']))]) + 0.5
	line_boundary_y = np.amax([abs(np.amin(df['weight'])), abs(np.amax(df['weight']))]) + 0.5
	
	plt.xlim(xmin = -line_boundary_x, xmax = line_boundary_x)
	plt.ylim(ymin = -line_boundary_y, ymax = line_boundary_y)
	
	plt.legend()
	plt.show()

#---------#
#main program
#---------#


runData('groupA.txt', 'Group A', 10 ** -5)
runData('groupB.txt', 'Group B', 10 * -1)
runData('groupC.txt', 'Group C', 5 * 10 ** -1)

