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
	
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	truePos = 0
	trueNeg = 0
	falsePos = 0
	falseNeg = 0
	
	guessList = a * height + b * weight + c
	guessList = np.where(guessList < 0, 1, 0)
	
	maleListDesired = (sex == 0)
	femaleListDesired = (sex == 1)
	
	maleListOutput = (guessList == 0)
	femaleListOutput = (guessList == 1)
	
	truePos = len(np.where(maleListOutput & maleListDesired)[0])
	trueNeg = len(np.where(femaleListOutput & femaleListDesired)[0])
	
	falsePos = len(np.where(maleListOutput & ~maleListDesired)[0])
	falseNeg = len(np.where(femaleListOutput & ~femaleListDesired)[0])
	
	truePosRate = truePos / (truePos + falseNeg)
	trueNegRate = trueNeg / (trueNeg + falsePos)
	falsePosRate = falsePos / (trueNeg + falsePos)
	falseNegRate = falseNeg / (truePos + falseNeg)
	
	accRate = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
	errRate = 1 - accRate
	
	print('----')
	print('True Positive:', truePos, truePosRate)
	print('True Negative:', trueNeg, trueNegRate)
	print('False Positive:', falsePos, falsePosRate)
	print('False Negative:', falseNeg, falseNegRate)
	print('Accuracy:', accRate)
	print('Error:',  errRate)
	
#guesses the sex based on height and weight
#returns 0 if male, 1 if female
def threshHard(row, weights):
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	height = row[1]
	weight = row[2]
	sex = row[3]
	
	net = a * height + b * weight + c
	
	if net <= 0:
		return 1
	else:
		return 0
		
def threshSoft(row, weights, k):
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	height = row[1]
	weight = row[2]
	sex = row[3]
	
	net = a * height + b * weight + c
	
	output = 1 / (1 + math.e ** (-k * net))
	
	return output
	
# def thresh2(data, weights):
	# height = data['height']
	# weight = data['weight']
	# sex = data['sex']
	
	# a = weights[0]
	# b = weights[1]
	# c = weights[2]
	
	# threshold = a * height + b * weight + c
	
	# threshold = np.where(threshold < 0, 1, 0)
	
	# return threshold
		
def learning(data, w, learningConst, mode):
	
	height = data['height']
	weight = data['weight']
	sex = data['sex']
	totalError = 0
	
	k = 0.1
	
	for row in data.itertuples():
		index = row[0]
		height = row[1]
		weight = row[2]
		sex = row[3]
		
		#determines what output to use depending on hardness or softness
		if mode == 'hard':
			output = threshHard(row, w)
		elif mode == 'soft':
			output = threshSoft(row, w, k)
		else:
			break
		
		lrn = learningConst * (sex - output)
		
		totalError = totalError + (sex - output) ** 2
		
		if lrn != 0 and output == 0:
			output = 1
		
		new_w = (height  * lrn, weight * lrn, output * lrn)
		
		w =  np.subtract(w, new_w)
	
	#quirk where in the soft activation function the number increases as it gets more accurate
	#this if statement fixes that
	if mode == 'soft':
		totalError = data.shape[0] - totalError
	
	return (w, totalError)

def drawLine(data, weights, learningConst, name, color = 0):
	a = weights[0]
	b = weights[1]
	c = weights[2]
	
	line_boundary_x = np.amax([abs(np.amin(data['height'])), abs(np.amax(data['height']))])
	line_boundary_y = np.amax([abs(np.amin(data['weight'])), abs(np.amax(data['weight']))])
	
	x = np.arange(-line_boundary_x, line_boundary_x)
	print(a, b, c)
	y = (-a * x - c) / b
	
	plt.plot(x, y, label = name)
	
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
	
    #data is split
    
	#makes the male and female entries alternate
	data = pd.concat([dataM, dataF]).sort_index(kind = 'merge')
	data = data.reset_index(drop = True)
	
    #split dataset
    #data_split = 
    
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
	
	#for drawing each line
	drawLine(df, weights, learningConst, 'original')
	
	for i in range(5000):
		result = learning(data, weights, learningConst, 'soft')
		weights = result[0]
		totalError = result[1]
		print(i, totalError)
		
		#EACH LINE INBETWEEN FIRST AND FINAL
		# drawLine(df, weights, learningConst, i + 1)
		
		if totalError < errThresh:
			break
	
	drawLine(df, weights, learningConst, 'final')
	
	testAccuracy(df, weights)
    #print("Weight A", a)
	print("Weight A:", a)
	print("Weight B:", b)
	print("Weight C:", c)
    
	line_boundary_x = np.amax([abs(np.amin(df['height'])), abs(np.amax(df['height']))]) + 0.5
	line_boundary_y = np.amax([abs(np.amin(df['weight'])), abs(np.amax(df['weight']))]) + 0.5
	
	plt.xlim(xmin = -line_boundary_x, xmax = line_boundary_x)
	plt.ylim(ymin = -line_boundary_y, ymax = line_boundary_y)
	
	plt.legend()
	plt.show()

#---------#
#main program
#---------#

oldErrThresh = (10 ** -5, 10 * -1, 5 * 10 ** -1)
newErrThresh = (10 ** -5, 100, 1.45 * 10 ** 3)
cusErrThresh = (10 ** -5, 50, 1.15 * 10 ** 3)

runData('groupA.txt', 'Group A', cusErrThresh[0])
runData('groupB.txt', 'Group B', cusErrThresh[1])
runData('groupC.txt', 'Group C', cusErrThresh[2])

