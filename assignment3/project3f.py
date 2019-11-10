import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from statistics import mean
from cycler import cycler

#amount of times neruon runs for
runs = 10000
#bolean value to check of Total Error is below threshold
error_cut_off = False

#Type of check for error threshold
#error_type = 'any' # cuts off threshold if any value hits below threshold
error_type = 'mean' #  uses mean error  if the mean value hits below threshold


#normalization type for this function
# n_type = 'standard'
n_type = 'normal' # for only time variable normalized
#n_type = 'normal2' # for time and power variables normalized

# sets alpha and error threshold based on type of normalization configured
# learning constant is alpha
if(n_type == 'standard'):
	errorThreshold = [40,30,2]
	alpha = 0.0001

elif(n_type == 'normal'):
	errorThreshold = [53,35,2.5]
	alpha = 0.1
else:
	errorThreshold = [0.0001,0.000001,0.000001]
	alpha = 0.01

def randomizeWeights(size):
	num = 0.5
	weights = [rand.uniform(-num, num) for x in range(size)]
	return weights

def getOutput(weights, type, value):
	if type == 1:
		a = weights[0]
		b = weights[1]
		return a * value + b
	elif type == 2:
		a = weights[0]
		b = weights[1]
		c = weights[2]
		return a * (value ** 2) + b * value + c
	elif type == 3:
		a = weights[0]
		b = weights[1]
		c = weights[2]
		d = weights[3]
		return a * (value ** 3) + b * (value ** 2) + c * value + d

def getTotalError(actual, output):
	#print(actual, output, actual - output)
	errors = [(actual[i] - output[i]) ** 2 for i in range(len(actual))]
	return sum(errors)

def learning(weights, type, value, alpha, actual):
	if type == 1:
		output = getOutput(weights, type, value)
		pattern = [value, 1]

	elif type == 2:
		output = getOutput(weights, type, value)
		pattern = [ (value ** 2), value, 1]

	elif type == 3:
		output = getOutput(weights, type, value)
		pattern = [ (value ** 3), (value ** 2),  value, 1]

	else:
		return -1

	delta = 2 * alpha * (actual - output)
	weightChange = [delta * p for p in pattern]
	newWeights = [weights[i] + weightChange[i] for i in range(len(weights))]

	return newWeights

def print_results (tot_error, error_cut_off, num_runs,minErr, neuron_type):
	print ('Neuron {} is done'.format(neuron_type))
	print ('Training Error for Neuron {} :'.format(neuron_type))
	print ('Number of total possible runs  is :', runs)
	print ('Learning Constant is :', alpha)
	print ('Number of runs before cutoff is :',num_runs)
	print ('Error Threshold is :', minErr)
	if (error_cut_off):
		print("The error Threshold reached belowe the Total Error")
	else:
		print ("The Error Threshold was never reached and training was for max amount of itterations")
	for i in range(3):
		print ('Day {}: {}'.format((i+1),tot_error[i]))



def neuron1(time, power, alpha, minErr):
	weights = randomizeWeights(2)
	num_runs = 0
	error_cut_off = False
	neuron_type = 1
	for i in range(runs):
		print(i, '/', runs - 1, end = "\r")
		num_runs = i
		for j, day in enumerate(time):
			for k, t in enumerate(day):
				# output = getOutput(weights, 1, t)
				desired = power[j][k]
				weights = learning(weights, 1, t, alpha, desired)

		outputs = [[getOutput(weights, 1, hour) for hour in day] for day in time]
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		if (error_type == 'any'):
			if any(err < minErr for err in totErrs):
				error_cut_off = True
				break

		else:
			if mean(totErrs) < minErr:
				# print('Error threshold reached at iteration', i)
				error_cut_off = True
				break


	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)
	print()
	return weights

def neuron2(time, power, alpha,minErr):
	weights = randomizeWeights(3)
	error_cut_off = False
	num_runs = 0
	neuron_type = 2
	for i in range(runs):
		print(i, '/', runs - 1, end = "\r")
		num_runs = i
		for j, day in enumerate(time):
			for k, t in enumerate(day):

				# output = getOutput(weights, 2, t)
				desired = power[j][k]

				weights = learning(weights, 2, t, alpha, desired)

		outputs = [[getOutput(weights, 2, hour) for hour in day] for day in time]
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		if (error_type == 'any'):
			if any(err < minErr for err in totErrs):
				error_cut_off = True
				break

		else:
			if mean(totErrs) < minErr:
				# print('Error threshold reached at iteration', i)
				error_cut_off = True
				break



	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)
	print()
	return weights

def neuron3(time, power, alpha, minErr):
	weights = randomizeWeights(4)
	error_cut_off = False
	num_runs = 0
	neuron_type = 3
	for i in range(runs):
		print(i, '/', runs - 1, end = "\r")
		num_runs = i
		for j, day in enumerate(time):
			for k, t in enumerate(day):
				# output = getOutput(weights, 3, t)
				desired = power[j][k]
				weights = learning(weights, 3, t, alpha, desired)
		# array for all the outputs at each time
		outputs = [[getOutput(weights, 3, hour) for hour in day] for day in time]
		# array of each days of total errors
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		if (error_type == 'any'):
			if any(err < minErr for err in totErrs):
				error_cut_off = True
				break

		else:
			if mean(totErrs) < minErr:
				# print('Error threshold reached at iteration', i)
				error_cut_off = True
				break



	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)
	print()
	return weights

def normalizationMethod(data,type ):
	if(type == 'standard'):
		data['time'] = (data['time'] - data['time'].mean()) / np.std(data['time'])

	elif(type == 'normal'  ):
		data['time'] = (data['time'] - data['time'].min()) / (data['time'].max() - data['time'].min())

	else:
		data =( data-data.min())/(data.max()-data.min())

def drawLine(weights, type, name,ntype):
	if (ntype == 'standard'):
		x = np.arange(-1.75,1.75, 0.1)
	else:
		x = np.arange(0,1,0.01)
	y = [getOutput(weights, type, i) for i in x]
	plt.plot(x, y, label = name)


def plot_neuron (df, weightList, label):
	sns.lmplot(x = 'time', y = 'power', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(label)
	drawLine(weightList[0], 1, 'neuron 1',n_type)
	drawLine(weightList[1], 2, 'neuron 2',n_type)
	drawLine(weightList[2], 3, 'neuron 3',n_type)
	plt.legend()


#------#
# MAIN #
#------#

dataDay1 = pd.read_csv('train_data_1.txt', header = None)
dataDay2 = pd.read_csv('train_data_2.txt', header = None)
dataDay3 = pd.read_csv('train_data_3.txt', header = None)
dataDay4 = pd.read_csv('test_data_4.txt', header = None)

dataDay1.columns = ['time', 'power']
dataDay2.columns = ['time', 'power']
dataDay3.columns = ['time', 'power']
dataDay4.columns = ['time', 'power']


#normalize the data based on type
normalizationMethod(dataDay1,n_type)
normalizationMethod(dataDay2,n_type)
normalizationMethod(dataDay3,n_type)
normalizationMethod(dataDay4,n_type)


#comment out for testing
# add each time column to a list
time = list()
time.append(dataDay1['time'].tolist())
time.append(dataDay2['time'].tolist())
time.append(dataDay3['time'].tolist())
time.append(dataDay4['time'].tolist())

# add each power column to a list
power = list()
power.append(dataDay1['power'].tolist())
power.append(dataDay2['power'].tolist())
power.append(dataDay3['power'].tolist())
power.append(dataDay4['power'].tolist())



#create lists of the weights for each neuron based on the threshold
neuron1weights = neuron1(time[:3], power[:3], alpha, errorThreshold[0])
neuron2weights = neuron2(time[:3], power[:3], alpha, errorThreshold[1])
neuron3weights = neuron3(time[:3], power[:3], alpha, errorThreshold[2])

#list of the weights
weightList = (neuron1weights, neuron2weights, neuron3weights)

# #display training data and predicted power line
plot_neuron(dataDay1,weightList, 'Day 1')
plot_neuron(dataDay2,weightList, 'Day 2')
plot_neuron(dataDay3,weightList, 'Day 3')


#display predicted power, predicted power line, and actual power
plot_neuron(dataDay4,weightList, 'Test Data')

#test against 4th day
testOutput1 = [getOutput(neuron1weights, 1, hour) for hour in time[3]]
testOutput2 = [getOutput(neuron2weights, 2, hour) for hour in time[3]]
testOutput3 = [getOutput(neuron3weights, 3, hour) for hour in time[3]]

print('Test Error')
print('Neuron 1:', getTotalError(power[3], testOutput1))
print('Neuron 2:', getTotalError(power[3], testOutput2))
print('Neuron 3:', getTotalError(power[3], testOutput3))

plt.legend()
plt.show()
