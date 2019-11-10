import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from statistics import mean
from cycler import cycler

#amount of times neruon runs for
runs = 20000
#bolean value to check of Total Error is below threshold
error_cut_off = False

#learning constant
# alpha = 0.01

#normalization type for this function
# n_type = 'standard'
n_type = 'normal'
#print(n_type)

# normalization method
# sets type of normalization method in normalization class
n_method = 1 # normalizes time only
# n_method = 2 # normalizes both

if(n_type == 'standard'):
	errorThreshold = [40,30,2]
	alpha = 0.0001

if(n_method == 1):
	errorThreshold = [45,25,2]
	alpha = 0.1
else:
	errorThreshold = [0.0001,0.000001,0.000001]
	alpha = 0.01


def randomizeWeights(size):
	num = 0.5
	weights = [rand.uniform(-num, num) for x in range(size)]
	return weights

# def errThreshCheck (error):

# 	if (error < errorCheck):
# 		return

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
		#pattern = [value, value, c]
	elif type == 3:


		output = getOutput(weights, type, value)

		pattern = [ (value ** 3), (value ** 2),  value, 1]
		#pattern = [value, value, value, d]
	else:
		return -1


	delta = 2 * alpha * (actual - output)
	weightChange = [delta * p for p in pattern]

	newWeights = [weights[i] + weightChange[i] for i in range(len(weights))]

	if type == -1:
		print(actual, output)
		print('delta', delta, '\nweight change', weightChange)
		print(weights)
		print(newWeights, '\n')

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
				# totErr1 = getTotalError(desired, output)
				weights = learning(weights, 1, t, alpha, desired)

		outputs = [[getOutput(weights, 1, hour) for hour in day] for day in time]
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		# if mean(totErrs) < minErr:
		# 	break
		if any(err < minErr for err in totErrs):
			error_cut_off = True
			break


	# print('The total Error for Neuron 1 is :',totErr1)
	# print('Neuron 1 done')
	#
	# print('Training Error Neuron 1')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])

	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)

	print()

	return weights

def neuron2(time, power, alpha,minErr):
	weights = randomizeWeights(3)
	# oldWeights = weights.copy()
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
				# totErr2 = getTotalError(desired, output)
				weights = learning(weights, 2, t, alpha, desired)
				# print(totErr2)
		outputs = [[getOutput(weights, 2, hour) for hour in day] for day in time]
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		# if mean(totErrs) < minErr:
		# 	break
		if any(err < minErr for err in totErrs):
			error_cut_off = True
			break



	# x = np.arange(5, 20)
	# y = [getOutput(oldWeights, 2, i) for i in x]
	# plt.plot(x, y, label = 'old')
	# print ('The total Error for Neuron 2 is :',totErr2)
	# print('Neuron 2 done')
	#
	# print('Training Error Neuron 2')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])

	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)
	print()

	return weights

def neuron3(time, power, alpha, minErr):
	weights = randomizeWeights(4)
	error_cut_off = False
	# oldWeights = weights.copy()
	num_runs = 0
	neuron_type = 3
	for i in range(runs):
		print(i, '/', runs - 1, end = "\r")
		num_runs = i
		for j, day in enumerate(time):
			for k, t in enumerate(day):
				# output = getOutput(weights, 3, t)
				desired = power[j][k]
				# timecurrent = time[j][k]
				# totErr3 = getTotalError(desired, output)
				weights = learning(weights, 3, t, alpha, desired)
				# errThreshCheck(totErr3)
		outputs = [[getOutput(weights, 3, hour) for hour in day] for day in time]
				# array of each days of total errors
		totErrs = [getTotalError(power[d], outputs[d]) for d in range(len(power))]

		if any(err < minErr for err in totErrs):
			error_cut_off = True
			break

				#testing purposes
				# print ("The time is :", timecurrent)
				# print ("The desired is:", desired)
				# print ("The output is :", output)
				# print ("the total error is ", totErr)
				# if (timecurrent == 0 or timecurrent == 1):
				# 	print ("The weight at time {} is {}".format(timecurrent, weights))




	print('Neuron 3 done')

	print('Training Error Neuron 3')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])

	print_results(totErrs,error_cut_off,num_runs, minErr,neuron_type)
	print()
	return weights

def drawLine(weights, type, name,ntype):
	if (ntype == 'standard'):
		x = np.arange(-1.75,1.75, 0.1)
	else:
		x = np.arange(0,1,0.1)
	y = [getOutput(weights, type, i) for i in x]
	plt.plot(x, y, label = name)

# def normalizeInputs (df,method = 1,feature = 'None'):
#
# 	# if (method == 1):
# 	# 	for feature in df.columns:
# 	# 		max_val = df[feature].max()
# 	# 		min_val = df[feature].min()
# 	# 		df[feature] = (df[feature] - min_val) / (max_val - min_val)
# 	# 	print("method 1")
# 	# 	return df
#
# 	if(method == 1):
# 		df['time'] = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min())
# 		# df['power'] = (df['power'] - df['power'].min()) / (df['power'].max() - df['power'].min())
# 		# print("method 1")
# 		return df
# 	else:
# 		normalized_df =( df-df.min())/(df.max()-df.min())
# 		# print("method 2")
# 		return normalized_df
#
#
# def standarnormalization (df):
# 	df['time'] = (df['time'] - df['time'].mean()) / np.std(df['time'])
# 	#df['power'] = (df['power'] - df['power'].mean()) / np.std(df['power'])
# 	return df

def readcsv(filename):
	df = pd.read_csv(filename, header = None)
	# df.columns = ['time', 'power']
	return df

# def column_name(data):
# 	data.columns = ['time', 'power']
# 	return data


def normalizationMethod(data,type,method ):
	if(type == 'standard'):
		# data = standarnormalization(data)
		data['time'] = (data['time'] - data['time'].mean()) / np.std(data['time'])


	elif(type == 'normal' and method == 1 ):
		# data = normalizeInputs(data,method)
		data['time'] = (data['time'] - data['time'].min()) / (data['time'].max() - data['time'].min())
		# data = normalizeInputs(data,'power',method)
		# data = normalizeInputs(data)
		# return data

	# elif(type == 'normal' and )

	else:
		# data = normalizeInputs(data,method)
		data =( data-data.min())/(data.max()-data.min())
		# print ("error")
		# return data





def prepare_data(filename, type):
	data = readcsv(filename)
	# data = column_name(data)
	# normalizationMethod(data,type)
	return data






dataDay1 = pd.read_csv('train_data_1.txt', header = None)
dataDay2 = pd.read_csv('train_data_2.txt', header = None)
dataDay3 = pd.read_csv('train_data_3.txt', header = None)
dataDay4 = pd.read_csv('test_data_4.txt', header = None)

dataDay1.columns = ['time', 'power']
dataDay2.columns = ['time', 'power']
dataDay3.columns = ['time', 'power']
dataDay4.columns = ['time', 'power']

# print("before normalization")
# # print ("day one time is :\n")
# # print (dataDay1['time'])
# print ("day one power is :\n")
# print (dataDay1['power'])
# # print ("alpha is :", alpha)

# normalization type
#n_type = normalizationMethod(n_type)



normalizationMethod(dataDay1,n_type,n_method)
normalizationMethod(dataDay2,n_type,n_method)
normalizationMethod(dataDay3,n_type,n_method)
normalizationMethod(dataDay4,n_type,n_method)
#print(n_type)

# For testing purposes
# test_1 = normalizationMethod(dataDay1,n_type,1)
# print("after normalization method 1\n")
# # print ("day one time is :\n")
# # print (test_1['time'])
# print ("day one power is :\n")
# print (test_1['power'])
# # print ("alpha is :", alpha)
#
# test_2 = normalizationMethod(dataDay1,n_type,2)
# print("after normalization method 2\n")
# # print ("day one time is :\n")
# # print (test_2['time'])
# print ("day one power is :\n")
# print (test_2['power'])
# # print ("alpha is :", alpha)
#
#
# test_3 = normalizationMethod(dataDay1,n_type,3)
# print("after normalization method 3\n")
# # print ("day one time is :\n")
# # print (test_3['time'])
# print ("day one power is :\n")
# print (test_3['power'])
# # print ("alpha is :", alpha)


#comment out for testing
time = list()
time.append(dataDay1['time'].tolist())
time.append(dataDay2['time'].tolist())
time.append(dataDay3['time'].tolist())
time.append(dataDay4['time'].tolist())

power = list()
power.append(dataDay1['power'].tolist())
power.append(dataDay2['power'].tolist())
power.append(dataDay3['power'].tolist())
power.append(dataDay4['power'].tolist())




neuron1weights = neuron1(time[:3], power[:3], alpha, errorThreshold[0])
neuron2weights = neuron2(time[:3], power[:3], alpha, errorThreshold[1])
neuron3weights = neuron3(time[:3], power[:3], alpha, errorThreshold[2])

# #display training data and predicted power line
# sns.lmplot(x = 'time', y = 'power', data = dataDay4, palette = 'plasma', fit_reg = False).fig.suptitle('Day 4')

def plot_neuron (df,label):
	sns.lmplot(x = 'time', y = 'power', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(label)
	drawLine(neuron1weights, 1, 'neuron 1',n_type)
	drawLine(neuron2weights, 2, 'neuron 2',n_type)
	drawLine(neuron3weights, 3, 'neuron 3',n_type)
	plt.legend()

plot_neuron(dataDay1, 'Day 1')
plot_neuron(dataDay2, 'Day 2')
plot_neuron(dataDay3, 'Day 3')
plot_neuron(dataDay4, 'Test Data')





testOutput1 = [getOutput(neuron1weights, 1, hour) for hour in time[3]]
testOutput2 = [getOutput(neuron2weights, 2, hour) for hour in time[3]]
testOutput3 = [getOutput(neuron3weights, 3, hour) for hour in time[3]]
print('Test Error')
print('Neuron 1:', getTotalError(power[3], testOutput1))
print('Neuron 2:', getTotalError(power[3], testOutput2))
print('Neuron 3:', getTotalError(power[3], testOutput3))

plt.legend()
plt.show()
