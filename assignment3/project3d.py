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
		#thresh = ax + b
		# a = weights[0]
		# b = weights[1]

		output = getOutput(weights, type, value)

		pattern = [value, 1]
	elif type == 2:
		#thresh = ax^2 + bx + c
		# a = weights[0]
		# b = weights[1]
		# c = weights[2]

		output = getOutput(weights, type, value)

		pattern = [ (value ** 2), value, 1]
		#pattern = [value, value, c]
	elif type == 3:
		#thresh = ax^3  + bx^2 + cx + d
		# a = weights[0]
		# b = weights[1]
		# c = weights[2]
		# d = weights[3]

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

def print_results (tot_error):
	for i in range(3):
		print ('Day {}: {}'.format((i+1),tot_error[i]))



def neuron1(time, power, alpha, minErr):
	weights = randomizeWeights(2)
	num_runs = 0
	error_cut_off = False
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
	print('Neuron 1 done')

	print('Training Error Neuron 1')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])
	if (error_cut_off):
		print("The error Threshold reached belowe the Total Error")
	print ('Number of runs before cutoff is :',num_runs)
	print_results(totErrs)
	print()

	return weights

def neuron2(time, power, alpha,minErr):
	weights = randomizeWeights(3)
	# oldWeights = weights.copy()
	error_cut_off = False
	num_runs = 0
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
	print('Neuron 2 done')

	print('Training Error Neuron 2')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])
	print ('Number of runs before cutoff is :',num_runs)
	if (error_cut_off):
		print("The error Threshold reached belowe the Total Error")
	print_results(totErrs)
	print()

	return weights

def neuron3(time, power, alpha, minErr):
	weights = randomizeWeights(4)
	error_cut_off = False
	# oldWeights = weights.copy()
	num_runs = 0
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



	# x = np.arange(0, 2)
	# y = [getOutput(weights, 3, i) for i in x]
	# plt.plot(x, y, label = 'predicted')

	# x = np.arange(5, 20)
	# y = [getOutput(oldWeights, 3, i) for i in x]
	# plt.plot(x, y, label = 'old')
	# print ('The total Error for Neuron 3 is :',totErr3)
	print('Neuron 3 done')

	print('Training Error Neuron 3')
	# print('Day 1:', totErrs[0])
	# print('Day 2:', totErrs[1])
	# print('Day 3:', totErrs[2])
	if (error_cut_off):
		print("The error Threshold reached belowe the Total Error")
	print ('Number of runs before cutoff is :',num_runs)
	print_results(totErrs)
	print()
	return weights

def drawLine(weights, type, name,ntype):
	if (ntype == 'standard'):
		x = np.arange(-1.75,1.75, 0.1)
	else:
		x = np.arange(0,1,0.1)
	y = [getOutput(weights, type, i) for i in x]
	plt.plot(x, y, label = name)

def normalizeInputs (data):
	data['time'] = (data['time'] - data['time'].min()) / (data['time'].max() - data['time'].min())
	#data['power'] = (data['time'] - data['time'].min()) / (data['time'].max() - data['time'].min())
	return data

def standarnormalization (df):
	df['time'] = (df['time'] - df['time'].mean()) / np.std(df['time'])
	#df['power'] = (df['power'] - df['power'].mean()) / np.std(df['power'])
	return df

def readcsv(filename):
	df = pd.read_csv(filename, header = None)
	# df.columns = ['time', 'power']
	return df

# def column_name(data):
# 	data.columns = ['time', 'power']
# 	return data


def normalizationMethod(data,type):
	if(type == 'standard'):
		data = standarnormalization(data)
		# dataDay1 = standarnormalization(dataDay1)
		# dataDay2 = standarnormalization(dataDay2)
		# dataDay3 = standarnormalization(dataDay3)
		# dataDay4 = standarnormalization(dataDay4)
		# return 'standard'

	elif(type == 'normal'):
		data = normalizeInputs(data)
		# dataDay1 = normalizeInputs(dataDay1)
		# dataDay2 = normalizeInputs(dataDay2)
		# dataDay3 = normalizeInputs(dataDay3)
		# dataDay4 = normalizeInputs(dataDay4)
		# return 'normal'
	else:
		data = normalizeInputs(data)




def prepare_data(filename, type):
	data = readcsv(filename)
	# data = column_name(data)
	# normalizationMethod(data,type)
	return data


#normalization type for this function
n_type = 'standard'
#n_type = 'normal'
#print(n_type)

# dataDay1 = prepare_data('train_data_1.txt',n_type)
# dataDay2 = prepare_data('train_data_2.txt',n_type)
# dataDay3 = prepare_data('train_data_3.txt',n_type)
# dataDay4 = prepare_data('test_data_4.txt',n_type)

dataDay1 = pd.read_csv('train_data_1.txt', header = None)
dataDay2 = pd.read_csv('train_data_2.txt', header = None)
dataDay3 = pd.read_csv('train_data_3.txt', header = None)
dataDay4 = pd.read_csv('test_data_4.txt', header = None)

dataDay1.columns = ['time', 'power']
dataDay2.columns = ['time', 'power']
dataDay3.columns = ['time', 'power']
dataDay4.columns = ['time', 'power']



# normalization type
#n_type = normalizationMethod(n_type)
normalizationMethod(dataDay1,n_type)
normalizationMethod(dataDay2,n_type)
normalizationMethod(dataDay3,n_type)
normalizationMethod(dataDay4,n_type)
#print(n_type)





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



#learning constant
# alpha1 = 0.001
# alpha2 = 0.001
# alpha3 = 0.001
alpha = 0.001

# For testing purposes
#print (time)
# print ("day one power is :\n")
# print (dataDay1['power'])
# print ("alpha is :", alpha3)

# neuron1weights = neuron1(time, power, alpha1)
# neuron2weights = neuron2(time, power, alpha2)
# neuron3weights = neuron3(time, power, alpha3)
neuron1weights = neuron1(time[:3], power[:3], alpha, 45)
neuron2weights = neuron2(time[:3], power[:3], alpha, 20)
neuron3weights = neuron3(time[:3], power[:3], alpha, 1)

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

# #display training data and predicted power line
# sns.lmplot(x = 'time', y = 'power', data = dataDay1, palette = 'plasma', fit_reg = False).fig.suptitle('Day 1')
# drawLine(neuron1weights, 1, 'neuron 1',n_type)
# drawLine(neuron2weights, 2, 'neuron 2',n_type)
# drawLine(neuron3weights, 3, 'neuron 3',n_type)
# plt.legend()

# sns.lmplot(x = 'time', y = 'power', data = dataDay2, palette = 'plasma', fit_reg = False).fig.suptitle('Day 2')
# drawLine(neuron1weights, 1, 'neuron 1',n_type)
# drawLine(neuron2weights, 2, 'neuron 2',n_type)
# drawLine(neuron3weights, 3, 'neuron 3',n_type)
# plt.legend()

# sns.lmplot(x = 'time', y = 'power', data = dataDay3, palette = 'plasma', fit_reg = False).fig.suptitle('Day 3')
# drawLine(neuron1weights, 1, 'neuron 1',n_type)
# drawLine(neuron2weights, 2, 'neuron 2',n_type)
# drawLine(neuron3weights, 3, 'neuron 3',n_type)
# plt.legend()

# # #display predicted power, predicted power line, and actual power
# sns.lmplot(x = 'time', y = 'power', data = dataDay4, palette = 'plasma', fit_reg = False).fig.suptitle('Test Data')
# drawLine(neuron1weights, 1, 'neuron 1',n_type)
# drawLine(neuron2weights, 2, 'neuron 2',n_type)
# drawLine(neuron3weights, 3, 'neuron 3',n_type)
# plt.legend()




testOutput1 = [getOutput(neuron1weights, 1, hour) for hour in time[3]]
testOutput2 = [getOutput(neuron2weights, 2, hour) for hour in time[3]]
testOutput3 = [getOutput(neuron3weights, 3, hour) for hour in time[3]]
print('Test Error')
print('Neuron 1:', getTotalError(power[3], testOutput1))
print('Neuron 2:', getTotalError(power[3], testOutput2))
print('Neuron 3:', getTotalError(power[3], testOutput3))

plt.legend()
plt.show()
