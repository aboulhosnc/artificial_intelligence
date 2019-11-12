import math
import pandas as pd
import re
from Porter_Stemmer_Python import PorterStemmer
from collections import Counter

def fileToList(filename):
	textList = []
	file = open(filename, encoding = "utf8")
	for line in file:
		textList.append(line.strip())

	return textList

#returns euclidean distance between two points
def distance(weights, pattern):
	sumList = [(weights[i] - pattern[i]) ** 2 for i in range(len(weights))]
	s = sum(sumList)
	return math.sqrt(s)

#returns index of the closest cluster
def findClosestCluster(clusterList, pattern, maxDist):
	clusterDistances = [distance(c, pattern) for c in clusterList]
	if min(clusterDistances) < maxDist:
		return clusterDistances.index(min(clusterDistances))
	else:
		return False

def adjustCluster(weights, pattern, m, alpha):
	newWeights = [(m * weights[i] +  alpha * pattern[i]) / (m + 1) for i in range(len(weights))]
	return newWeights

def formingClustersAsNeeded(vectorMatrix):
	alpha = 1
	maxDist = 3

	clusterWeights = list()

	#which cluster each pattern belongs to
	#index is pattern number, value is which cluster
	clusterPattern = list()

	for index, row in vectorMatrix.iterrows():
		if len(clusterWeights) == 0:
			clusterWeights.append(row.tolist())
			clusterPattern.append(0)
		else:
			closestCluster = findClosestCluster(clusterWeights, row.tolist(), maxDist)
			if closestCluster == False:
				clusterWeights.append(row.tolist())
				clusterPattern.append(len(set(clusterPattern)))
			else:
				adjustCluster(clusterWeights[closestCluster], row.tolist(), clusterPattern.count(closestCluster), alpha)
				clusterPattern.append(closestCluster)

	#gets total number of patterns in each cluster
	clusterCount = sorted(Counter(clusterPattern).items())
	clusterCount = [c[1] for c in clusterCount]

	clusterMatrix = pd.DataFrame(clusterWeights)
	clusterMatrix.columns = vectorMatrix.columns
	clusterMatrix.insert(0, 'pattern amount', clusterCount)

	print(clusterMatrix)

	return clusterMatrix

#------#
# Main #
#------#
p = PorterStemmer()

sentences = fileToList('sentences.txt')
stopwords = fileToList('stop_words.txt')

#converts to lower case
sentences = [s.lower() for s in sentences]

#removes punctuation
sentences = [re.sub('[\(\)\.\,\“\”\'\"\\\/\[\]–]', ' ', s) for s in sentences]

#tokenizes string with whitespace as the delimiter
sentences = [s.split(' ') for s in sentences]

#removes empty strings in the list
sentences = [[w for w in s if re.search('^$', w) == None] for s in sentences]

#removes any digits in the list
sentences = [[w for w in s if re.search('\d+.*', w) == None] for s in sentences]

#removes any stop words
sentences = [[w for w in s if w not in stopwords] for s in sentences]

#stems the words
sentences = [[p.stem(w, 0, (len(w) - 1)) for w in s] for s in sentences]

df = pd.DataFrame()

for s in sentences:
	for w in s:
		if w not in df.columns.values.tolist():
			df.insert(0, w, [], allow_duplicates = False)

for i, s in enumerate(sentences):
	df.loc[i] = [0 for i in range(len(df.columns))]

	for w in s:
		if w in df.columns.values.tolist():
			df.at[i, w] += 1

wordFreq = pd.DataFrame(columns = df.columns)
wordFreq.loc[0] = [0 for i in range(len(wordFreq.columns))]

for w in wordFreq.columns.values.tolist():
	total = df[w].sum()
	wordFreq.at[0, w] = total

wordFreq = wordFreq.sort_values(by = 0, axis = 1, ascending = False)

mostFreq = list()

count = 0
for w in wordFreq.columns.values.tolist():
	mostFreq.append(w)
	count += 1
	if count >= 10:
		break

termDocMatrix = pd.DataFrame()

for w in wordFreq.columns.values.tolist():
	if w in mostFreq:
		termDocMatrix[w] = df[w]
		termDocMatrix[w] = df[w].values

clusterMatrix = formingClustersAsNeeded(termDocMatrix)

# print(termDocMatrix)
