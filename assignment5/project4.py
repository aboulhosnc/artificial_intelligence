import re

def fileToList(filename):
	textList = []
	file = open(filename, encoding="utf8")
	for line in file:
		textList.append(line.strip())

	return textList

sentences = fileToList('sentences.txt')
stopwords = fileToList('stop_words.txt')

#converts to lower case
sentences = [s.lower() for s in sentences]

#removes punctuation
sentences = [re.sub('[\(\)\.\,\“\”\'\"\\\/\[\]\-–]', ' ', s) for s in sentences]

#splits string into list with whitespace as the delimiter
sentences = [s.split(' ') for s in sentences]

#removes empty strings in the list
sentences = [[i for i in s if re.search('^$', i) == None] for s in sentences]

#removes any digits in the list
sentences = [[i for i in s if re.search('\d+.*', i) == None] for s in sentences]

#removes any stop words
sentences = [[i for i in s if i not in stopwords] for s in sentences]

for s in sentences:
	print(s)

# s = sentences[2]
# print(s)

# s = s.lower()
# print(s)

# s = re.sub('[\(\)\.\,\“\”\'\"\\\/\[\]\-–]', ' ', s)
# print(s)

# s = s.split(' ')
# print(s)

# s = [i for i in s if re.search('^$', i) == None]
# print(s)

# s = [i for i in s if re.search('\d+.*', i) == None]
# print(s)

# s = [i for i in s if i not in stopwords]
# print(s)
