import re
# from Porter_Stemmer import PorterStemmer
from Porter_Stemmer import PorterStemmer
p = PorterStemmer()

def fileToList(filename):
	textList = []
	file = open(filename, encoding="utf8")
	for line in file:
		textList.append(line.strip())

	return textList

def create_tdm (unique_word_vector, list_sentence_count):
	""" Input list of all the unique words, and the list with every sentency frequency  """
	df = pd.DataFrame()
	list_series_count = []
	keyword_set = []

	# create columns for each unique word
	for col_num, word in enumerate(unique_word_vector):
		# df.insert(num,type,  num)
		df.insert(col_num,word,'insert')

	# create list of all the series of rows to add to dataframe
	#create a list for a column for keywords
	for  num in range (len(list_sentence_count)):
		list_series_count.append(pd.Series(list_sentence_count[num],index = df.columns))
		keyword_set.append('Sentence ' + str(num+1))

	#add all rows to new df
	new_df = df.append(list_series_count,ignore_index=True)
	new_df.insert(0,'Keyword Set', keyword_set)
	# create cv based on create_tdm outputs cv to it
	new_df.to_csv("output.csv",index =False, header = True)
	# prints df for testing
	# print(new_df)


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

# sentences = p.stem(sentences)

# test_stem = []

# print(sentences)
# for s in sentences:
# 	# s = p.stem(s,0,(len(s)-1))
# 	# test_stem.append(s)
# 	# print(s)
# 	# print(p.stem(s,0,len(s)-1))
# 	print(s)

print(sentences)

# print(test_stem)

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
