import pandas as pd
import re
# from collections import counter
test_File = list('abcdefghijklmnopqrstuvwxyz')

# print(test_File)

# for i in test_File:
# 	print(i)


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

	# create a list for a column of
	# for num in range(len(list_sentence_count)):


	#add all rows to new df
	new_df = df.append(list_series_count,ignore_index=True)
	new_df.insert(0,'Keyword Set', keyword_set)
	#insert a first column into the def
	# new_df.i
	# create cv based on create_tdm outputs cv to it
	new_df.to_csv("output.csv",index =False, header = True)
	# prints df for testing
	# print(new_df)

def fileToList(filename):
	textList = []
	file = open(filename, encoding="utf8")
	for line in file:
		textList.append(line.strip())

	return textList

test_insert = list('01234567890123456789012345')
test_total = []
# test_total.append(test_insert)

for i in range(50):
	test_total.append(test_insert)

create_tdm(test_File,test_total)
# create_tdm()



# sentences = fileToList('sentences.txt')
sentences = fileToList('test.txt')
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

# [print (i) for i in sentences]
# print(sentences)



# list_counts = tuple()
# list_counts = []
# count_list = list('123456789')
# list_counts.append(count_list)
# count_list.clear()
# count_list = list('987654321')
# list_counts.append(count_list)
# print(list_counts)


# def count_list ()


# for sentence in sentences:
# 	for word in sentence:
# 		count_list.append(1)
# 		# if word not in list2:
# 		# 	list2.append(word)
# 		# else:
# 		# 	list3.append(word)
# 	list_counts.append(count_list)
# 	print(list_counts)
# 	print(len(count_list))
# 	print(count_list)
# 	count_list.clear()
# # print (list2)
# # print(list3)
# # print(count_list)
# print('length of first sentence is ', len(sentences[0]))
# # print(len(sentences[0]))
# # print(sentences[0])
# # print(len(count_list))
# print('length of first count in list is ',len(list_counts[0]))
# print(list_counts[0])


list3 = []
list2 = []
count_list = []
# print(sentences)
def word_count(sentence, count_list,uniq_list):
	count_insert = []

	for index, word in sentence:
		if word not in uniq_list:
			# uniq_list.append(word)
			count_insert.append(1)
		else:
			# list3.append(word)
			count_insert.append(2)
	count_list.append(count_insert)
	# print(uniq_list)

# def unique_word_Count()

# for i in range(len(sentences)):
# 	word_count(sentences[i],count_list,list2)

# print(count_list)
# print(list2)

for i in sentences:
	print (i)
