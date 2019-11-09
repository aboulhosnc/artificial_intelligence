import pandas as pd

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
	print(new_df)



# for num, type in enumerate(test_File):
# 	# df.insert(num,type,  num)
# 	df.insert(num,type,'insert')
# 	# print (i)

test_insert = list('01234567890123456789012345')
test_total = []
# test_total.append(test_insert)

for i in range(50):
	test_total.append(test_insert)

create_tdm(test_File,test_total)
# create_tdm()

# print(len(test_insert))
# print(len(test_total))
# print(len(test_File))
# test_total.append(test_File)
# Pass a series in append() to append a row in dataframe
# modDfObj = dfObj.append(pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns ), ignore_index=True)

# test_series = []
# for i in range(len(test_total)):
# 	# new_df = df.append(pd.Series(test_total[i],index = df.columns),ignore_index=True)
# 	test_series.append(pd.Series(test_total[i],index = df.columns))

# new_df = df.append(test_total, ignore_index = True)
# new_df = df.append(pd.Series(test_total[0],index = df.columns),ignore_index=True)
# new_df = df.append(test_series,ignore_index=True)
# df.append(test_series,ignore_index=True)
# df.append()
# print(test_total)
# print(new_df)
# print(df)
# new_df.to_csv("output_" +(value) + ".csv", index=False, header = True)
# new_df.to_csv("output.csv",index =False, header = True)
