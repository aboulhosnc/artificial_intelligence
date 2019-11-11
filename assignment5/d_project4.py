import re
import pandas as pd
from Porter_Stemmer_Python import PorterStemmer
p = PorterStemmer()


def fileToList(filename):
	textList = []
	file = open(filename, encoding = "utf8")
	for line in file:
		textList.append(line.strip())
		
	return textList
	
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
dummy_list = []
dummy_list2 = []
for s in sentences:
    for w in s:
        if w not in dummy_list:
            dummy_list.append(w)
        elif w not in dummy_list2:
            dummy_list2.append(w)
        
        elif w not in df.columns.values.tolist():
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

print(wordFreq)

df.to_csv("output1.csv",index =False, header = True)
print(df)