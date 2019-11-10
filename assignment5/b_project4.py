


def fileToList(filename):
	textList = []
	file = open(filename, encoding="utf8")
	for line in file:
		textList.append(line.strip())

	return textList


sentences = fileToList('test.txt')
stopwords = fileToList('stop_words.txt')

count = 0
list2 = []
list3 = []
for i in sentences:
    for word in i.split():
        if (word not in list2):
            list2.append(word)
        else:
            list3.append(word)
        count = count +1
        # print(word)
    # print(i)

print(count)
print(list2)
len_2 = len(list2)
len_3 = len(list3)
print(len_2)
print(len_3 )
print(len_2 + len_3)
# print(list3)
