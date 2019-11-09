import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from cycler import cycler


def readCSV(filename):
    df = pd.read_csv(filename, header = None)
    df.columns = ['time', 'kilowatt']
    return df

def printcsv(file):
    print(file)

def df_list_time(data):
    return  data['time'].tolist()

def df_list_kw(data):
    return data['kilowatt'].tolist()

def drawLine (data):
    plt.plot(df_list_time(data),df_list_kw(data), label = 'train day')
    xlabel = plt.xlabel("Time")
    ylabel = plt.ylabel("kilowatt")
    
    plt.show()
    plt.legend(handles=[xlabel, ylabel])
    #print("test")

def randomize_weights (size):
    array = [rand.uniform(-0.5,0.5) for x in range(size)]
    return array

def neuron_1 (time, kilowatt):
    weights = randomizeWeights(2)
    return 1

### main program


df1 = readCSV('train_data_1.txt')
df2 = readCSV('train_data_2.txt')
df3 = readCSV('train_data_3.txt')
df4 = readCSV('test_data_4.txt')

# for i in df_list_time(df1):
#     print (i)
#
# for i in df_list_kw(df1):
#     print (i)
drawLine(df1)



#printcsv(df1)

#train_df_set = [df1,df2,df3]
# for i in train_df_set:
#     printcsv(i)
