from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from statistics import mean
from cycler import cycler


class Student():

    def __init__ (self,user_name,class_number,assignment, date, class_title = "",class_name = ""):
        self.user_name = user_name
        self.class_number = class_number
        self.assignment = assignment
        self.date = date
        self.class_title = class_title
        self.class_name = class_name

    def show(self):
        print("..........................................")
        print("Name   : {}".format(self.user_name))
        print("Class  : {}".format(self.class_number))
        print("Assign : {}".format(self.assignment))
        print("Date   : {}".format(self.date))
        print("..........................................")


        # print("Today's date:", today)

def neuron1(value):
    print(value)

def neuron2(value2):
    print(value2)

def neruon3(value3):
    print(value3)

def neuron4(value4):
    print(value4)

def loadData(filename):
    df = pd.read_csv(filename)
    df.columns = ["x1","x2"]
    return df

def normalization(data):
    data['x1'] = (data['x1'] - data['x1'].min())/ (data['x1'].max() - data['x1'].min())
    data['x2'] = (data['x2'] - data['x2'].min())/ (data['x2'].max() - data['x2'].min())
    return data

def plotPoints(listx,listy, df):
    sns.lmplot(x = 'x1', y = 'x2', data = df, palette = 'plasma', fit_reg = False).fig.suptitle("xy")
    # plt.plot(listx, listy)
    plt.show()

def randomize_weights ():
    num = 0.5
    weights = [rand.uniform(0,1) for x in range(3)]
    return weights
	# df.columns = ['height', 'weight', 'sex']

# .....................
# ........Main.........
# .....................

def main():
    # print ("hello world!")
    # print ("Guru99")
    value = "test"
    value2 = "neuron3"
    value3 = "neuron2"
    neuron1(value)
    today = date.today()
    d3 = today.strftime("%m/%d/%y")
    chady = Student("Chady Aboulhosn", "CMSC 409","Final Exam Prep" , d3)
    file_name = "Ex1_data.txt"

    chady.show()

    df = loadData(file_name)

    # print(df)
    normalization(df)
    # print(df)


    # print(df)
    # convert to lists
    x_list = list()
    y_list = list()
    x_list = df['x1'].tolist()
    y_list = df['x2'].tolist()

    plotPoints(x_list,y_list,df)
    weights = randomize_weights()
    print(weights[0])
    print(weights[1])
    # print(x_list)
    # print(y_list)






if __name__== "__main__":
  main()
