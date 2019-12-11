# problem 2_py
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
import math
from statistics import mean, stdev
from cycler import cycler

# print("hello World")



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


def list_create(data,name1,name2):

     feature_list = data[name1].tolist()
     class_list = data[name2].tolist()

     list_0, list_1, list_2 = ([] for i in range(3))

     for i, value in enumerate(class_list):
        if (value == 0):
            list_0.append(feature_list[i])

        elif(value == 1):
            list_1.append(feature_list[i])
        elif(value == 2):
            list_2.append(feature_list[i])

     return list_0, list_1, list_2


 def prob_distribution ():
     print("test")

def gausian_distribution (std_input, mean_input, prob_class, input_value):
    prob_input = (1 / (std_input * math.sqrt(2 * math.pi))) * math.exp((-(input_value - mean_input)**2)/(2 * std_input ** 2))
    prob_case_1 = math.log(prob_input) + math.log(prob_class)
    return prob_case_1


# def predict_class(input, input2):
#     if(input1 > input2):
#         class = ""

def class_sort (list):
    print("example")

def loadData(filename):
    df = pd.read_csv(filename, header = None)
    # df
    df.columns = ["feature","class"]
    return df

    # print ("hello world!")
    # print ("Guru99")

    # neuron1(value)
    today = date.today()
    d3 = today.strftime("%m/%d/%y")
    chady = Student("Chady Aboulhosn", "CMSC 409","Problem 2" , d3)
    chady.show()
    # file_name = "Ex1_data.tx


    train_file = "Ex2_train.csv"
    test_file = "Ex2_test.csv"


    test_female_list = [56.0307 ,64.2989
                        ,60.3343
                        ,51.8031
                        ,47.8763
                        ,58.5809
                        ,65.7290
                        ,60.9058
                        ,60.2713
                        ,63.4388]

    test_male_list = [73.0330,
                        87.1268,
                        75.5307,
                        80.1888,
                        78.1822,
                        80.7478,
                        70.2774,
                        87.6195,
                        82.7291,
                        90.0496,
                        87.0834,
                        80.0574,
                        75.3048,
                        71.3055,
                        80.0849]

    # train_df = loadData(train_file)
    # test_df = loadData(test_file)

    # list_class_0, list_class_1, list_class_2 = list_create(train_df,"feature","class")

    # print(list_class_0)

    # print(test_female_list)
    mean_female = mean(test_female_list)
    mean_male = mean(test_male_list)
    std_female = stdev(test_female_list)
    std_male = stdev(test_male_list)
    len_list1 = len(test_female_list)
    len_list2 = len(test_male_list)
    prob_female =  len_list1/ (len_list1 + len_list2)
    prob_male = len_list2 / ((len_list1 + len_list2))

    std_input = std_female
    mean_input = mean_female

    input_value = 75
    strim = math.exp((-(input_value - mean_input)**2)/(2 * std_input ** 2))

    print(strim)


    prob_case_1 = math.log(prob_input) + math.log(prob_female)


    print(prob_case_1)

    print(prob_input)






    # print(mean_female)
    # print(mean_male)
    # print(std_female)
    # print(std_male)



    # if()



    # print(train_df)

    # for i in len(train_list)
    # print train_df.describe(include = 'all')


if __name__== "__main__":
  main()
