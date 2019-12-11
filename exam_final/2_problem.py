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


def class_list_create(data,name1,name2):


    feature_list = data[name1].tolist()
    #  max_val = max(feature_list)
    #  min_val = min(feature_list)
    class_list = data[name2].tolist()
    list_length = len(feature_list)
    print("length of list is : {}".format(list_length))

    list_0, list_1, list_2 = ([] for i in range(3))

    for i, value in enumerate(class_list):
        if (value == 0):
            list_0.append(feature_list[i])
            

        elif(value == 1):
            list_1.append(feature_list[i])
            
        elif(value == 2):
            list_2.append(feature_list[i])

    # print("length of list is : {}".format(len(list_0)))
    # print("length of list is : {}".format(len(list_1)))
    # print("length of list is : {}".format(len(list_2)))
    return list_0, list_1, list_2, list_length
            

    #  return normalize(list_0), normalize(list_1), normalize(list_2), list_length, max_val, min_val
    #  return list_0, list_1, list_2, list_length, max_val, min_val
        


def prob_distribution (list1):
    # prob_case_1 = math.log(prob_input) + math.log(prob_class)
    print("test")
    
# def normalize(list1):
#     max_val = max(list1)
#     min_val = min(list1)
#     norm_list = [((i - min_val)/(max_val - min_val)) for i in list1 ]
#     return norm_list

def normalize_data (data, name):
    data[name] = (data[name] - data[name].min()) / (data[name].max() - data[name].min())
    return data


def gausian_distribution (std_input, mean_input, input_value):
    prob_input = (1 / (std_input * math.sqrt(2 * math.pi))) * math.exp((-(input_value - mean_input)**2)/(2 * std_input ** 2))
    return prob_input

def bays_prob(value, prob_1, prob_2, prob_3):
    class1 = math.log(value) + math.log(prob_1)
    class2 = math.log(value) + math.log(prob_2)
    class3 = math.log(value) + math.log(prob_3)

    result = list()
    result = predict_class(class1,class2,class3)
    return result


def predict_class(input1, input2, input3):
    # input1, input2, input3 = bays_prob(value)
    class_test = list()
    if(input1 > input2 and input1 > input3):
        class_test.append(0)
    elif(input2 > input1 and input2 > input3):
        class_test.append(1)
    elif(input3 > input1 and input3 > input2):
        class_test.append(2)
    return class_test

def prepare_data (list1, class_length):
    mean_input = mean(list1)
    std_input = stdev(list1)
    prob_class = (len(list1)/ class_length)
    return mean_input, std_input, prob_class

def plot_line (list1,list2,list3, x_list):
    # title = 
    plt.plot(x_list,list1, 'r', label = "class 0")
    plt.plot(x_list,list2, 'b', label = "class 1")
    plt.plot(x_list,list3, 'g', label = "class 2")
    plt.xlim(0,1)
    # plt.ylim(-0.04,0.1)
    plt.legend()
    plt.show()


def loadData(filename):
    df = pd.read_csv(filename, header = None)
    # df
    df.columns = ["feature","class"]
    return df


def main():

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

    train_df = loadData(train_file)
    normalize_data(train_df,"feature")
    # print(train_df)
    test_df = loadData(test_file)

    test_set = test_df['feature'].tolist()

    list_class_0, list_class_1, list_class_2, total_length = ([] for i in range(4))
    list_class_0, list_class_1, list_class_2, total_length = class_list_create(train_df,"feature","class")

    # normalize
    # print(list_class_0)
    
    # print(list_class_1)
    # print(list_class_2)
    # print(max_val)
    # print(min_val)

    # print(list_class_0)

    # print(test_female_list)

    mean_1, std_1, prob_1 = prepare_data(list_class_0, total_length)
    mean_2, std_2, prob_2 = prepare_data(list_class_1, total_length)
    mean_3, std_3, prob_3 = prepare_data(list_class_2, total_length)

    # mean_female, std_female, prob_female = prepare_data(test_female_list, 25)

    # print(prob_1)
    # print(prob_2)
    # print(prob_3)

    mean_female = mean(test_female_list)
    mean_male = mean(test_male_list)
    std_female = stdev(test_female_list)
    std_male = stdev(test_male_list)
    len_list1 = len(test_female_list)
    len_list2 = len(test_male_list)
    prob_female1 =  len_list1/ (len_list1 + len_list2)

    print(prob_female1)

    prob_male = len_list2 / ((len_list1 + len_list2))

    std_input = std_female
    mean_input = mean_female

    input_value = 75
    # strim = math.exp((-(input_value - mean_input)**2)/(2 * std_input ** 2))

    class1, class2, class3 = ([] for i in range(3))
    # male_distribution = list()

    test_value = np.arange(0,1,0.01)
    print(len(test_value))

    # print(test_value)

    for input_value in (test_value):
        class1.append(gausian_distribution (std_1, mean_1,  input_value))
        class2.append(gausian_distribution (std_2, mean_2,  input_value))
        class3.append(gausian_distribution (std_3, mean_3,  input_value))
    
    test_set = test_df['feature'].tolist()
    normalize_data(test_df,"feature")
    test_result = test_df['class'].tolist()

    test_model_result = list()
    for i in (test_set):
        test_model_result.append(bays_prob(i,prob_1,prob_2,prob_3))
    
    print(test_model_result)





    plot_line(class1,class2,class3,test_value)


    


    # prob_case_1 = math.log(prob_input) + math.log(prob_female)


    # print(prob_case_1)

    # print(prob_input)






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
