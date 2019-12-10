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

def neuron1(x_value, y_value, alpha):
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

def plotPoints(df,old_weight1,old_weight2,old_weight3,weight1,weight2,weight3,title,x_val,y_val):
    title_str = "Iteration {}".format(title)
    sns.lmplot(x = 'x1', y = 'x2', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(title_str)
    plt.plot(old_weight1[0],old_weight1[1], 'r+', label = 'weight1')
    plt.plot(old_weight2[0],old_weight2[1], 'y+', label = "weight2")
    plt.plot(old_weight3[0],old_weight3[1], 'g+', label = 'weight3')
    plt.plot(weight1[0],weight1[1], 'ro', label = 'weight1')
    plt.plot(weight2[0],weight2[1], 'ys', label = "weight2")
    plt.plot(weight3[0],weight3[1], 'g^', label = 'weight3')
    plt.plot(weight3[0],weight3[1], 'g^', label = 'x')
    plt.plot(x_val,y_val, 'r*', label = 'point')
    plt.legend()
    plt.show()

    # plt.plot(listx, listy)
    # plt.show()

def train(x_val, y_val):
    print("test")

def randomize_weights (value):
    num = 0.5
    weights = [round(rand.uniform(0,1),4) for x in range(value)]
    return weights
	# df.columns = ['height', 'weight', 'sex']

def normalize_weights(weight1, weight2):
    normalized_weight1 = (weight1/math.sqrt((weight1**2)+(weight2**2)))
    normalized_weight2 = (weight2/math.sqrt((weight1**2)+(weight2**2)))

    normalized_weight1 = round(normalized_weight1, 4)
    normalized_weight2 = round(normalized_weight2, 4)


    return normalized_weight1, normalized_weight2

def  new_weight_avg (weight1, weight2, alpha, x_val, y_val):
    # print(weight1)
    # print(weight2)
    avg_weight1  = weight1  + (alpha * x_val)
    avg_weight2  =  weight2  + (alpha * y_val)
    avg_weight1 = round(avg_weight1, 4)
    avg_weight2 = round(avg_weight2, 4)
    return avg_weight1, avg_weight2

def train_neuron (x_val, y_val, weight):
    # print("train Neuron")
    net = x_val *weight[0] + y_val * weight[1]

    return net

def update_neuron(weight,alpha,x_val,y_val):

    weight = new_weight_avg(weight[0],weight[1],alpha,x_val,y_val)
    # print("new weights are : {}".format(weight))
    weight = normalize_weights(weight[0], weight[1])
    # print("new weights normalized  are : {}".format(weight))
    return weight



# .....................
# ........Main.........
# .....................

def main():
    # print ("hello world!")
    # print ("Guru99")
    value = "test"
    value2 = "neuron3"
    value3 = "neuron2"
    # neuron1(value)
    today = date.today()
    d3 = today.strftime("%m/%d/%y")
    chady = Student("Chady Aboulhosn", "CMSC 409","Final Exam Prep" , d3)
    # file_name = "Ex1_data.txt"
    file_name = "norm_weights.txt"
    alpha = .3
    # number_neruons = 2
    #
    # file_name = "final_konan_test.txt"

    chady.show()

    df = loadData(file_name)

    # print(df)
    # plotPoints(df)
    normalization(df)
    # print(df)


    # print(df)
    # convert to lists
    x_list = list()
    y_list = list()
    x_list = df['x1'].tolist()
    y_list = df['x2'].tolist()
    list_length = len(x_list)
    itterations = list_length * 10
    print (list_length)
    print(int(list_length /2 ))

    # plotPoints(df)
    weights = randomize_weights(6)
    print(weights)

    #original weights
    ow1 = [weights[0],weights[1]]
    ow2 = [weights[2],weights[3]]
    ow3 = [weights[4],weights[5]]

    weight_neuron1 = [weights[0],weights[1]]
    weight_neuron2 = [weights[2],weights[3]]
    weight_neuron3 = [weights[4],weights[5]]


    #random weights
    # weights = [x_list[0],y_list[0],x_list[int(list_length /2) ],y_list[list_length /2],x_list[list_length -1],y_list[list_length -1]]
    # ow1 = [weights[0],weights[1]]
    # ow2 = [weights[2],weights[3]]
    # ow3 = [weights[4],weights[5]]
    #
    # weight_neuron1 = [weights[0],weights[1]]
    # weight_neuron2 = [weights[2],weights[3]]
    # weight_neuron3 = [weights[4],weights[5]]

    # ow1 = [.9459, .3243]
    # ow2 = [.6690, .7433]
    # ow3 = [.3714, .9285]
    # #
    # weight_neuron1 = [.9459, .3243]
    # weight_neuron2 = [.6690, .7433]
    # weight_neuron3 = [.3714, .9285]



    # print(list_length)
    value = 0
    plotPoints(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,0,x_list[0],y_list[0])
    # print(weight_neuron1)
    # print(weight_neuron2)
    # print(weight_neuron3)



    # for i in range(4):

    # value = 0
    print("Starting Run")
    print("Original Weights")
    print(weight_neuron1)
    print(weight_neuron2)
    print(weight_neuron3)
    for index in range(itterations):
        if(value > list_length -1):
            # print(value)
            # print(index)
            value = 0
        print("..................................")
        print("Pattern is : {}".format(value))
        print("Itteration is : {}".format(index))

        run = index + 1
        win_neuron = 0


        print("Applying {} Itteration".format(index))
        net1 = train_neuron(x_list[value],y_list[value],weight_neuron1)
        net2 = train_neuron(x_list[value],y_list[value],weight_neuron2)
        net3 = train_neuron(x_list[value],y_list[value],weight_neuron3)

        # print("net1 is : {}".format(net1))
        # print("net2 is : {}".format(net2))
        # print("net3 is : {}".format(net3))

        if((net1 > net2) and (net1 > net3)):
            # print("Neruon 1 was the winner")
            # print("net is : {}".format(net1))

            # update_neuron(weight,alpha,x_list,y_list):
            win_neuron = 1
            weight_neuron1 = update_neuron(weight_neuron1,alpha,x_list[value],y_list[value])



        elif((net2 > net1) and (net2 > net3)):
            # print("2 was the winner")
            # print("net is : {}".format(net2))
            win_neuron = 2
            weight_neuron2 = update_neuron(weight_neuron2,alpha,x_list[value],y_list[value])

        else:
            # print("3 was the winner")
            # print("net is : {} ".format(net3))
            win_neuron = 3
            weight_neuron3 = update_neuron(weight_neuron3,alpha,x_list[value],y_list[value])

        # print(net1,net2,net3)
        # if(index == 0 or index == 1 or index == 2 or index == 3 or index == 29):
        if (run % 10 == 0):
            print("Weight {} was updated".format(win_neuron))
            print("x_val {}  y_val {}".format(x_list[value],y_list[value]))
            print("Updated Weights are")
            # print("...................")
            print(weight_neuron1)
            print(weight_neuron2)
            print(weight_neuron3)
            print("...................")
            plotPoints(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,run,x_list[value],y_list[value])


        value += 1



    plotPoints(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,run,x_list[0],y_list[0])
    # test_num = 0.4567521
    # print(test_num)
    # test_num = round(test_num,4)
    # print("rounded number is : {}".format(test_num))

    # new_weight1, new_weight2 = normalize_weights(test_weight1, test_weight2)
    # print(new_weight1, new_weight2)


    # print(x_list)
    # print(y_list)






if __name__== "__main__":
  main()
