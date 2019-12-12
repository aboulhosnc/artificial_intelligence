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

def plotPoints2(df,old_weight1,old_weight2,weight1,weight2,title,x_val,y_val):
    title_str = "Iteration {}".format(title)
    sns.lmplot(x = 'x1', y = 'x2', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(title_str)
    plt.plot(old_weight1[0],old_weight1[1], 'r+', label = 'weight1')
    plt.plot(old_weight2[0],old_weight2[1], 'y+', label = "weight2")
    # plt.plot(old_weight3[0],old_weight3[1], 'g+', label = 'weight3')
    plt.plot(weight1[0],weight1[1], 'ro', label = 'weight1')
    plt.plot(weight2[0],weight2[1], 'ys', label = "weight2")
    # plt.plot(weight3[0],weight3[1], 'g^', label = 'weight3')
    # plt.plot(weight3[0],weight3[1], 'g^', label = 'x')
    plt.plot(x_val,y_val, 'r*', label = 'point')
    plt.legend()
    plt.show()

def plotPoints3(df,old_weight1,old_weight2,old_weight3,weight1,weight2,weight3,title,x_val,y_val):
    title_str = "Iteration {}".format(title)
    sns.lmplot(x = 'x1', y = 'x2', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(title_str)
    plt.plot(old_weight1[0],old_weight1[1], 'r+', label = 'weight1')
    plt.plot(old_weight2[0],old_weight2[1], 'y+', label = "weight2")
    plt.plot(old_weight3[0],old_weight3[1], 'g+', label = 'weight3')
    plt.plot(weight1[0],weight1[1], 'ro', label = 'weight1')
    plt.plot(weight2[0],weight2[1], 'ys', label = "weight2")
    plt.plot(weight3[0],weight3[1], 'g^', label = 'weight3')
    # plt.plot(weight3[0],weight3[1], 'g^', label = 'x')
    plt.plot(x_val,y_val, 'r*', label = 'point')
    plt.legend()
    plt.show()

def plotPoints7(df,old_weight1,old_weight2,old_weight3,ow4,ow5,ow6,ow7,weight1,weight2,weight3,w4,w5,w6,w7,title,x_val,y_val):
    title_str = "Iteration {}".format(title)
    sns.lmplot(x = 'x1', y = 'x2', data = df, palette = 'plasma', fit_reg = False).fig.suptitle(title_str)
    plt.plot(old_weight1[0],old_weight1[1], 'r+', label = 'weight1')
    plt.plot(old_weight2[0],old_weight2[1], 'y+', label = "weight2")
    plt.plot(old_weight3[0],old_weight3[1], 'g+', label = 'weight3')
    plt.plot(ow4[0],ow4[1], 'r-', label = 'weight4')
    plt.plot(ow5[0],ow5[1], 'y-', label = 'weight5')
    plt.plot(ow6[0],ow6[1], 'p-', label = 'weight6')
    plt.plot(ow7[0],ow7[1], 'b-', label = 'weight7')


    plt.plot(weight1[0],weight1[1], 'ro', label = 'weight1')
    plt.plot(weight2[0],weight2[1], 'ys', label = "weight2")
    plt.plot(weight3[0],weight3[1], 'g^', label = 'weight3')
    plt.plot(w4[0],w4[1], 'r--', label = 'weight4')
    plt.plot(w5[0],w5[1], 'y--', label = 'weight5')
    plt.plot(w6[0],w6[1], 'p--', label = 'weight6')
    plt.plot(w7[0],w7[1], 'b--', label = 'weight7')
    # plt.plot(weight3[0],weight3[1], 'g^', label = 'x')
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
    net = x_val * weight[0] + y_val * weight[1]

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
    
    today = date.today()
    d3 = today.strftime("%m/%d/%y")
    chady = Student("Chady Aboulhosn", "CMSC 409","Final Exam Prep" , d3)
    file_name = "Ex1_data.csv"
    # file_name = "norm_weights.txt"
    alpha = .5
    number_neruons = 7
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
    # itterations = list_length * 10
    itterations = 500
    test_Num =  len(x_list)
    if(test_Num % 2 == 0):
        test_length  = test_Num / 2
        print("exactly half")
    else:
        test_length = (test_Num + 1) / 2 
        print("not exactly half")
    half_length = (test_Num / 2)
    print(test_Num)
    print(half_length)
    print(test_length)
    print(test_length)
    print (list_length)


    # if()

    # 3
    #weights in exactly half of data
    # weights = [x_list[0],y_list[0],x_list[75],y_list[75],x_list[list_length - 1],y_list[list_length - 1]]

    #random weights
    # weights = randomize_weights(6)

    #weights at half data max and value
    # weights = [min(x_list), min(y_list),max(x_list)/2, max(y_list)/2, max(x_list), max(y_list)]


    #manual weight based on clusters
    # weights = [0.25,0.15, 0.1,0.9, 0.8, 0.9]
    # ow1 = [weights[0],weights[1]]
    # ow2 = [weights[2],weights[3]]
    # ow3 = [weights[4],weights[5]]
    
    # weight_neuron1 = [weights[0],weights[1]]
    # weight_neuron2 = [weights[2],weights[3]]
    # weight_neuron3 = [weights[4],weights[5]]

    #2 Neurons
    #random weights
    # weights = randomize_weights(4)
    #weights in exactly half of data
    # weights = [x_list[0],y_list[0],x_list[list_length - 1],y_list[list_length - 1]]
    #manual weights
    # weights = [0.25,0.15, 0.1,0.9]

    #random weights
    # weights = randomize_weights(4)

    #weights at half data max and value
    # weights = [min(x_list), min(y_list),max(x_list)/2, max(y_list)/2, max(x_list), max(y_list)]
    # weights = [0.25,0.15, 0.1,0.9]

    # 7 neurons
    #random
    weights = randomize_weights(14)
    #manual
    weights = [0.25,0.15, 0.1,0.9, 0.5,0.5, 0.2,0.7, 0.7,0.8, 0.15, 0.9, 0.9,0.9]

    ow1 = [weights[0],weights[1]]
    ow2 = [weights[2],weights[3]]
    ow3 = [weights[4],weights[5]]
    ow4 = [weights[6],weights[7]]
    ow5 = [weights[8],weights[9]]
    ow6 = [weights[10],weights[11]]
    ow7 = [weights[12],weights[13]]
    
    weight_neuron1 = [weights[0],weights[1]]
    weight_neuron2 = [weights[2],weights[3]]
    weight_neuron3 = [weights[4],weights[5]]
    weight_neuron4 = [weights[6],weights[7]]
    weight_neuron5 = [weights[8],weights[9]]
    weight_neuron6 = [weights[10],weights[11]]
    weight_neuron7 = [weights[12],weights[13]]



    # print(list_length)
    value = 0

    # Original Itteration  Neurons
    # plotPoints2(df,ow1,ow2,weight_neuron1,weight_neuron2,0,x_list[0],y_list[0])

    # Original Itteration 3 Neurons
    # plotPoints3(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,0,x_list[0],y_list[0])

    # original Iteration 7 neurons
    plotPoints7(df,ow1,ow2,ow3,ow4,ow5,ow6,ow7,weight_neuron1,weight_neuron2,weight_neuron3,\
                weight_neuron4,weight_neuron5,weight_neuron6,weight_neuron7,0,x_list[0],y_list[0])
    # print(weight_neuron1)
    # print(weight_neuron2)
    # print(weight_neuron3)



    # for i in range(4):

    # value = 0
    print("Starting Run")
    print("Original Weights")
    print(weight_neuron1)
    print(weight_neuron2)
    # only for 3 neurons
    # print(weight_neuron3)
    for index in range(itterations):
        if(value > list_length -1):

            value = 0

        run = index + 1
        win_neuron = 0


        # print("Applying {} Itteration".format(index))
        
        net1 = train_neuron(x_list[value],y_list[value],weight_neuron1)
        net2 = train_neuron(x_list[value],y_list[value],weight_neuron2)

        if(number_neruons == 2):
            # net3 = train_neuron(x_list[value],y_list[value],weight_neuron3)
            if ((net1 > net2) ):
                win_neuron = 1
                weight_neuron1 = update_neuron(weight_neuron1,alpha,x_list[value],y_list[value])

            elif((net2 > net1)):
                win_neuron = 2
                weight_neuron2 = update_neuron(weight_neuron2,alpha,x_list[value],y_list[value])

            # else:

            #     win_neuron = 3
            #     weight_neuron3 = update_neuron(weight_neuron3,alpha,x_list[value],y_list[value])

            if (run % 500 == 0):

                plotPoints2(df,ow1,ow2,weight_neuron1,weight_neuron2,run,x_list[value],y_list[value])

        if(number_neruons == 3):
            net3 = train_neuron(x_list[value],y_list[value],weight_neuron3)
            if((net1 > net2) and (net1 > net3)):
                win_neuron = 1
                weight_neuron1 = update_neuron(weight_neuron1,alpha,x_list[value],y_list[value])

            elif((net2 > net1) and (net2 > net3)):
                win_neuron = 2
                weight_neuron2 = update_neuron(weight_neuron2,alpha,x_list[value],y_list[value])

            else:

                win_neuron = 3
                weight_neuron3 = update_neuron(weight_neuron3,alpha,x_list[value],y_list[value])

            if (run % 500 == 0):

                plotPoints3(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,run,x_list[value],y_list[value])

        if(number_neruons == 7):
            net3 = train_neuron(x_list[value],y_list[value],weight_neuron3)
            net4 = train_neuron(x_list[value],y_list[value],weight_neuron4)
            net5 = train_neuron(x_list[value],y_list[value],weight_neuron5)
            net6 = train_neuron(x_list[value],y_list[value],weight_neuron6)
            net7 = train_neuron(x_list[value],y_list[value],weight_neuron7)

            if((net1 > net2) and (net1 > net3) and(net1 > net4) and net1 > net5 and net1 > net6 and net1 > net7):
                win_neuron = 1
                weight_neuron1 = update_neuron(weight_neuron1,alpha,x_list[value],y_list[value])

            elif((net2 > net1) and (net2 > net3) and(net2 > net4) and net2 > net5 and net2 > net6 and net2 > net7):
                win_neuron = 2
                weight_neuron2 = update_neuron(weight_neuron2,alpha,x_list[value],y_list[value])

            elif((net3 > net1) and (net3 > net2) and(net3 > net4) and net3 > net5 and net3 > net6 and net3 > net7):
                win_neuron = 3
                weight_neuron3 = update_neuron(weight_neuron3,alpha,x_list[value],y_list[value])
            
            elif((net3 > net1) and (net3 > net2) and(net3 > net4) and net3 > net5 and net3 > net6 and net3 > net7):
                win_neuron = 4
                weight_neuron4 = update_neuron(weight_neuron4,alpha,x_list[value],y_list[value])
            
            elif((net5 > net1) and (net5 > net2) and(net5 > net4) and net5 > net3 and net5 > net6 and net5 > net7):
                win_neuron = 5
                weight_neuron5 = update_neuron(weight_neuron5,alpha,x_list[value],y_list[value])
            
            elif((net6 > net1) and (net6 > net2) and(net6 > net4) and net6 > net3 and net6 > net5 and net6 > net7):
                win_neuron = 6
                weight_neuron6 = update_neuron(weight_neuron6,alpha,x_list[value],y_list[value])
            
            elif((net7 > net1) and (net7 > net2) and(net7 > net4) and net7 > net3 and net7 > net5 and net7 > net6):
                win_neuron = 7
                weight_neuron7 = update_neuron(weight_neuron7,alpha,x_list[value],y_list[value])
            


            if (run % 500 == 0):
                plotPoints7(df,ow1,ow2,ow3,ow4,ow5,ow6,ow7,weight_neuron1,weight_neuron2,weight_neuron3,\
                weight_neuron4,weight_neuron5,weight_neuron6,weight_neuron7,run,x_list[0],y_list[0])
            
        


        
        


        value += 1

    print("Updated Weights are")
    # print("...................")
    print(weight_neuron1)
    print(weight_neuron2)
    # only with 3 neurons
    # print(weight_neuron3)
    print("...................")

    # 2 neurons
    # plotPoints2(df,ow1,ow2,weight_neuron1,weight_neuron2,"Final",x_list[0],y_list[0])
    # 3 neurons
    # plotPoints3(df,ow1,ow2,ow3,weight_neuron1,weight_neuron2,weight_neuron3,"Final",x_list[0],y_list[0])
    # 7 neurons
    plotPoints7(df,ow1,ow2,ow3,ow4,ow5,ow6,ow7,weight_neuron1,weight_neuron2,weight_neuron3,\
                weight_neuron4,weight_neuron5,weight_neuron6,weight_neuron7,"Final",x_list[0],y_list[0])






if __name__== "__main__":
  main()
