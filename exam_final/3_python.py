from datetime import date
from mpl_toolkits import mplot3d
from matplotlib import cm
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

# calculates value from peak and returns that number
def distance_from_center(center, total_distance, point):
    num = 1 - ( abs(center - point) / total_distance)
    return num
    
# assigns values to different singletons
def singleton_value(value):
    num = 0
    if(value == "E"):
        num = 0.1
    elif(value == "D"):
        num = 0.2
    elif(value == "C"):
        num = 0.4
    elif(value == "B"):
        num = 0.5
    elif(value == "A"):
        num = 0.6
    elif(value == "G"):
        num = 0
    return num

# assigns a singleton based on row and column value is in
def singlton_output(col, row):
    singleton = "F"
    if (row == 1):
        if(col == 1):
            singleton = "E"
        elif(col == 2):
            singleton = "E"
        elif(col == 3):
            singleton = "D"
        elif(col == 4):
            singleton = "D"
        elif(col == 5):
            singleton = "D"
    elif(row == 2):
        if(col == 1):
            singleton = "E"
        elif(col == 2):
            singleton = "D"
        elif(col == 3):
            singleton = "C"
        elif(col == 4):
            singleton = "C"
        elif(col == 5):
            singleton = "D"

    elif(row == 3):
        if(col == 1):
            singleton = "E"
        elif(col == 2):
            singleton = "D"
        elif(col == 3):
            singleton = "C"
        elif(col == 4):
            singleton = "B"
        elif(col == 5):
            singleton = "C"
    
    elif(row == 4):
        if(col == 1):
            singleton = "E"
        elif(col == 2):
            singleton = "D"
        elif(col == 3):
            singleton = "C"
        elif(col == 4):
            singleton = "B"
        elif(col == 5):
            singleton = "B"
    
    elif(row == 5):
        if(col == 1):
            singleton = "E"
        elif(col == 2):
            singleton = "C"
        elif(col == 3):
            singleton = "B"
        elif(col == 4):
            singleton = "A"
        elif(col == 5):
            singleton = "B"
    
    return singleton

# return min value
def min_val(x,y):
    if(x < y):
        return x
    return y

# assign  col's and rows to x and y cordinates
# sends it to singleton output
def fuzzy_table(x,y):
    col = 0
    row = 0

    if(x <= 4):
        col = 1
    elif(x> 4 and x <= 8):
        col = 2
    elif(x > 8 and x <= 12):
        col = 3
    elif(x > 12 and x <= 16):
        col = 4
    elif(x > 16):
        col = 5
    
    if(y <= 4):
        row = 1
    elif(y> 4 and y <= 8):
        row = 2
    elif(y > 8 and y <= 12):
        row = 3
    elif(y > 12 and y <= 16):
        row = 4
    elif(y > 16):
        row = 5
    print("Col : Row")
    print(col,row)
    singleton = singlton_output(col, row)
    
    return singleton


#checks if singletons are the same and if they are return max value and letter of that singleton
# if no singleton is there or only one letter for everything returns a nonsense letter G
def max_singleton(min1, min2, min3, min4, point1, point2, point3, point4):
    list1 = [min1,min2,min3,min4]
    print("min list")
    print(list1)
    list2 = [point1,point2,point3,point4]
    print("point for letters")
    print(list2)
    a_list, b_list, c_list, d_list, e_list =[ [] for i in range(5)]
    list_max = [0,0]
    list_max_Letter = []
    list_singleton = ["A","B","C","D","E"]
    # print("test12lk3j")
    print(list_singleton[2])
    for num in range(len(list2)):
        if(list2[num] == "E"):
            e_list.append(list1[num])
        if(list2[num] == "D"):
            d_list.append(list1[num])
        if(list2[num] == "C"):
            # print("test")
            c_list.append(list1[num])
        if(list2[num] == "B"):
            b_list.append(list1[num])
        if(list2[num] == "A"):
            a_list.append(list1[num])

    list_all = [a_list, b_list, c_list, d_list, e_list]

    
    
    # if(len(a_list) > 0):
    #     max_val = max(a_list)
    #     list_max.append(max_val)
    #     list_max_Letter.append("A")
    # if(len(b_list) > 0):
    #     max_val = max(b_list)
    #     list_max.append(max_val)
    #     list_max_Letter.append("B")
    # if(len(c_list) > 0):
    #     max_val = max(c_list)
    #     list_max.append(max_val)
    #     list_max_Letter.append("C")
    # if(len(d_list) > 0):
    #     max_val = max(d_list)
    #     list_max.append(max_val)
    #     list_max_Letter.append("D")
    # if(len(e_list) > 0):
    #     max_val = max(e_list)
    #     list_max.append(max_val)
    #     list_max_Letter.append("E")
    # else
    # list_all.sort(reverse = True)
    # print(list_all)
    # for
    # print("lkajsdf {}".format(list_max))

    for i in range(len(list_all)):
        if(len(list_all[i]) > 0):
            max_val = max(list_all[i])
            # print(max_val)
            # print(list_singleton[i])
            if(max_val > 0 ):
                list_max.append(max(list_all[i]))
                list_max_Letter.append(list_singleton[i])
        else:
            list_max.append(0)
            list_max_Letter.append("G")
    
    # print(list_max[0], list_max[1])
    # print(list_max_Letter[0], list_max_Letter[1] )
    
    list_max.sort(reverse = True)
    # print(list_max)
    list_max_Letter.sort(reverse = True)
    return list_max[0], list_max[1], list_max_Letter[0], list_max_Letter[1] 
    # return ("print")
    # if(a_min =)

def defuzzy_output (letter_1,letter_2, max_1, max_2):
    if(max_1 + max_2 == 0):
        output = 0
    else:
        output = (max_1 * singleton_value(letter_1) + max_2 * singleton_value(letter_2)) / (max_1 + max_2)
    return output
   
#starts by fuzzyfying the input
#returns two x and ys to find which singleton it belongs too to plot points
def fuzzy_value(x,y):
    x1, x2, out1, out2 = fuzzfier(x)
    y1, y2, out3, out4 = fuzzfier(y)

    # print("x cordinates are")
    # print(x1, x2, out1, out2)
    # print("y cordinates are")
    # print(y1, y2, out3, out4)

    min1 = min_val(out1,out3)
    min2 = min_val(out2,out3)
    min3 = min_val(out1,out4)
    min4 = min_val(out2,out4)

    point1 = fuzzy_table(x1,y1)
    point2 = fuzzy_table(x2,y1)
    point3 = fuzzy_table(x1,y2)
    point4 = fuzzy_table(x2,y2)

    max_1, max_2, letter_1, letter_2 = max_singleton(min1, min2, min3, min4, point1, point2, point3, point4)
    # test_run = max_singleton(min1, min2, min3, min4, point1, point2, point3, point4)

    # print("1")
    # print(max_1)
    # print("2")
    # print(max_2)
    # print("A:::")
    # print(letter_1)
    # print("B::::")
    # print(letter_2)

    z_score =  defuzzy_output(letter_1,letter_2, max_1, max_2)
    return z_score

    # singleton_value(letter_1) * max_1 + letter





    # if(singleton)

# hard coded triangles that have a range of values for x to go into
# also returns decimal place that it goes in
def fuzzfier(x):
    first_try = 0
    second_try = 0
    x1 = 0
    x2 = 0
    # print(" Input is {}".format(x))
    if (x <= 5):
        first_try = 1
        second_try = 0
        x1 = 6
        x2 = 6
    elif ( x > 5 and x < 10):
        first_try = distance_from_center(6,3.8,x)
        second_try = distance_from_center(10,5,x)
        x1 = 6
        x2 = 10
    elif(x == 10):
        first_try = 1
        second_try = 0
        x1 = 10
        x2 = 10
    elif( x > 10 and x < 12 ):
        first_try = distance_from_center(10,1.8,x)
        second_try = distance_from_center(12,1.7,x)
        x1 = 10
        x2 = 12
    elif(x == 12):
        first_try = 1
        second_try = 0
        x1 = 12
        x2 = 12
    elif(x > 12 and x < 14):
        first_try = distance_from_center(12,2.5,x)
        second_try = distance_from_center(14,1.8,x)
        x1 = 12
        x2 = 14
    elif(x == 14):
        first_try = 1
        second_try = 0
        x1 = 14
        x2 = 14
    elif(x > 14 and x < 18):
        first_try = distance_from_center(14,2.7,x)
        second_try = distance_from_center(18,4.1,x)
        x1 = 14
        x2 = 18
    elif(x > 18):
        x1 = 18
        x2 = 18
    # print(first_try)
    # print(second_try)
    # print("two coordinates are")
    # print(x1,x2)
    return x1, x2, first_try, second_try

def plot_countour_real():
    xlist = np.linspace(0,20)
    ylist = np.linspace(0, 20)
    X, Y = np.meshgrid(xlist, ylist)
    Z = 0.6 * np.exp((-0.003 * (X - 20) ** 2) - ( 0.015 * (Y -14)** 2))
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    #ax.set_xlabel('x (cm)')
    ax.set_ylabel('y label')
    ax.set_xlabel("x label")
    plt.show()


def plot_mesh_real():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, 20, 0.25)
    Y = np.arange(0, 20, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = 0.6 * np.exp((-0.003 * (X - 20) ** 2) - ( 0.015 * (Y -14)** 2))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap= cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 0.6)
    ax.set_ylabel('y label')
    ax.set_xlabel("x label")
    
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# def plot_countour_tringle(z_list):
#     ig, ax = plt.subplots()
#     CS = ax.contour(X, Y, Z)
#     x_list = np.arange(0, 20, 0.25)
#     y_list = np.arange(0, 20, 0.25)
#     # X, Y = np.meshgrid(X, Y)
    
#     manual_locations = [x_list, y_List, z_list]
#     ax.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
#     ax.set_title('labels at selected locations')
#     plt.show()

def plot_mesh_triangle(z_list):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, 20, 0.25)
    Y = np.arange(0, 20, 0.25)

    X, Y = np.meshgrid(X, Y)
    Z = fuzzy_value(X,Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.asarray(z_list), cmap= cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 0.6)
    ax.set_ylabel('y label')
    ax.set_xlabel("x label")
    
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# main function
def main():
    # print("hello World")
    today = date.today()
    d3 = today.strftime("%m/%d/%y")
    chady = Student("Chady Aboulhosn", "CMSC 409","Problem 3" , d3)
    chady.show()

    # real_z = 0.6 * math.exp(-0.003*(x)-20) ** 2-0.015*(y)-14)**2)
    # test_z = 0.6 * math.exp((-0.003 * (x - 20) ** 2) -( 0.015 * (y -14)** 2))

    
    

    z_list = list()
    z_all_list = list()
    # for i in range(20):
    #     for j in range(20):
    #         z_list.append(fuzzy_value(i,j))
    #     z_all_list.append(z_list)
    # print(len(z_all_list))
    # print(len(z_list))
    # np_test  = np.arange(400).reshape(20,20)
    # # print(np_test)
    # # np_test[1,1] = 150
    # # print(np_test)
    # for i in range(20):
    #     for j in range(20):
    #         # input = fuzzy_value(i,j)
    #         # print(input)
    #         # print("{} {} {}".format(i,j,fuzzy_value(i,j) ))
    #         # np_z.insert(fuzzy_value(i,j),i,j)
    #         np_test[i,j] = 5
            
            
    
    # np_test[0,1] = 150
    # print(np_test)
    z_out_1 = fuzzy_value(5,5)
    print("final output for coordinates 5, 5 is : {}".format(z_out_1))
    z_out = fuzzy_value(16,16)
    print("final output for coordinates 16, 16 is : {}".format(z_out))

    # z_1 = fuzzy_value(11.5,11.5)
    # print(z_1)
    # plot_countour_real()
    # plot_mesh_real()
    # plot_mesh_triangle(z_list)
    # plot_countour_tringle(z_list)



    # fuzzfier_x(7)
    # chady.show()





if __name__== "__main__":
  main()
