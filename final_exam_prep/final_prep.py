from datetime import date
import pandas as pd
import numpy as np


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
    print(value3))

def neuron4(value4):
    print(value4)

def loadData(filename):
    df = pd.read_csv(filename)
	# splitPercent_title = (splitPercent * 100)
	# title = "{} {}%".format(title,splitPercent_title)
    # df.columns = [,]
    return df;


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
    




if __name__== "__main__":
  main()
