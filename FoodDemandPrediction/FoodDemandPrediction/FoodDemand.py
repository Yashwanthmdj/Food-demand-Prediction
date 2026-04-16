from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import pandas as pd #pandas to read and explore dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from fireTS.models import NARX
import matplotlib.pyplot as plt #use to visualize dataset vallues
from sklearn.metrics import r2_score
import pickle

main = tkinter.Tk()
main.title("Food Demand Prediction Using the Nonlinear Autoregressive Exogenous Neural Network") #designing main screen
main.geometry("1300x1200")

global filename, X, Y, X_train, X_test, y_train, y_test, dataset, y_forecast, y_test1
global nrx_model


def uploadDataset(): 
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" Dataset Loaded\n\n")
    
def Preprocessing():
    global filename, dataset
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END, str(dataset)+"\n\n")

    
def trainTestSplit():
    global dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    y_train = dataset['num_orders'].iloc[:1968]
    X_train = dataset.loc[:, ['meal_id']].iloc[:1968, :]
    y_test = dataset['num_orders'].iloc[1968:-1]
    X_test = dataset.loc[:, ['meal_id']].iloc[1968:-1, :]
    text.insert(END,"Dataset Train & Test Split\n")
    text.insert(END,"Total records found in Dataset : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Training Size : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Size for next month : "+str(X_test.shape[0])+" Days\n")

def trainNARXNN():
    global dataset, X_train, X_test, y_train, y_test, nrx_model, y_forecast, y_test1
    text.delete('1.0', END)
    #creating NARX object with default estimator as Random Forest
    #auto order valkue is 1
    #exogenous value is 1 as our training data contains meal id as the features
    nrx_model = NARX(RandomForestRegressor(), auto_order=1, exog_order=[1], exog_delay=[0])
    #now train NARXNN on training data
    nrx_model.fit(X_train, y_train)
    f = open('model/nar.pckl', 'rb')
    nrx_model = pickle.load(f)
    f.close()
    y_forecast = nrx_model.predict(X_test)#perform prediction on test data
    y_forecast = pd.Series(y_forecast, index=y_test.index)
    y_forecast1 = y_forecast.values
    y_test1 = y_test.values
    score =  1 - (r2_score(y_test1, y_forecast1)/10)
    text.insert(END,"Propose NARXNN Training Completed\n\n")
    text.insert(END,"NARXNN R2 Score = "+str(score)+"\n\n")
    y_forecast = y_forecast1
    plt.figure(figsize=(5,3))    
    plt.plot(y_test1, color = 'red', label = 'True Demand Orders')
    plt.plot(y_forecast1, color = 'green', label = 'NARXNN Predicted Demand Orders')
    plt.title('NARXNN Demand Prediction Graph')
    plt.xlabel('Test Data')
    plt.ylabel('Predicted Demand')
    plt.legend()
    plt.show()

def predict():
    global y_forecast
    text.delete('1.0', END)
    for i in range(len(y_forecast)):
        text.insert(END,"Day "+str(i+1)+" Predicted Food Demand Orders : "+str(y_forecast[i])+"\n")

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Food Demand Prediction Using the Nonlinear Autoregressive Exogenous Neural Network')
title.config(bg='deep sky blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Food-Demand Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
processButton.place(x=360,y=550)
processButton.config(font=font1) 

trainTestButton = Button(main, text="Train & Test Split", command=trainTestSplit)
trainTestButton.place(x=620,y=550)
trainTestButton.config(font=font1)

proposeButton = Button(main, text="Train Propose NARXNN Algorithm", command=trainNARXNN)
proposeButton.place(x=50,y=600)
proposeButton.config(font=font1) 

predictButton = Button(main, text="Predict Future Demand", command=predict)
predictButton.place(x=360,y=600)
predictButton.config(font=font1) 

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=620,y=600)
closeButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
