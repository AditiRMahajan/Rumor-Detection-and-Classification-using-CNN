import PySimpleGUI as sg
from scrape_twitter import TwitterClient
from keras.models import Sequential
from keras.layers import Dropout, Dense
import pickle
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from pheme_data import main

print=sg.Print
sg.theme('Black')
layout = [     
    [sg.Button("Get Tweets",size=(20,2))],
    [sg.Button("Preprocess Data",size=(20,2))],
    [sg.Button("Train",size=(20,2))],
    [sg.Text("Training Parameters",font=('Helvetica', 15))],
    [sg.Text("Batch Size       ",font=('Helvetica', 14)),sg.Slider(range=(16,128),
         default_value=32,
         size=(30,30),
         orientation='horizontal',
         font=('Helvetica', 14),key='batch_size')],
    [sg.Text("Epochs            ",font=('Helvetica', 14)),sg.Slider(range=(1,50),
         default_value=20,
         size=(10,15),
         orientation='horizontal',
         font=('Helvetica', 14),key='epochs')],
    [sg.Text("Validation Split",font=('Helvetica', 14)),sg.Slider(range=(0,50),
         default_value=20,
         size=(10,15),
         orientation='horizontal',
         font=('Helvetica', 14),key='validation_split')]
    ]
window = sg.Window("Twitter", layout,size=(400,400))

while True:  # The Event Loop
    event, values = window.read()
    #    print(event, values) #debug
    if event in (None, "Cancel"):
        break
    if event == "Get Tweets":
        username = sg.popup_get_text('Enter Twitter Handle','Get Tweets',default_text='gsanjeev432')
        sg.popup("Please Wait","Please wait while the tweets are being downloaded")
        # creating object of TwitterClient Class
        twitter_api = TwitterClient()
        # calling function to get tweets
        twitter_api.get_tweets(screen_name=username, count=200)
        sg.popup('DONE', "All tweets downloaded for "+username)
        
    if event == "Preprocess Data":
        sg.popup("Please Wait","Please wait while the data is being processed")
        main()
        

        
    if event == "Train":
        file = sg.popup_get_file("Choose Training Data file","Training Data",default_path="training_data.pkl")
        f = open(file,"rb")
        (X_data, X_label) = pickle.load(f)
        f.close()
    
        # Read test data
        file = sg.popup_get_file("Choose Testing Data file","Testing Data",default_path="testing_data.pkl")
        f = open(file, "rb")
        (Y_data, Y_label) = pickle.load(f)
        f.close()
    
        X_train = []
        Y_train = []
    
        label2no = {"rumours": 1, "non-rumours": 0}
    
        for key in X_label.keys():
            temp_lb = np.zeros(2)
            X_train.append(X_data[key])
            temp_lb[label2no[X_label[key]]] = 1
            Y_train.append(temp_lb)
    
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
    
        X_test = []
        Y_test = []
    
        for key in Y_label.keys():
            temp_lb = np.zeros(2)
            X_test.append(Y_data[key])
            temp_lb[label2no[Y_label[key]]] = 1
            Y_test.append(temp_lb)
    
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
    
        model = Sequential()
        model.add(Dense(512, activation="relu", input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation="sigmoid"))
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
        model.fit(X_train, Y_train, batch_size=int(values['batch_size']), validation_split=float(values['validation_split']/100), epochs=int(values['epochs']))
        pred = model.predict(X_test,)
    
        Y_pred = np.zeros((len(pred), 2))
        for i in range(len(pred)):
            pred_class = np.argmax(pred[i])
            Y_pred[i][pred_class] = 1
        print(classification_report(Y_test, Y_pred))
        print("accuracy - ")
        print(accuracy_score(Y_test, Y_pred))

window.close()
