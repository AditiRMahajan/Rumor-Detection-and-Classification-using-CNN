# MODEL
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D, Flatten
import pickle
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd


def main():
    # Read training data
    f = open("training_data.pkl", "rb")
    (X_data, X_label) = pickle.load(f)
    f.close()

    # Read test data
    f = open("testing_data.pkl", "rb")
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
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    Y_train = np.array(Y_train)

    X_test = []
    Y_test = []

    for key in Y_label.keys():
        temp_lb = np.zeros(2)
        X_test.append(Y_data[key])
        temp_lb[label2no[Y_label[key]]] = 1
        Y_test.append(temp_lb)

    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    Y_test = np.array(Y_test)

    model = Sequential()
    model.add(Conv1D(32, (3), input_shape=(X_train.shape[1],1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, Y_train, batch_size=32, validation_split=0.2, epochs=20)
    pred = model.predict(X_test,)

    Y_pred = np.zeros((len(pred), 2))
    for i in range(len(pred)):
        pred_class = np.argmax(pred[i])
        Y_pred[i][pred_class] = 1
    target_names = ['Non-Rumours', 'Rumours']
    report_data = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)
    report = pd.DataFrame.from_dict(report_data)[target_names].transpose()
    # report = report[target_names].transpose()
    print(report)
    print("accuracy - ")
    print(accuracy_score(Y_test, Y_pred))


if __name__ == "__main__":
    main()
