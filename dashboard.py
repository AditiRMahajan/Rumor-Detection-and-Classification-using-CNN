import streamlit as st
import pandas as pd

def get_tweets(username):
    from scrape_twitter import TwitterClient
    twitter_api = TwitterClient()
    filename = twitter_api.get_tweets(screen_name=username, count=200)
    return filename


@st.cache
def preprocess_data():
    from pheme_data import main
    main()

@st.cache
def train_model(batch_size, epochs, validation_split):
    from keras.models import Sequential
    from keras.layers import Dropout, Dense, Conv1D, Flatten
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    
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
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, Y_train, batch_size=batch_size, validation_split=validation_split, epochs=epochs)
    pred = model.predict(X_test,)

    Y_pred = np.zeros((len(pred), 2))
    for i in range(len(pred)):
        pred_class = np.argmax(pred[i])
        Y_pred[i][pred_class] = 1

    target_names = ['Non-Rumours', 'Rumours']
    report_data = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)
    report = pd.DataFrame.from_dict(report_data)[target_names].transpose()
    score = accuracy_score(Y_test, Y_pred) * 100

    return score, report

st.sidebar.image("twitter.png",use_column_width=True)
st.sidebar.title("Twitter Rumour Detection")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Show instructions", "Get Tweets", "Preprocess Data", "Train Model"])
if app_mode == "Show instructions":
    st.sidebar.success('To continue select any one')

elif app_mode == "Get Tweets":
    st.title("Retrieve User Tweets")
    username = st.text_input("Enter Twitter Handle", value="gsanjeev432")
    if st.button("Get Tweets"):
        with st.spinner('Please wait, retrieving user tweets...'):
            filename = get_tweets(username)
        my_bar = st.progress(0)
        import time
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        st.success("All tweets downloaded to "+filename)
        st.subheader("Preview of the tweets file...")
        data = pd.read_csv(filename)
        st.write(data)

elif app_mode == "Preprocess Data":
    st.title("Preprocess Data")
    st.header("Do you want to preprocess the Data..??")
    if st.button("Yes"):
        with st.spinner('Please wait, Preprocessing Data...'):
            preprocess_data()
        st.success("Done, with Preprocessing Data")
    elif st.button("No"):
        st.title("OK")

elif app_mode == "Train Model":
    st.title("Train Model")
    st.sidebar.subheader("Select Training Parameters")
    batch_size = st.sidebar.slider('Batch Size', 16, 128, 32)
    epochs = st.sidebar.slider('Epochs', 1, 50, 20)
    validation_split = st.sidebar.slider('Validation Split (%)', 0.0, 0.5, 0.2)
    file = st.file_uploader("Choose Training Data file",
                            type="pkl", encoding=None)
    import pickle
    if file is not None:
        (X_data, X_label) = pickle.load(file)
    file = st.file_uploader("Choose Testing Data file",
                            type="pkl", encoding=None)
    if file is not None:
        (Y_data, Y_label) = pickle.load(file)
        if st.button('Train Model'):
            score, report = train_model(batch_size, epochs, validation_split)
            st.text("Accuracy of Model is: ")
            st.write(score,"%")
            st.text("Report of Model is: ")
            st.write(report)
            output_file = "Result.txt"
            with open(output_file,"w") as f:
                f.write("Accuracy of Model is: ")
                f.write(str(score)+"%")
                f.write("\nReport of Model is: \n")
                f.write(str(report))