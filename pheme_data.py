import os
import json
import nltk
from gensim.models import KeyedVectors
import string
from nltk.tokenize import RegexpTokenizer
import pickle
import re
from textblob import TextBlob


filename = "GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)
f = open("model.pkl", "wb")
pickle.dump(model, f)
f.close()
stop = set(nltk.corpus.stopwords.words("english"))


class Data:
    def __init__(self, structure, tweet_data, source):
        self.structure = structure
        self.tweet_data = tweet_data
        self.source = source

    def getfeature(self, tweet):
        text = tweet["text"]
        feature = []
        words = nltk.word_tokenize(text)

        tokenizer = RegexpTokenizer(r"\w+")
        word_nopunc = tokenizer.tokenize(text)
        word_nopunc = [i for i in word_nopunc if i not in stop]

        # top 20 features using word2vec
        for i in word_nopunc:
            if i in model.wv:
                feat_list = model.wv[i].tolist()
                feature.extend(feat_list[:20])

        # append 0 if no feature found
        if len(feature) < 100:
            for i in range(len(feature), 101):
                feature.append(0)
        feature = feature[:100]

        # Has question marks
        if text.find("?") > 0:
            feature.append(1)
        else:
            feature.append(0)

        # has !
        if text.find("!") > 0:
            feature.append(1)
        else:
            feature.append(0)

        # has hastag
        if len(tweet["entities"]["hashtags"]) > 0:
            # feature.append(len(tweet['entities']['hashtags']))
            feature.append(1)
        else:
            feature.append(0)

        # has usermention
        if len(tweet["entities"]["user_mentions"]) > 0:
            # feature.append(len(tweet['entities']['user_mentions']))
            feature.append(1)
        else:
            feature.append(0)

        # has url
        if len(tweet["entities"]["urls"]) > 0:
            # feature.append(len(tweet['entities']['urls']))
            feature.append(1)
        else:
            feature.append(0)

        # has media
        if "media" in tweet["entities"]:
            # feature.append(len(tweet['entities']['media']))
            feature.append(1)
        else:
            feature.append(0)

        # sentiment analysis
        clean_tweet = " ".join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()
        )
        analysis = TextBlob(clean_tweet)

        if analysis.sentiment.polarity > 0:
            feature.append(1)
        else:
            feature.append(0)

        # Capital to lower case ratio
        uppers = [l for l in text if l.isupper()]
        capitalratio = len(uppers) / len(text)
        feature.append(capitalratio)

        count_punct = 0
        # negative words list
        neg_words = [
            "not",
            "no",
            "nobody",
            "none",
            "never",
            "neither",
            "nor",
            "nowhere",
            "hardly",
            "scarcely",
            "barely",
            "don't",
            "isn't",
            "wasn't",
            "shouldn't",
            "wouldn't",
            "couldn't",
            "doesn't",
        ]

        count_neg_words = 0
        # count number of punctuations and negative words
        for i in words:
            if i in (string.punctuation):
                count_punct += 1
            if i in neg_words:
                count_neg_words += 1

        feature.append(count_punct)
        feature.append(count_neg_words)
        swearwords = []
        with open("badwords.txt", "r") as f:
            for line in f:
                swearwords.append(line.strip().lower())

        hasswearwords = 0
        for token in word_nopunc:
            if token in swearwords:
                hasswearwords += 1
        feature.append(hasswearwords)

        return feature

    def extract_features(self):
        feat_dict = {}
        for i in self.tweet_data:
            feat_dict[i] = self.getfeature(self.tweet_data[i])
        return feat_dict

def main():
    path = "../pheme-rnr-dataset"
    
    data = []
    fold = os.listdir(path)
    
    print("Processing Training Data...........")
    
    # Read DATA
    X_label = {}
    for k in fold:
        labels = path + "/" + k
        labels_dir = os.listdir(labels)
    
        for labels in labels_dir:
            temp_files = path + "/" + k + "/" + labels
            print(temp_files)
            lis = []
            temp_inner = os.listdir(temp_files)[0:100]
    
            # Get data for each topic
            for i in temp_inner:
                if i == ".DS_Store" or i == ".":
                    continue
                X_label.update({i: labels}.items())
                temp_source = temp_files + "/" + i + "/source-tweet/"
                temp_replies = temp_files + "/" + i + "/reactions/"
    
                # store source tweet
                source_file = os.listdir(temp_source)
                source = source_file[0].split(".")[0]
    
                # store all twitter data
                tweet_data = {}
                with open(temp_source + source_file[0]) as f:
                    tweet_data[source] = json.load(f)
    
                reply_file = os.listdir(temp_replies)
                for j in reply_file:
                    with open(temp_replies + j) as f:
                        tweet_data[j.split(".")[0]] = json.load(f)
    
                lis.append(Data(None, tweet_data, source))
            data.append(lis)
    
    # Find feature vectors for each tweet
    X_data = {}
    for i in data:
        for j in i:
            X_data.update(j.extract_features().items())
    
    # Get Test data
    path = "../pheme-rnr-dataset"
    
    test_data = []
    fold = os.listdir(path)
    
    print("Processing Training Data...........")
    
    # Read DATA
    Y_label = {}
    for k in fold:
        labels = path + "/" + k
        labels_dir = os.listdir(labels)
    
        for labels in labels_dir:
            temp_files = path + "/" + k + "/" + labels
            print(temp_files)
            lis = []
            temp_inner = os.listdir(temp_files)[0:100]
    
            # Get data for each topic
            for i in temp_inner:
                if i == ".DS_Store" or i == ".":
                    continue
                Y_label.update({i: labels}.items())
                temp_source = temp_files + "/" + i + "/source-tweet/"
                temp_replies = temp_files + "/" + i + "/reactions/"
    
                # store source tweet
                source_file = os.listdir(temp_source)
                source = source_file[0].split(".")[0]
    
                # store all twitter data
                tweet_data = {}
                with open(temp_source + source_file[0]) as f:
                    tweet_data[source] = json.load(f)
    
                reply_file = os.listdir(temp_replies)
                for j in reply_file:
                    with open(temp_replies + j) as f:
                        tweet_data[j.split(".")[0]] = json.load(f)
    
                lis.append(Data(None, tweet_data, source))
            test_data.append(lis)
    
    # get testing features
    Y_data = {}
    for i in test_data:
        for j in i:
            Y_data.update(j.extract_features().items())
    
    
    f = open("training.pkl", "wb")
    pickle.dump((X_data, X_label), f)
    f.close()
    
    f = open("testing.pkl", "wb")
    pickle.dump((Y_data, Y_label), f)
    f.close()
    
    print("Done.!!!!")
    
if __name__ == "__main__":
    main()
