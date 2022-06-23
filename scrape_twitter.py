import csv
import re

import tweepy
from tweepy import OAuthHandler


class TwitterClient(object):
    """ 
    Generic Twitter Class
    """

    def __init__(self):
        """ 
        Class constructor or initialization method. 
        """
        # keys and tokens from the Twitter Dev Console
        consumer_key = "YGhEN9z1kaaKezibJomv9VFL1"
        consumer_secret = "E0seBRhjPd6gWWCjiBgpGnOXfLpwCnrzUAg64MKuOI89Encz3m"
        access_token = "2460342872-yFV4A99zCMwHLJZ2aqEXtOwvqFj6F1tKFs7yXc3"
        access_token_secret = "BAdy9uQszVGOe5wB3HYPiMyQ2kHSpjQ1WnGgqm9Jr1GPW"

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        """ 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        """
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+) | ([ ^ 0-9A-Za-z \t]) | (\w+: \/\/\S+)", " ", tweet
            ).split()
        )

    def get_tweets(self, screen_name, count=10):
        """ 
        Main function to fetch tweets and parse them. 
        """
        # initialization of a list to hold all Tweets

        all_the_tweets = []

        try:
            # call twitter api to fetch tweets
            new_tweets = self.api.user_timeline(
                screen_name=screen_name, count=count, tweet_mode="extended"
            )

            # saving the most recent tweets
            all_the_tweets.extend(new_tweets)

            # save id of 1 less than the oldest tweet
            oldest_tweet = all_the_tweets[-1].id - 1

            # grabbing tweets till none are left
            while len(new_tweets) > 0:
                # The max_id param will be used subsequently to prevent duplicates
                new_tweets = self.api.user_timeline(
                    screen_name=screen_name,
                    count=count,
                    max_id=oldest_tweet,
                    tweet_mode="extended",
                )

                # save most recent tweets
                all_the_tweets.extend(new_tweets)

                # id is updated to oldest tweet - 1 to keep track
                oldest_tweet = all_the_tweets[-1].id - 1
                print("...%s tweets have been downloaded so far" % len(all_the_tweets))

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

        # transforming the tweets into a 2D array that will be used to populate the csv
        outtweets = [
            [tweet.id_str, tweet.created_at, self.clean_tweet(tweet.full_text)]
            for tweet in all_the_tweets
        ]

        # writing to the csv file
        with open(screen_name + "_tweets.csv", "w", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "created_at", "text"])
            writer.writerows(outtweets)
            
        return screen_name + "_tweets.csv"



if __name__ == "__main__":
    # creating object of TwitterClient Class
    twitter_api = TwitterClient()
    # calling function to get tweets
    twitter_api.get_tweets(screen_name="BarackObama", count=200)
