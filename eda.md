---
title: EDA on Tweets
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}

<hr style="height:2pt">


## Data Resources

Data is mined and collected from the Twitter developer API (using tweepy). Twitter API passes json format data, we fetched them for two groups of users, i) 23 famous bots ii)100 famous verified real accounts. We saved the fetched tweets from both group after cleaning as a dataframe format in a csv file. We currently have a bit over 12000 tweets and we plan to increase this database as the project moves forward. For each tweet, we have 28 variables that describe some feature of the tweet.

## EDA

Based on below preliminary EDA (at the end of this notebook) and histograms it seems that below feature can have predicting power. • Followers Count • Friends Count • Favorites Count • Retweet Count As we continue to explore our data, this might change a bit as we learn which variables might have predictive power.

We first create some helper functions that aid in fetching data from Twitter:

```Python
def get_api_2(screen_name,count):
    # defining twitter developer access
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
     
    tweets = []
    try:
        latest_tweets = api.user_timeline(screen_name = screen_name, count = count)
        tweets.extend(latest_tweets)
    except tweepy.TweepError:
        print('Error occured with users: {:s}, ignoring...'.format(screen_name))    
    
    return tweets
```

```Python
def store_tweets(tweets,file_name='data/tweets.json'):
    
    features = ['created_at', 'id', 'text',  'source','in_reply_to_status_id',  'geo', 
             'retweet_count', 'favorite_count', 'favorited', 'lang']
    sub_tags = ['entities', 'user', 'place']
    
    entities_tag = ['hashtags', 'urls']
    
    user_tag = ['id', 'name', 'screen_name', 'location', 'url','protected', 'followers_count', 'friends_count',
                'created_at', 'geo_enabled', 'verified', 'lang']
    
    place_tag = ['id', 'place_type', 'name', 'country_code']

    
    # a list of all formatted tweets
    tweet_list=[]

    for tweet in tweets:
 
        # a dict to contain information about single tweet
        tweet_information=dict()
        
        for feat in features:
            if (feat == 'created_at'):
                tweet_information['created_at']=tweet.created_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                tweet_information[feat]=tweet._json[feat]
        for s_tag in sub_tags:
            if (s_tag == 'entities'):
                for s_t in entities_tag:
                    if (s_t=='hashtags'):
                        tweet_information[s_t]=tweet._json['entities'][s_t]
#                         if (tweet._json['entities'][s_t]!= None):
#                             tweet_information[s_t]=tweet._json['entities'][s_t][0]['text']
#                         else:
#                             tweet_information[s_t]= np.nan
                        
                    else:
                        tweet_information[s_t]=tweet._json['entities'][s_t]
              
            elif (s_tag == 'user'):
                for s_t in user_tag:
                    tweet_information[s_t]=tweet._json['user'][s_t]
            
            elif (s_tag == 'place'):
                for s_t in place_tag:
                    if (tweet._json['place']!= None):
                        tweet_information[s_t]=tweet._json['place'][s_t] 
                    else:
                        tweet_information[s_t]= np.nan
    

        # add this tweet to the tweet_list
        tweet_list.append(tweet_information)
    #Save dictionary into json file
    with open(file_name,"w") as fd:
        json.dump( tweet_list, fd)

    # close the file_des
    fd.close()
    return tweet_list
```

```Python
def fetch_tweets(screen_names, count):
    alltweets=[]
    # get all tweets for each screen name
    for  screen_name in screen_names:
        alltweets.extend(get_api_2(screen_name,count=count))

    return alltweets
```
