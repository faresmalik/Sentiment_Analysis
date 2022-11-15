import numpy as np 
import re
import string
from nltk.corpus import stopwords, twitter_samples 
import nltk


def split_tweets(positive_tweets_path = 'positive_tweets.json',negative_tweets_path = 'negative_tweets.json', number_positive = 4000, number_negative = 4000):

    """
    This function used to split the tweets into training/validation. 

    positive_tweets, negative tweets: loaded from nltk 
    number_positive: number of positive tweets for training
    number_negative: number of negative tweets for training

    return: 
        tweets_train    
        tweets_val  
        tweets_train_labels 
        tweets_val_labels   
    """

    positive_tweets = twitter_samples.strings(positive_tweets_path)
    negative_tweets = twitter_samples.strings(negative_tweets_path)

    print(f'Number of positive tweets = {len(positive_tweets)} \t Number of negative tweets = {len(negative_tweets)}')

    #split tweets train/val 

    #positive 
    positive_train = positive_tweets[:number_positive]
    positive_val = positive_tweets[number_positive:]

    negative_train = negative_tweets[:number_negative]
    negative_val = negative_tweets[number_negative:]

    print(f'Training tweets: {len(positive_train)} Positive tweets and {len(negative_train)} Negative tweets')
    print(f'Validation tweets: {len(positive_val)} Positive tweets and {len(negative_val)} Negative tweets')

    #Concatenate positive and negative tweets for training and validation 
    tweets_train = positive_train + negative_train
    tweets_val = positive_val + negative_val

    #Create labels (0 for negative and 1 for positive)
    tweets_train_labels = np.append(np.ones(len(positive_train)), np.zeros(len(negative_train)))
    tweets_val_labels = np.append(np.ones(len(positive_val)), np.zeros(len(negative_val)))

    print(f'Number of training tweets: {len(tweets_train)}')
    print(f'Number of validation tweets: {len(tweets_val)}')
    
    return tweets_train, tweets_val, tweets_train_labels , tweets_val_labels, positive_train ,negative_train, positive_val, negative_val,positive_tweets ,negative_tweets 

def process_tweet(tweet):
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    stemmer = nltk.stem.PorterStemmer()
    stopwords_english  = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean

def get_vocabulary(train_tweets): 
    """
    This function used to build the vocabulary, and a dict for each word and an integer. 

    Input: 
        Training tweets
    
    Return: 
        vocabulary
    """

    vocab = {'__PAD__':0, '__</e>__':1, '__UNK__':2}

    for tweet in train_tweets: 
        proccessed_tweet = process_tweet(tweet)
        for word in proccessed_tweet: 
            if word not in vocab.keys(): 
                vocab[word] = len(vocab)
    
    return vocab

def tweet_to_tensor(tweet, vocabulary, unk_token = '__UNK__', process_tweet_fun = process_tweet): 

    """
    This function transform the tweet into a tensor of numbers. 

    Input: 
        Tweet 
    Return: 
        Tensor of numbers 
    """

    #process the tweet
    proccessed_tweet = process_tweet_fun(tweet)

    #get the unique integer for each word in the proccessed tweet
    int_list = []

    for word in proccessed_tweet: 
        if word not in vocabulary: 
            int_list.append(vocabulary[unk_token])
        else: 
            int_list.append(vocabulary[word])
    
    return int_list
