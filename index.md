---
title: Detecting Twitter Bots
---


![google recaptcha](https://developers.google.com/recaptcha/images/newCaptchaAnchor.gif)
<p style="font-size:70%;"><i>source: google recaptcha</i></p>

>By Yalda Amini, João Araujo and Daniel Barjum, May 2019

## Introduction and Problem Statement

With spread of social media and their increasing impact on the news the topic of bots and spread of fake news is one of the hot topics now. The goal of this project is to use machine learning techniques which we have learned during this semester to detect the tweets which were generated by Twitter bots vs generated by human users.

We use data from two main sources:
1. We requested permission and obtained the data from MIB researcher’s database. Detailed information on the dataset can be found here: http://mib.projects.iit.cnr.it/dataset.html. 
2. When necessary, we complemented this data using Twitters API.

We then clean the datasets we obtain and prepare them to be used in our models.

Our models are two Multileyer Preceptron Neural Networks, one that works as a baseline model and the second an improved model. We test the results of our model agains the results of a well established neural net that does this developed by researchers at  Indiana University Network Science Institute (IUNI) and the Center for Complex Networks and Systems Research (CNetS). Information on Botometer can be found [here](https://botometer.iuni.iu.edu/#!/).

Our results based on a total of 1,013 users that we tested against Botometer was of 88.2% accuracy. This means that for 88.2% of our scores where within 10% of the score given by Botometer.

## Literature Review

Bots have a heavy presence in the social media. “Of all tweeted links to popular websites, 66% are shared by accounts with characteristics common among automated “bots,” rather than human users.” . Due to this huge impact of pots on many political, social and economical topics around the world There is considerable literature trying to solve the issue of the fake bot generated contents in social media. Based on Ferrera et all paper in 2018, there are three types of approaches to detect bot generated contents in social media, consisting “(a) methods based on social network; (b) systems based on crowd-sourcing and human computation; (c) algorithms based on predictive features that separate bots from humans“ . Recently using RNN and long short-term memory (LSTM) networks seems to provide good performance for bot detection.

*References*
1. Tweepy Python Library.
2. Twitter’s developer resources: [developer.twitter.com](developer.twitter.com).
3. Kudugunta, Sneha, and Emilio Ferrara. [“Deep Neural Networks for Bot Detection.” Information Sciences 467 (October 2018)](https://doi.org/10.1016/j.ins.2018.08.019).
4. Twitter Bots: [An Analysis of the Links Automated Accounts Share, Pew Research Center,” April 9, 2018](https://www.pewinternet.org/2018/04/09/bots-in-the-twittersphere/).
