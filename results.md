---
title: EDA on Tweets and Data Description
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}

<hr style="height:2pt">

## Comparing Results Against Botometer

In our initial results, we measured accuracy of our models by comparing predictions to the classified tweets from the researchers' data. We obtained probabilities, and we assumed that a probability above 0.5 was bot and below was genuine. This metric is vulnerable to the choice of threshold used. Perhaps a better metric would be to compare with an already established machine learning model for detecting if a user is a bot or not.

This is what we did in the following section. We obtained the probability of a user in the test set to be a bot or not based on the result of Botometer. We then compared our predicted score to the Botometer score and test how many of our predictions fall within 10% of the Botometer score for any given user.

We read in accounts from our test set to be used in Botometer.

```Python
accounts = pd.read_csv('data/botometer_df.csv', index_col=0)
accounts.shape

Output:
(1659, 2)
```

We request information for the accounts to Botometer. Unfortunatley, not all accounts in our test set are still twitter active accounts. Therefore we try to get information, but if twitter has no info on these accounts, we append a Null to flag it later and ignore it.

```Python
bot_pred = []
for i in accounts.user_id:
    try:
        result = bom.check_account(str(i))
        bot_pred.append(result['display_scores']['english'])
    except:
        bot_pred.append(np.nan)
        
accounts['botometer_score'] = bot_pred
accounts.head()
```


<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>user_id</th>
      <th>y_pred</th>
      <th>botometer_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>722623</td>
      <td>1.505279e-05</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>785080</td>
      <td>1.312385e-05</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>806585</td>
      <td>1.018241e-05</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2770511</td>
      <td>1.467852e-05</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3873101</td>
      <td>1.385364e-07</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>

Botometer returns a probability score between 0 and 5, 0 for human and 5 for bot. We normalize these scores in order to compare with our results.

```Python
accounts['bot_score_norm']=(accounts.botometer_score-accounts.botometer_score.min())/(accounts.botometer_score.max()-accounts.botometer_score.min())
```

We check to see if our score is within 10% of the score that Botometer gave, we create an upper and lower bound and see if our score falls between these two values for each user score.

```Python
accounts['upper_b'] = accounts.bot_score_norm*1.10
accounts['lower_b'] = accounts.bot_score_norm*0.90

clean_df = accounts.loc[accounts.bot_score_norm.isna()==False]
within_10 = [clean_df.y_pred>=clean_df.lower_b]and[clean_df.y_pred<=clean_df.upper_b]
clean_df['within_10'] = within_10[0]

clean_df.within_10.sum()/len(clean_df)

Output:
0.8815399802566634
```

Our score based on a total of 1013 users that we tested against Botometer was of 88.2% accuracy. This means that for 88.2% of our scores where within 10% of the score given by Botometer.

## Conclusion and Future Work



We set out to create a model that can predict whether a twitter account is likely to be a bot or a genuine account. We were motivated to do this and test our performance against an already working machine learning model created by the Indiana University called Botometer.

We reached out to the researchers who have collected and labeled data related to numerous twitter accounts. We obtained over 6 Million observations corresponding to over 9,000 users. This data was labeled. Cleaning of the data was necesarry as the data was in raw format.

Additionally, we learned that feature engineering is important for this process. We generated various features from the raw data such as time data, if a tweet was a reply, the source of the tweet, and others. Some of these features proved to be useful, some proved to be not useful at all.

We measured the performance of our model against results obtained from Botometer itself. When comparing our model's performance against Botometer's performance, we see that 88.2% of our scores fall within 10% of the scores given by Botometer.

Next iterations of this project could include more feature engineering and tuning the neural network further. Botometer uses over 1000 features to create a score for a given user. We only used 25 features, so there is still a lot of opportunity to improve our model. We present a list of what other features could be done that might useful for prediction. It is important to test these as in our model, we discoverd that some engineered features actually lowered the accuracy score of our models (in particular features related to datetime variables).

A few features that could be engineered for later use in the models:

- Days a user account has been active.
- frequencey of tweets for users (per day frequency, per hour frequency, per minute frequency).
- Categorizing tweet text into topics as a predictor for bot or genuine.
