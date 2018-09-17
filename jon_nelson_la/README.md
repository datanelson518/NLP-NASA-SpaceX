## Classifying subreddit posts using Natural Language Processing (NLP)

#### Problem Statement

Is there a significant difference between what NASA and Space X are discussing that can be targeted to advertise to the fans of each corporation?

#### Description

Using NLP on the titles of the subreddits of Space X and NASA I will fit classification models that can predict which specific posts came from either Space X or NASA. With this model we can then infer what topics are being discussed within each subreddit and if possible identify how to specifically advertise to the fans of Space X or to the fans of NASA.

With this data I have built three different models that will analyze the text of these subreddits and make predictions on whether a subreddit post belongs to Space X or NASA. During EDA I used the CountVectorizer function to analyze both the total sum of the words being used in each post along with their average usage. I then use the TfidfVectorizer to give the words (features) a weight that would be used in my model to make the predictions. I was able to use a GridSearch on the following three models to search for the optimal hyperparameters to use for modeling:
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest

### Logistic Regression

My first choice in models will be Logistic Regression. The reason I will use this model is because the model needs to make predictions on a discrete variable and in a binary output 1 = Space X and 0 = NASA.

#### Gridsearch with Logistic Regression

GridSearchCV is the modeling technique that searches for the optimal hyperparameters provided during the instantiating of a model. Using its built in cross validation it can search over the grid of the provided hyperparameters to evaluate the performance of each and then use the parameters it found to be the best when making the predictions.

#### Logistic Regression

This is a modeling technique that will predict the probability of an event occuring given some previous data about the variable it's predicting. This is a modeling technique used to predict discreate variables and in this specific case we will be predicting a binary outcome of whether a post is from Space X (1) or NASA (0).

#### Interpretation of Logistic Regression model scores

The logistic regression model is explaining 91.5% of the variance within the model. The other main thing to note is that our testing accuracy is scoring a little lower than our training accuracy score which means the model is a little overfit. The difference between the training and testing score is extremely low which is good and means we are not overfit by very much.

### KNeighbors Classifier

My second choice in models will be KNeighborsClassifier. The reason I will use this model is because the model uses the data points that are closest together to assign the weights to the features in model to then predict our binary output 1 = Space X and 0 = NASA.

#### Gridsearch with K-Nearest Neighbors

GridSearchCV is the modeling technique that searches for the optimal hyperparameters provided during the instantiating of a model. Using its built in cross validation it can search over the grid of the provided hyperparameters to evaluate the performance of each and then use the parameters it found to be the best when making the predictions.

#### K-Nearest Neighbors

This is a modeling technique that will assign weight to the neighbors (features) that are closest to the higher weighted neighbors. The TfidfVectorizer will feed to the K-Nearest Neighbors model to assign the weights. We'll specifically be using K-NN classification model to predict the discreate variables and in this specific case we will be predicting a binary outcome of whether a post is from Space X (1) or NASA (0).

#### Interpretation of KNeighbors Classifier model scores

The KNN model is explaining 89% of the variance within the model. The other main thing to note is that our testing accuracy is scoring quite a bit lower than our training accuracy score which means the model is definitely overfit. The difference between the training and testing scores is larger than the difference from the Logistic Regression model so I believe that model to still be the better of the models thus far.

### Random Forest

My third choice in models will be Random Forest. The reason I will use this model is because I want to introduce randomness through decision trees into the model to then predict our binary output 1 = Space X and 0 = NASA.

#### GridSearch with Random Forest

GridSearchCV is the modeling technique that searches for the optimal hyperparameters provided during the instantiating of a model. Using its built in cross validation it can search over the grid of the provided hyperparameters to evaluate the performance of each and then use the parameters it found to be the best when making the predictions.

#### Random Forest

This is an ensemble modeling technique that will construct a bunch of random decision trees based on the features and make predictions based on what the model learns about the features the trees are outputting most often. We'll specifically be using Random Forest to predict a discrete variable and in this specific case we will be predicting a binary outcome of whether a post is from Space X (1) or NASA (0).

#### Interpretation of Random Forest model scores

The Random Forest model is explaining 92.6% of the variance within the model. The other main thing to note is that our testing accuracy is scoring lower than our training accuracy score which means the model is overfit.

This model is explaining more variance than the KNN model and Logistic Regression model and is arguably the best model that I've built for this data. However, there is an argument to be made that the Logistic Regression model is slightly better because of the smaller difference in training and testing scores than any of my other models.

## Conclusions
With these models I believe that I've identified major influences in the text that can be used for targeted advertisements to the fans of each of these companies. My models can prove that there is a strong reference to the historical accomplishments in the NASA subreddit and a strong reference to the future of mankind on another planet in the Space X subreddit.

In conclusion, for the Space X fans targeting the future visions and the innovative technologies being used to get mankind to fulfilling these visions will spark their passions for Space X. For NASA, targeting the scientific explorations of the past and what they believe to be the next steps in science for space will keep the NASA fans interested.   
