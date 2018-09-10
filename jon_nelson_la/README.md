Natural Language Processing -Subreddit analysis of Space X vs NASA

In this project I used Natural Language Processing techniques to analyze the text data within the posts from both the Space X and NASA subreddits. Using the public Reddit API I was able to webscrape both of these subreddiits to obtain the data. 

With this data I have built three different models that will analyze the text of these subreddits and make predictions on whether a subreddit post belongs to Space X or NASA. During EDA I used the CountVectorizer function to analyze both the total sum of the words being used in each post along with their average usage. I then use the TfidfVectorizer to give the words (features) a weight that would be used in my model to make the predictions. I was able to use a GridSearch on the following three models to search for the optimal hyperparameters to use for modeling:
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest

With these models I believe that I've identifed major influences in the text that can be used for targeted advertisements to the fans of each of these companies. My models can prove that there is a strong reference to the historical accomplishments in the NASA subreddit and a strong reference to the future of mankind on another planet in the Space X subreddit. 

In conclusion, for the Space X fans targetting the future visions and the innovative technologies being used to get mankind to fulfilling these visions will spark their passions for Space X. For NASA, targeting the scientific explorations of the past and what they believe to be the next steps in science for space will keep the NASA fans interested.   
