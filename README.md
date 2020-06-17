The python file contains the code and explanation, here is an outline and some main results.

Data from Kaggle: https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

Attributes:

              *is_sarcastic: 1 if the record is sarcastic otherwise 0
              *headline: the headline of the news article
              *article_link: link to the original news article

In data cleaning, we separately used 

              *original text
              *text after removing punctuations
              *text after removing stop words form package nltk
              *text after removing stop words form package nlp
              *text after normalizing(stemming and lemmatisation)
 
The RNN structure:

Input --> Embedding --> LSTM --> Global Max Pool 1D--> Dropout --> Dense --> Output

Conclusion

The accuracy of using original data is 86%, 6% higher than the result from prediction using cleaned data (=80%).

Why noisy data wins?

Especially when using a model like LSTM, where the model ONLY captures the semantic meanings of a word that are depended upon the information provided by previous texts, removing these critial stop words negates the original context and meaning of the sentenses. As a result, if we restore these removed stop words to the data, although the noise level increases, the accuracy rate is also significantly improved due to more information in the sentence providing ground for more precise judgement for the algorithm. Therefore, one possible way to improve our model is to put more consideration into determining the selection of words to remove.
