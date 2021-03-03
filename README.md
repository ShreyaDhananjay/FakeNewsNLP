# FakeNewsNLP
A simple Flask application to check if the given news article is real or fake. 
Uses the LIAR dataset with fewer attributes.

### NLP methods used: 
* Stopword removal
* Lemmatization
* Bag of Words
* TF-IDF

### Classifier used: Multinomial Naive Bayes

### To run this application:
1. Extract the dataset 'fake_or_real_news.csv' from the zip file
2. Run [clean.py](https://github.com/ShreyaDhananjay/FakeNewsNLP/blob/main/fakenews/clean.py) in the fakenews module to preprocess the dataset
3. Run the application by executing [run.py](https://github.com/ShreyaDhananjay/FakeNewsNLP/blob/main/run.py)
