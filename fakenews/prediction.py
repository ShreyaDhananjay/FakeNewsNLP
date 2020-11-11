import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
def predict(example):
    df = pd.read_csv("fakenews/clean_input_data.csv")
    y = df.label 
    df = df.drop("label", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=52)
    
    #TF-IDF Vectorizer for news text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['text'].values.astype('U')) 
    #tfidf_test = tfidf_vectorizer.transform(X_test['text'])
    tfidf_test = tfidf_vectorizer.transform(X_test['text'].values.astype('U'))
    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)                      
    pred = clf.predict(tfidf_test)    
    if(pred[0] == 'REAL'):
        return pred[0]
    
    #count vectorizer for text
    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train['text'].values.astype('U'))                  
    print(example.head())
    count_test = count_vectorizer.transform(example['text'].values.astype('U'))
    clf = MultinomialNB()
    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    if(pred[0] == 'REAL'):
        return pred[0]
    
    #count vectorizer for title
    count_vectorizer = CountVectorizer(stop_words='english')
    #fit transform for training set to learn the vocabulary dictionary and transform test set
    count_train = count_vectorizer.fit_transform(X_train['title'].values.astype('U'))        
    count_test = count_vectorizer.transform(example['title'])#.values.astype('U'))          
    clf = MultinomialNB()
    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    if(pred[0] == 'REAL'):
        return pred[0]
    
    #tfidf vectorizer for title
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['title'].values.astype('U')) 
    #tfidf_test = tfidf_vectorizer.transform(X_test['title'])
    tfidf_test = tfidf_vectorizer.transform(example['title'].values.astype('U'))
    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)                       # Fit Naive Bayes classifier according to X, y
    pred = clf.predict(tfidf_test)  
    return pred[0]


#def countVecTitle(example):
    '''
    CountVectorizer for the news headlines
    '''
    #global X_train, X_test, y_train, y_test
    
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for Count with title:   %0.3f" % score)
    #return count_vectorizer, clf, pred
    #return pred[0]


#def countVecText(example):
    '''
    CountVectorizer for the news text
    '''
    


#def tfidfVecTitle(example):
    
                       # Perform classification on an array of test vectors X.
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for TF-IDF with title:   %0.3f" % score)
    #return pred[0]
    
    
#def tfidfVecText(example):
                     # Perform classification on an array of test vectors X.
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for TF-IDF with text:   %0.3f" % score)
    #return tfidf_vectorizer, clf, pred
    #return pred[0]

#def main():
#    pass