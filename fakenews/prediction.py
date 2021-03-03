import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
def predict(example, choice):
    df = pd.read_csv("fakenews/clean_input_data.csv")
    y = df.label 
    df = df.drop("label", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=52)
    if(choice['method'] == 'TFIDF'):
        if(choice['cl_on'] == 'Title'):
            return tfidfVecTitle(example, y_train, X_train)
        else:
            return tfidfVecText(example, y_train, X_train)
    else:
        if(choice['cl_on'] == 'Title'):
            return countVecTitle(example, y_train, X_train)
        else:
            return countVecText(example, y_train, X_train)


def countVecTitle(example, y_train, X_train):
    '''
    CountVectorizer for the news headlines
    '''
    print("in 1")
    count_vectorizer = CountVectorizer(stop_words='english')
    #fit transform for training set to learn the vocabulary dictionary and transform test set
    count_train = count_vectorizer.fit_transform(X_train['title'].values.astype('U'))        
    count_test = count_vectorizer.transform(example['title'])#.values.astype('U'))          
    clf = MultinomialNB()
    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    return pred[0]
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for Count with title:   %0.3f" % score)



def countVecText(example, y_train, X_train):
    '''
    CountVectorizer for the news text
    '''
    print("in 2")
    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train['text'].values.astype('U'))                  
    print(example.head())
    count_test = count_vectorizer.transform(example['text'].values.astype('U'))
    clf = MultinomialNB()
    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    return pred[0]
    


def tfidfVecText(example, y_train, X_train):
    '''
    TfidfVectorizer for the news text
    '''
    print("in 3")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=8000)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['text'].values.astype('U')) 
    tfidf_test = tfidf_vectorizer.transform(example['text'].values.astype('U'))
    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)                      
    pred = clf.predict(tfidf_test)  
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for TF-IDF with title:   %0.3f" % score)
    return pred[0]
    
    
def tfidfVecTitle(example, y_train, X_train):
    '''
    TfidfVectorizer for the news title
    '''
    print("in 4")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['title'].values.astype('U')) 
    tfidf_test = tfidf_vectorizer.transform(example['title'].values.astype('U'))
    clf = MultinomialNB() 
    clf.fit(tfidf_train, y_train)                       
    pred = clf.predict(tfidf_test)  
    return pred[0]
    #score = metrics.accuracy_score(y_test, pred)
    #print("Accuracy for TF-IDF with text:   %0.3f" % score)
    

