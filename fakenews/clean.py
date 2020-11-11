import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re     


w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
stopword = set(stopwords.words("english"))

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

def preprocess(text):
    '''
    This function preprocesses the input data frame. 
    It first converts everything to lower case, then removes stopwords like 'it', 'the', etc.
    Using regex, punctuation and digits are removed since they do not impact the classification.
    This process is done for title and text, and the resultant dataframe is returned after renaming the columns
    '''
    clean_data = []
    for (x, y) in zip(text['title'], text['text']): 
        item = []
        new_titles = x.lower() #lowercase
        new_titles = re.split('\W+', new_titles)
        #print(new_titles)
        new_titles = " ".join([word for word in new_titles if word not in stopword]).strip()
        #print(new_titles)
        new_titles = re.sub(r'[^\w\s\'\"]', '', new_titles) #remove everything except punctuation
        new_titles = re.sub(r'\d+','',new_titles)  #remove digits
        new_titles = lemmatize_text(new_titles) 
        item.append(new_titles)
        
        new_text = y.lower() #lowercase
        new_text = re.split('\W+', new_text)
        new_text = " ".join([word for word in new_text if word not in stopword]).strip()
        new_text = re.sub(r'[^\w\s\'\"]', '', new_text) #remove everything except punctuation
        new_text = re.sub(r'\d+','',new_text)  #remove digits
        new_text = lemmatize_text(new_text) 
        item.append(new_text)
        clean_data.append(item)
    
    clean_data = pd.DataFrame(clean_data)
    #clean_data['label'] = df['label']
    clean_data.rename(columns={0: 'title', 1: 'text'}, inplace=True)
    return clean_data

def main():
    df = pd.read_csv("../fakenews/fake_or_real_news.csv")
    clean_data = preprocess(df)
    clean_data.to_csv("./clean_input_data.csv")

if __name__ == '__main__':
    main()