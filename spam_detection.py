import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


encoder = LabelEncoder()
ps = nltk.PorterStemmer()
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
tfidf = TfidfVectorizer(max_features=3000)
mnb = MultinomialNB()





#read dataset
df = pd.read_csv("./spam.csv", encoding='latin1')
df.sample(5)


#clean data
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
# df.sample(5)
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
# df.info()
df['label'] = encoder.fit_transform(df['label'])
# print(df.head(5))
# df.info()
# print(df.isnull().sum())
# print(df.duplicated().sum())
df= df.drop_duplicates(keep='first')
# print(df.duplicated().sum())
# print(df.shape)


#Analyze data
# print(df['label'].value_counts())
plt.pie(df['label'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()
#shows imbalance in data
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
print(df.head(5))
df[df['label'] == 0][['num_characters','num_words','num_sentences']].describe()
df[df['label'] == 1][['num_characters','num_words','num_sentences']].describe()
#visualize data
#histogram
plt.figure(figsize=(12,6))
sns.histplot(df[df['label'] == 0]['num_characters'])
sns.histplot(df[df['label'] == 1]['num_characters'],color='red')
plt.figure(figsize=(12,6))
sns.histplot(df[df['label'] == 0]['num_words'])
sns.histplot(df[df['label'] == 1]['num_words'],color='red')
#pairplot
# sns.pairplot(df,hue='label')
#heatmap
numeric_df = df[['label', 'num_characters', 'num_words', 'num_sentences']]
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()


#text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

#apply function
df['transformed_text'] = df['text'].apply(transform_text)
# print(df.head())
#word cloud
spam_wc = wc.generate(df[df['label'] == 1]['transformed_text'].str.cat(sep=" "))
ham_wc = wc.generate(df[df['label'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
print(df.head(5))
#most common words in spam
spam_corpus = []
for msg in df[df['label'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
# print(len(spam_corpus))
spam_common = pd.DataFrame(Counter(spam_corpus).most_common(30))
sns.barplot(x=spam_common[0], y=spam_common[1])
plt.xticks(rotation='vertical')
plt.show()
#most common words in ham
ham_corpus = []
for msg in df[df['label'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
# print(len(ham_corpus))
ham_common = pd.DataFrame(Counter(ham_corpus).most_common(30))
sns.barplot(x=ham_common[0], y=ham_common[1])
plt.xticks(rotation='vertical')
plt.show()


#Model building
X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape
y = df['label'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


#save model
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))