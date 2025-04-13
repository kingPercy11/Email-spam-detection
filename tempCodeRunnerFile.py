
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))