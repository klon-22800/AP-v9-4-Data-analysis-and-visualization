import pandas as pd
import os
import nltk
import csv 
import re

import lab_2_1
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

def make_df(path: str)-> pd.DataFrame:
    lab_2_1.make_csv(path)
    num_list = []
    text_list = []
    r = open('paths.csv', 'r')
    reader = list(csv.reader(r))
    for item in reader: 
        f = open(item[0], 'r', encoding= 'utf-8')
        text = f.read()
        num_list.append(item[2])
        text_list.append(text)
    d = {'Num': num_list, 'Text': text_list}
    df1 = pd.DataFrame(data=d)
    df1 = df1.dropna()
    return df1

def text_update(text: str) -> list:
    text = re.sub(r"[^\w\s]", "", text)
    text = text.split()
    return text

def add_word_count(df:pd.DataFrame)-> None:
    df['word_count'] = 0
    for i in range(len(df)):
        print(df.iloc[i, 1])
        df.iloc[i, 2] = len(text_update(df.iloc[i, 1]))

    




df = make_df('dataset')
add_word_count(df)
print('----')
print(df)

# mystem = Mystem() 

# russian_stopwords = stopwords.words("russian")


# def preprocess_text(text):
#     tokens = mystem.lemmatize(text.lower())
#     tokens = [token for token in tokens if token not in russian_stopwords\
#               and token != " " \
#               and token.strip() not in punctuation]
    
#     text = " ".join(tokens)
    
#     return text
# text = 'Носки'
# tokens = mystem.lemmatize(text.lower())

# print(''.join(tokens))



