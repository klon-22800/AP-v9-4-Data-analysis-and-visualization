import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import lab_2_1
import numpy as np
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from collections import Counter


mystem = Mystem()

russian_stopwords = stopwords.words("russian")


def make_df(path: str) -> pd.DataFrame:
    lab_2_1.make_csv(path)
    num_list = []
    text_list = []
    r = open('paths.csv', 'r')
    reader = list(csv.reader(r))
    for item in reader:
        f = open(item[0], 'r', encoding='utf-8')
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


def add_word_count(df: pd.DataFrame) -> None:
    df['word_count'] = 0
    for i in range(len(df)):
        df.iloc[i, 2] = len(text_update(df.iloc[i, 1]))


def stats_by_word_count(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Num", "word_count"]].groupby("Num").mean()


def sort_by_word_count(df: pd.DataFrame, max_count: int) -> pd.DataFrame:
    new_df = df.loc[df['word_count'] <= max_count]
    return new_df


def sort_by_num(df: pd.DataFrame, num: str) -> pd.DataFrame:
    new_df = df.loc[df['Num'] == num]
    return new_df


def preprocess_text(text:str)->list:
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords
              and token != " "
              and token.strip() not in punctuation
              and token.isalpha()]

    text = " ".join(tokens)
    return tokens

def group_by_num(df:pd.DataFrame):
    num = []
    max = []
    min = []
    mean = []
    groupped_d = {'num':num, 'max':max, 'min':min, 'mean':mean}
    for i in range(1, 6):
        num.append(str(i))
        max.append((df.loc[df['Num'] == str(i)][['word_count']].max()).loc['word_count'])
        min.append((df.loc[df['Num'] == str(i)][['word_count']].min()).loc['word_count'])
        mean.append((df.loc[df['Num'] == str(i)][['word_count']].mean()).loc['word_count'])
    groupped_df = pd.DataFrame(data = groupped_d)
    return groupped_df
    
def make_histogram(df:pd.DataFrame, num:str)-> list:
    res = []
    lenght = len(df.loc[df['Num'] == num]['Text'])
    for i in range(20):
        text = df.loc[df['Num'] == num]['Text'].iloc[i]
        text = preprocess_text(text)
        res += text
        print(i)
    res = dict(Counter(res))
    res = sorted(res.items(), key=lambda item: item[1], reverse = True)
    res = res[0:20]
    return res

def graph_build(hist_list:list)->None:
    words = []
    count = []
    for i in range(len(hist_list)):
        words.append(hist_list[i][0])
        count.append(hist_list[i][1])

    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(words))

    ax.barh(y_pos, count, align='center')
    ax.set_yticks(y_pos, labels=words)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Word count')
    ax.set_title('The most popular words')

    plt.show()


df = make_df('dataset')
add_word_count(df)
print('----')
print(df)
print('----')
print(stats_by_word_count(df))
print(sort_by_word_count(df, 100))
print(sort_by_num(df, '5'))
print(preprocess_text(''))
print(group_by_num(df))
hist = make_histogram(df, "5")

graph_build(hist)

