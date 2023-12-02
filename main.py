import csv
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk

import lab_2_1

from typing import List, Dict

from nltk.corpus import stopwords
from pymystem3 import Mystem
from collections import Counter

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


def make_df(path: str) -> pd.DataFrame:
    """Function gets path to dataset and make DataFrame with num mark and text field 

    Args:
        path (str): path to dataset

    Returns:
        pd.DataFrame: ready-made dataset
    """
    if 'paths.csv' not in os.listdir():
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
    d = {'num': num_list, 'text': text_list}
    df1 = pd.DataFrame(data=d)
    df1 = df1.dropna()
    return df1


def text_update(text: str) -> List[str]:
    """Function remove from text punctuation marks and split it

    Args:
        text (str): text for update

    Returns:
        List[str]: List with words from text
    """
    text = re.sub(r"[^\w\s]", "", text)
    text = text.split()
    return text


def add_word_count(df: pd.DataFrame) -> None:
    """Function add to DataFrame column with word count information 

    Args:
        df (pd.DataFrame): DataFrame to edit
    """
    df['word_count'] = 0
    for i in range(len(df)):
        df.iloc[i, 2] = len(text_update(df.iloc[i, 1]))


def stats_by_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Function make DataFrame with information about average number of word to every mark

    Args:
        df (pd.DataFrame): DataFrame with text information

    Returns:
        pd.DataFrame: DataFrame num mark: word count
    """
    return df[["num", "word_count"]].groupby("num").mean()


def sort_by_word_count(df: pd.DataFrame, max_count: int) -> pd.DataFrame:
    """The function creates a dataframe by removing lines with a word count exceeding max count

    Args:
        df (pd.DataFrame): DataFrame with text information
        max_count (int): max count of words count

    Returns:
        pd.DataFrame: DataFrame with a word count exceeding max count
    """
    new_df = df.loc[df['word_count'] <= max_count]
    return new_df


def sort_by_num(df: pd.DataFrame, num: str) -> pd.DataFrame:
    """Function sorts DataFrame by the num mark

    Args:
        df (pd.DataFrame): DataFrame to sort
        num (str): num mark 

    Returns:
        pd.DataFrame: sorted DataFrame
    """
    new_df = df.loc[df['num'] == num]
    return new_df


def preprocess_text(text: str) -> List[str]:
    """Function gets text, lemmatize them and removes stopwords

    Args:
        text (str): text for preprocess 

    Returns:
        List[str]: preprocessed text
    """
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords]
    text = " ".join(tokens)
    return tokens


def preprocess_text_only_A(text: str) -> List[str]:
    """Function gets text, lemmatize them and removes all word without adjective and adverb

    Args:
        text (str): text for preprocess

    Returns:
        List[str]: preprocessed text
    """
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords]
    text = " ".join(tokens)
    words = nltk.word_tokenize(text)
    functors_pos = {'A=m', 'ADV'}
    res = [word for word, pos in nltk.pos_tag(words, lang='rus')
           if pos in functors_pos]
    return res


def group_by_num(df: pd.DataFrame) -> pd.DataFrame:
    """The function groups the dataframe by class label and 
    calculates the minimum and maximum number of words in the text

    Args:
        df (pd.DataFrame): DataFrame with text information

    Returns:
        pd.DataFrame: groupped DataFrame
    """
    num = []
    max = []
    min = []
    mean = []
    groupped_d = {'num': num, 'max': max, 'min': min, 'mean': mean}
    for i in range(1, 6):
        num.append(str(i))
        max.append((df.loc[df['num'] == str(i)]
                   [['word_count']].max()).loc['word_count'])
        min.append((df.loc[df['num'] == str(i)]
                   [['word_count']].min()).loc['word_count'])
        mean.append((df.loc[df['num'] == str(i)]
                    [['word_count']].mean()).loc['word_count'])
    groupped_df = pd.DataFrame(data=groupped_d)
    return groupped_df


def make_histogram(df: pd.DataFrame, num: str) -> Dict[str, int]:
    """Function make dictionary word: count by num marks for making histogram
    Args:
        df (pd.DataFrame): DataFrame with text information
        num (str): num mark 

    Returns:
        Dict[str, int]: dictionary word: count in num marks text
    """
    res = []
    lenght = len(df.loc[df['num'] == num]['text'])
    for i in range(lenght):
        text = df.loc[df['num'] == num]['text'].iloc[i]
        text = preprocess_text_only_A(text)
        res += text
        print(i)
    res = dict(Counter(res))
    res = sorted(res.items(), key=lambda item: item[1], reverse=True)
    res = res[0:30]
    return res


def graph_build(hist_list: Dict[str, int]) -> None:
    """Function make plot with matplotlib x-axe is word count y-axe is word

    Args:
        hist_list (Dict[str, int]): dictionary word: num mark
    """
    words = []
    count = []
    for i in range(len(hist_list)):
        words.append(hist_list[i][0])
        count.append(hist_list[i][1])

    fig, ax = plt.subplots()

    y_pos = np.arange(len(words))

    ax.barh(y_pos, count, align='center')
    ax.set_yticks(y_pos, labels=words)
    ax.invert_yaxis()
    ax.set_xlabel('Word count')
    ax.set_title('The most popular words')

    plt.show()


if __name__ == "__main__":
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
    hist = make_histogram(df, "1")

    graph_build(hist)
