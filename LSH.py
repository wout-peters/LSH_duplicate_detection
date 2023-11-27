# author: Wout Peters (531818)

import json
import numpy as np
import pandas as pd
import random


def create_df(data):
    '''
    Creates a pandas dataframe from JSON input with high-level info
    '''
    raw_df = pd.json_normalize(data).transpose()
    df = pd.DataFrame(columns = ['shop', 'url', 'modelID', 'title', 'features'])
    for i in range(len(raw_df)):
        for j in range(len(raw_df[0][i])):
            row = [raw_df[0][i][j]['shop'], 
                    raw_df[0][i][j]['url'], 
                    raw_df[0][i][j]['modelID'], 
                    raw_df[0][i][j]['title'], 
                    raw_df[0][i][j]['featuresMap']]
            df.loc[len(df.index)] = row
    return df

def normalize_text(titles):
    '''
    String normalisation: all lower case, change " to inch, remove punctuations
    '''
    normalized_titles = []
    
    for row in titles:
        row = row.lower()
        row = row.replace('"','inch')

        #remove punctuations
        row = row.replace('.','')
        row = row.replace('-','')
        row = row.replace('(','')
        row = row.replace(')','')
        row = row.replace('[','')
        row = row.replace(']','')
        row = row.replace(':','')
        row = row.replace('/','')
        row = row.replace(',','')

        normalized_titles.append(row)

    return normalized_titles

def is_model_word(input_string):
    has_letters = any(char.isalpha() for char in input_string)
    has_numbers = any(char.isdigit() for char in input_string)

    return has_letters and has_numbers

def product_representation(titles,TV_brands):
    '''
    Creates a list of product representations
    Includes model words from the titles, and TV brand names in the title
    '''
    representations = []

    for row in titles:
        tv_rep = []
        wordList = row.split()
        for string in wordList:
            if is_model_word(string):
                tv_rep.append(string)
            if string in TV_brands:
                tv_rep.append(string)
        representations.append(tv_rep)
    
    return representations


def main():
    with open("D:/Studie/23-24/Blok 2/Computer Science/Personal Assignment/TVs-all-merged (1)/TVs-all-merged.json", 'r') as read_file:
        data = json.load(read_file)
    read_file.close()

    df = create_df(data)

    #normalize titles
    df['title'] = normalize_text(df['title'])

    #for now, use manual list of TV brands. In the future, make webscraper.
    TV_brands = ["Bang & Olufsen","Continental Edison","Denver","Edenwood","Grundig","Haier","Hisense","Hitachi","HKC","Huawei","Insignia","JVC","LeEco","LG","Loewe","Medion","Metz","Motorola","OK.","OnePlus","Panasonic","Philips","RCA","Samsung","Sceptre","Sharp","Skyworth","Sony","TCL","Telefunken","Thomson","Toshiba","Vestel","Vizio","Xiaomi","Nokia","Engel","Nevir","TD Systems","Hyundai","Strong","Realme","Oppo","Metz Blue","Asus","Amazon","Cecotec","Nilait","Daewoo"]
    TV_brands = normalize_text(TV_brands)

    print(product_representation(df['title'],TV_brands))
    

if __name__ == "__main__":
    main()