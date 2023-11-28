# author: Wout Peters (531818)

import json
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import itertools

from sklearn.metrics import jaccard_score

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

def normalize_list(textList):
    '''
    String normalisation: all lower case, inch and hertz expressions, remove punctuations
    '''
    normalized_titles = []
    
    for row in textList:
        row = row.lower()
        row = row.replace('"','inch')
        row = row.replace('inches','inch')
        row = row.replace('hertz','hz')

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
        row = row.replace('°','')
        row = row.replace('º','')
        row = row.replace('>','')
        row = row.replace('<','')

        normalized_titles.append(row)

    return normalized_titles

def normalize_dictionary(textDict):
    '''
    String normalisation: all lower case, inch and hertz expressions, remove punctuations
    '''
    normalized_dict = {}

    for key,val in textDict.items():
        val = val.lower()
        val = val.replace('"','inch')
        val = val.replace('inches','inch')
        val = val.replace('hertz','hz')

        #remove punctuations
        val = val.replace('.','')
        val = val.replace('-','')
        val = val.replace('(','')
        val = val.replace(')','')
        val = val.replace('[','')
        val = val.replace(']','')
        val = val.replace(':','')
        val = val.replace('/','')
        val = val.replace(',','')
        val = val.replace('°','')
        val = val.replace('º','')
        val = val.replace('>','')
        val = val.replace('<','')

        normalized_dict[key] = val
    
    return normalized_dict

def is_model_word(input_string):
    has_letters = any(char.isalpha() for char in input_string)
    has_numbers = any(char.isdigit() for char in input_string)

    return has_letters and has_numbers

def product_representation(titles,TV_brands,features):
    '''
    Creates a list of product representations
    Includes model words from the titles, and TV brand names in the title, and model words in the values
    '''
    representations = []

    for TV in range(len(titles)):
        TV_rep = []
        title_list = titles[TV].split()
        for string in title_list:
            if is_model_word(string):
                TV_rep.append(string)
            if string in TV_brands:
                TV_rep.append(string)
        for key,val in features[TV].items():
            val_list = val.split()
            for word in val_list:
                if is_model_word(word):
                    TV_rep.append(word)        
        representations.append(TV_rep)
    
    return representations

def binary_matrix(product_representations):
    '''
    Create binary matrix for product representations
    Entries are binary vectors for each TV with 1 if token present, 0 else
    Returns binary numpy matrix and the list of tokens in correct order
    '''
    #use unique tokens
    flat_list = [item for sublist in product_representations for item in sublist]
    tokens = list(set(flat_list))
    
    #rows are elements of token set, columns are TVs
    binary_matrix = np.zeros((len(tokens),len(product_representations)), dtype=int)
    for tv in range(len(product_representations)):
        for token in range(len(tokens)):
            if tokens[token] in product_representations[tv]:
                binary_matrix[token][tv] = 1

    return binary_matrix, tokens

def find_next_prime(number):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    next_number = number + 1
    while not is_prime(next_number):
        next_number += 1
    return next_number    
        
def randomCoefficients(numHashes):
    randList = []
    for i in range(numHashes):
        randIndex = random.randint(0,numHashes)
        while randIndex in randList:
            randIndex = random.randint(0,numHashes)
        randList.append(randIndex)
    return randList

def minhash(numHashFunc, binary_matrix):
    '''
    Performs min-hashing using the implementation in the lecture
    The hash functions are of the form ax + b % P, where x is the input,
    a and b random coefficients, and P is the first prime number larger than
    the number of tokens
    '''
    #Create numHashFunc random (and unique) hash functions
    P = find_next_prime(binary_matrix.shape[0])
    a = randomCoefficients(numHashFunc)
    b = randomCoefficients(numHashFunc)
    hashFunc = lambda a,b,P,x: (a * x + b) % P
    hash_values = np.zeros(numHashFunc)
    
    #Initialize M with infinity
    M = np.full((numHashFunc,binary_matrix.shape[1]), np.inf)

    #iterate over ROWS of TVs
    for row_idx, row in enumerate(binary_matrix):
        for hash_idx in range(numHashFunc):
            hash_values[hash_idx] = hashFunc(a[hash_idx],b[hash_idx],P,row_idx)
        for col_idx, col in enumerate(binary_matrix.T):
            if col[row_idx] == 1:
                for idx in range(numHashFunc):
                    if hash_values[idx] < M[idx,col_idx]:
                        M[idx,col_idx] = hash_values[idx]
    
    return M

def initialize_hash_bucket(bands, nBuckets):
    '''
    Initializes the hash matrix, with nBuckets empty lists for each band
    '''
    hash_buckets = []
    for band in range(bands):
        hash_buckets.append([[] for i in range(nBuckets)])
    return hash_buckets

def get_b(numSig, threshold):
        """
        Calculates the number of bands needed such that approximately it holds that
        (1/b)^(1/r) = threshold and b*r = numSig
        """
        n = numSig
        t = threshold
        def get_bandwidth(n, t):
            best = n, 1
            minerr  = float("inf")
            for r in range(1, n + 1):
                try:
                    b = 1. / (t ** r)
                except:             # Divide by zero, your signature is huge
                    return best
                err = abs(n - (b * r))
                if err < minerr:
                    best = r
                    minerr = err
            return best
        r = get_bandwidth(n, t)
        b = int(n / r)
        return b

def LSH(signature_matrix, thres):
    '''
    Perform LSH on the signature matrix, with number of bands b,
    rows per band r, and threshold t.
    '''
    # Take nBuckets very large such that columns only hashed to same bucket when identical
    nBuckets = find_next_prime(100*signature_matrix.shape[0])
    nBands = get_b(signature_matrix.shape[0],thres)
    hash_buckets = initialize_hash_bucket(nBands, nBuckets)
    numHashFunc, numTV = signature_matrix.shape
    candidates = set()
    rowsPerBand = math.floor(numHashFunc/nBands)
    rowsLeft = numHashFunc % nBands

    for b in range(nBands):
        #put rest of the rows in the final band
        if b == nBands - 1:
            band = signature_matrix[b*rowsPerBand:(b+1)*rowsPerBand + rowsLeft,:]
            # As hash function, take a random linear transformation of the signatures and take modulo of nBuckets
            hash_function = np.random.randint(1, numHashFunc, size = rowsPerBand + rowsLeft)
        else:
            band = signature_matrix[b*rowsPerBand:(b+1)*rowsPerBand,:]
            hash_function = np.random.randint(1, numHashFunc, size = rowsPerBand)
        for col in range(numTV):
            key = int(hash_function.dot(band[:,col]) % nBuckets)
            hash_buckets[b][key].append(col)

        #for band b, check each bucket for duplicates
        for bucket in hash_buckets[b]:
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    if pair not in candidates:
                        A = signature_matrix[:,pair[0]]
                        B = signature_matrix[:,pair[1]]
                        similarity = jaccard_score(A,B,average='macro')
                        #if similarity > thres, add to candidates  
                        if similarity >= thres:
                            candidates.add(pair)            
    
    return list(candidates)        

#def similarity(candidate_pairs, bin_matrix):
#    for pair in candidate_pairs:
#        TV_1 = bin_matrix[:,pair[0]]
#        TV_2 = bin_matrix[:,pair[1]]
#        similarity = np.linalg.norm(TV_1-TV_2)
#        print(similarity)

def main():
    print("Loading data...")
    
    with open("D:/Studie/23-24/Blok 2/Computer Science/Personal Assignment/TVs-all-merged (1)/TVs-all-merged.json", 'r') as read_file:
        data = json.load(read_file)
    read_file.close()
    df = create_df(data)

    #toy data
    
    #TV1 = "philips 1080p 530hz 30inch"
    #TV1 = "samsung 4k 100hz 50inch 1080p 30inch 530hz philips"
    #TV2 = "samsung 4k 100hz 50inch 1080p 30inch 530hz"
    #data = {'title': [TV1,TV2]}
    #df = pd.DataFrame(data)
    
    print("Creating product representations...")
    
    #Product representations are model words in the title, model words in the values, and brands in the titles
    df['title'] = normalize_list(df['title'])
    TV_brands_1 = ["Bang & Olufsen","Continental Edison","Denver","Edenwood","Grundig","Haier","Hisense","Hitachi","HKC","Huawei","Insignia","JVC","LeEco","LG","Loewe","Medion","Metz","Motorola","OK.","OnePlus","Panasonic","Philips","RCA","Samsung","Sceptre","Sharp","Skyworth","Sony","TCL","Telefunken","Thomson","Toshiba","Vestel","Vizio","Xiaomi","Nokia","Engel","Nevir","TD Systems","Hyundai","Strong","Realme","Oppo","Metz Blue","Asus","Amazon","Cecotec","Nilait","Daewoo","insignia","nec","supersonic","viewsonic","Element","Sylvania","Proscan","Onn","Vankyo","Blaupunkt","Coby","Kogan","RCA","Polaroid","Westinghouse","Seiki","Insignia","Funai","Sansui","Dynex","naxa"]
    TV_brands_2 = ['Philips', 'Samsung', 'Sharp', 'Toshiba', 'Hisense', 'Sony', 'LG', 'RCA', 'Panasonic', 'VIZIO', 'Naxa', 'Coby', 'Vizio', 'Avue', 'Insignia', 'SunBriteTV', 'Magnavox', 'Sanyo', 'JVC', 'Haier', 'Venturer', 'Westinghouse', 'Sansui', 'Pyle', 'NEC', 'Sceptre', 'ViewSonic', 'Mitsubishi', 'SuperSonic', 'Curtisyoung', 'Vizio', 'TCL', 'Sansui', 'Seiki', 'Dynex']
    TV_brands = normalize_list(list(set(TV_brands_1) | set(TV_brands_2)))
    df['features'] = df['features'].apply(lambda x: normalize_dictionary(x))

    product_representations = product_representation(df['title'],TV_brands,df['features'])

    #print(product_representations)

    print("Creating binary matrix...")

    bin_matrix, tokens = binary_matrix(product_representations)
    
    print("Min-hashing...")

    signatures = minhash(100,bin_matrix)

    print("LSH...")

    candidate_pairs = LSH(signatures, 0.8)

    print(candidate_pairs)
    

if __name__ == "__main__":
    main()