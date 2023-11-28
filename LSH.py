# author: Wout Peters (531818)

import json
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import itertools

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
    String normalisation: all lower case, inch and hertz expressions, remove punctuations
    '''
    normalized_titles = []
    
    for row in titles:
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

def initialize_array_bucket(bands, nBuckets):
    '''
    Initializes the hash matrix, with nBuckets empty lists for each band
    '''
    array_buckets = []
    for band in range(bands):
        array_buckets.append([[] for i in range(nBuckets)])
    return array_buckets

#def apply_LSH_technique(SIG = np.matrix(signatures).transpose() , t = threshold, bands=bands, rows=rows):
#        array_buckets = initialize_array_bucket(bands)
#        candidates = {}
#        i = 0
#        for b in range(bands):
#            buckets = array_buckets[b]        
#            band = SIG[i:i+rows,:]
#            for col in range(band.shape[1]):
#                key = int(sum(band[:,col]) % len(buckets))
#                buckets[key].append(col)
#            i = i+rows
#        
#            for item in buckets:
#                if len(item) > 1:
#                    pair = (item[0], item[1])
#                    if pair not in candidates:
#                        A = SIG[:,item[0]]
#                        B = SIG[:,item[1]]
#                        similarity = jaccard_score(A,B, average='macro')
#                        if similarity >= t:
#                            candidates[pair] = similarity
#    
#        sort = sorted(candidates.items(), reverse=True)
#        return candidates,sort

def LSH(signature_matrix, thres, nBands):
    '''
    Perform LSH on the signature matrix, with number of bands b,
    rows per band r, and threshold t.
    '''
    # NUMBER OF BUCKETS DEFINED CORRECTLY?
    nBuckets = find_next_prime(signature_matrix.shape[0])
    array_buckets = initialize_array_bucket(nBands, nBuckets)
    numHashFunc, numTV = signature_matrix.shape
    candidates = {}
    rowsPerBand = math.floor(numHashFunc/nBands)
    rowsLeft = numHashFunc % nBands

    # CHECK IF THIS IS OKAY FOR THE LAST BAND
    for b in range(nBands):
        band = signature_matrix[b*rowsPerBand:(b+1)*rowsPerBand,:]
        for col in range(numTV):
            key = int(sum(band[:,col]) % nBuckets)
            array_buckets[b][key].append(col)

    # TO DO: DIFFERENT HASH FUNCTIONS
    # It generates two hash functions (b)!!, two vectors of size numHashFunc that will be multiplied with
    # the signature matrix
    # Generate random hash functions
    hash_functions = [np.random.randint(1, 1000, size=numHashFunc) for _ in range(b)]

    # Initialize hash tables
    # Creates two empty dictionaries
    hash_tables = [defaultdict(list) for _ in range(b)]

    # Hash signatures into buckets
    for col_idx in range(numTV):
        for i, hash_function in enumerate(hash_functions):
            hash_value = hash_function.dot(signature_matrix[:, col_idx]) % b
            hash_tables[i][hash_value].append(col_idx)

    # Identify candidate pairs from hash tables
    candidate_pairs = set()
    for table in hash_tables:
        for bucket, columns in table.items():
            if len(columns) > 1:
                for pair in itertools.combinations(columns, 2):
                    # Check similarity using the Jaccard similarity
                    similarity = np.sum(signature_matrix[:, pair[0]] == signature_matrix[:, pair[1]]) / numHashFunc
                    if similarity >= t:
                        candidate_pairs.add(pair)

    return list(candidate_pairs)


def main():
    #print("Loading data...")
    #
    #with open("D:/Studie/23-24/Blok 2/Computer Science/Personal Assignment/TVs-all-merged (1)/TVs-all-merged.json", 'r') as read_file:
    #    data = json.load(read_file)
    #read_file.close()
    #df = create_df(data)

    #toy data
    
    TV1 = "philips 1080p 530hz 30inch"
    #TV1 = "samsung 4k 100hz 1080p"
    TV2 = "samsung 4k 100hz 50inch"
    data = {'title': [TV1,TV2]}
    df = pd.DataFrame(data)
    
    print("Creating product representations...")
    
    df['title'] = normalize_text(df['title'])
    #for now, use manual list of TV brands. In the future, make webscraper.
    TV_brands_1 = ["Bang & Olufsen","Continental Edison","Denver","Edenwood","Grundig","Haier","Hisense","Hitachi","HKC","Huawei","Insignia","JVC","LeEco","LG","Loewe","Medion","Metz","Motorola","OK.","OnePlus","Panasonic","Philips","RCA","Samsung","Sceptre","Sharp","Skyworth","Sony","TCL","Telefunken","Thomson","Toshiba","Vestel","Vizio","Xiaomi","Nokia","Engel","Nevir","TD Systems","Hyundai","Strong","Realme","Oppo","Metz Blue","Asus","Amazon","Cecotec","Nilait","Daewoo","insignia","nec","supersonic","viewsonic","Element","Sylvania","Proscan","Onn","Vankyo","Blaupunkt","Coby","Kogan","RCA","Polaroid","Westinghouse","Seiki","Insignia","Funai","Sansui","Dynex","naxa"]
    TV_brands_2 = ['Philips', 'Samsung', 'Sharp', 'Toshiba', 'Hisense', 'Sony', 'LG', 'RCA', 'Panasonic', 'VIZIO', 'Naxa', 'Coby', 'Vizio', 'Avue', 'Insignia', 'SunBriteTV', 'Magnavox', 'Sanyo', 'JVC', 'Haier', 'Venturer', 'Westinghouse', 'Sansui', 'Pyle', 'NEC', 'Sceptre', 'ViewSonic', 'Mitsubishi', 'SuperSonic', 'Curtisyoung', 'Vizio', 'TCL', 'Sansui', 'Seiki', 'Dynex']
    TV_brands = normalize_text(list(set(TV_brands_1) | set(TV_brands_2)))
    product_representations = product_representation(df['title'],TV_brands)

    print("Creating binary matrix...")

    bin_matrix, tokens = binary_matrix(product_representations)

    print("Min-hashing...")

    signatures = minhash(9,bin_matrix)

    print("LSH...")

    LSH(signatures, 0.8, 2)


    

if __name__ == "__main__":
    main()