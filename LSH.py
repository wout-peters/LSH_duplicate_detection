# author: Wout Peters (531818)

import json
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from collections import Counter
import string
import re
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

def product_representation(titles,TV_brands):
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

def candidate_evaluation(lsh_candidates, df):
    true_duplicates = all_pairs(df) 
    lsh_candidates = [set(tuple_a) for tuple_a in lsh_candidates]
    result = [tuple_a for tuple_a in lsh_candidates if any(set(tuple_a).issubset(set_b) for set_b in true_duplicates)]

    def generate_2_combinations(input_set):
        return list(itertools.combinations(input_set, 2))
    combinations_list = [generate_2_combinations(set_b) for set_b in true_duplicates]
    flattened_combinations = [combination for sublist in combinations_list for combination in sublist]

    number_comparisons = len(lsh_candidates)
    true_duplicates_found = len(result)
    true_num_duplicates = len(flattened_combinations)

    PQ = true_duplicates_found / number_comparisons
    PC = true_duplicates_found / true_num_duplicates
    F1_star = (2*PQ*PC) / (PQ + PC)

    print("Amount of comparisons: ", number_comparisons)
    print("Amount of true duplicates: ",true_num_duplicates)
    print("Amount of true duplicates found: ",true_duplicates_found)
    print('Pair Quality (PQ): ', PQ)
    print('Pair Completeness (PC): ', PC)
    print('The F1_star score is: ', F1_star)

    return

def all_pairs(df):
    indices_dict = {}
    # Iterate over rows and populate the dictionary
    for index, row in df.iterrows():
        model_id = row['modelID']
        model_id = model_id.lower()
        model_id = model_id.replace('.','')
        model_id = model_id.replace('-','')
        model_id = model_id.replace('(','')
        model_id = model_id.replace(')','')
        model_id = model_id.replace('[','')
        model_id = model_id.replace(']','')
        model_id = model_id.replace(':','')
        model_id = model_id.replace('/','')
        model_id = model_id.replace(',','')
        model_id = model_id.replace('°','')
        model_id = model_id.replace('º','')
        model_id = model_id.replace('>','')
        model_id = model_id.replace('<','')
        if model_id in indices_dict:
            indices_dict[model_id].add(index)
        else:
            indices_dict[model_id] = {index}
    result = [indices for indices in indices_dict.values() if len(indices) > 1]
    return result
    
def get_modelID(df):
    # Unique modelID finder
    # Returns 2 lists with potential modelIDs by extracting words from title and checking how often they occur.
    # First list are values that occur once (bigger probability to contain noise)
    # Second list contains values that occur 2,3,4,5 times (quite low noise)

    # Add all titles togheter and split the title list into words
    modelID_set = set()
    for i in range(len(df['title'])):
        title_split = df['title'][i].split()
        title_split_unique = set(title_split)             # in case modelID occurs 2 times in title

        # First time filter
        def filter_words(word_list):
            filtered_words = [word for word in word_list if not (word.isalpha() or word.isdigit() or 'inch' in word
                              or 'hz' in word or '3d ' in word or '+' in word or '”' in word or "'" in word or "1080p" in word
                              or '720p' in word or '4k' in word or '2160p' in word or 'cdm2' in word or '2k' in word
                              or 'dynex' in word or "0cmr" in word or '480i' in word or '3dready' in word)]
            return filtered_words

        extracted_modelID = filter_words(title_split_unique)
        pattern = re.compile(r'\b(\w+)\b')
        extracted_words = [match.group(1) for item in extracted_modelID for match in re.finditer(pattern, item)]
        single_string = ' '.join(extracted_words)

        # Regular expression to match standalone "3d"
        pattern = re.compile(r'\b3d\b')

        # Remove standalone "3d"
        result_string = re.sub(r'\b3d\b', '', single_string).strip()

        if len(result_string) >= 1 :
            modelID_set.add(result_string)
        else:
            modelID_set.add(f'{i}')

    return {entry for entry in modelID_set if not entry.isnumeric()}
    
def get_title_modelID_pairs(modelID_list, df):
    modelID_indices = {}
    for index, row in df.iterrows():
        for word in row['title'].split():
            if word in modelID_list:
                if word in modelID_indices:
                    modelID_indices[word].append(index)
                else:
                    modelID_indices[word] = [index]
    modelID_indices = {model_id: indices for model_id, indices in modelID_indices.items() if len(indices) <= 4}

    title_modelID_pairs = []
    for modelID, indices in modelID_indices.items():
        if len(indices) > 1:
            for pair in itertools.combinations(indices, 2):
                if pair not in title_modelID_pairs:
                    title_modelID_pairs.append(pair)

    return title_modelID_pairs

def get_model_df(df,candidate_pairs,signature,modelID_pairs,TV_brands):
    '''
    Constructs dataframe used for training/testing duplicate detection method.
    Duplicate detection method is trained on LSH output of train set.
    Missing: multiple similarity measures
    '''
    candidate_df = pd.DataFrame(columns = ['duplicate', 'idx_tv1', 'idx_tv2', 'same_shop', 'same_brand' 'modelID_pair', 'signature_similarity', 'title_similarity', 'key_similarity'])

    for i in range(len(candidate_pairs[0])):
        # Indices
        idx1 = int(candidate_pairs[i][0])
        idx2 = int(candidate_pairs[i][1])
        candidate_df.loc[i, 'idx_tv1'] = idx1
        candidate_df.loc[i, 'idx_tv2'] = idx2   
        
        # Signature similarity
        A = signature[:,idx1]
        B = signature[:,idx2]
        similarity = jaccard_score(A,B,average='macro')
        candidate_df.loc[i, 'signature_similarity'] = similarity

        # True duplicate
        if df['modelID'][idx1] == df['modelID'][idx2]:
            candidate_df.loc[i, 'duplicate'] = 1
        else:
            candidate_df.loc[i, 'duplicate'] = 0
        
        # Check if model ID pairs
        if (idx1,idx2) in modelID_pairs:
            candidate_df.loc[i, 'modelID_pair'] = 1
        else:
            candidate_df.loc[i, 'modelID_pair'] = 0

        # Same store
        if df['shop'][idx1] == df['shop'][idx2]:
            candidate_df.loc[i, 'same_shop'] = 1
        else:
            candidate_df.loc[i, 'same_shop'] = 0

        # Same brand
        brand1 = ''
        brand2 = ''
        for word1 in df['title'][idx1].split():
            if word1 in TV_brands:
                brand1 = word1
        for word2 in df['title'][idx2].split():
            if word2 in TV_brands:
                brand2 = word2
        if brand1 == brand2:
            candidate_df.loc[i, 'same_brand'] = 1
        else:
            candidate_df.loc[i, 'same_brand'] = 0

        # 3-gram title similarity
        titA = ''.join(word for word in df['title'][idx1] if word not in string.punctuation)
        titA = titA.replace(' ','')
        titA = titA.lower()

        titB = ''.join(word for word in df['title'][idx2] if word not in string.punctuation)
        titB = titB.replace(' ','')
        titB = titB.lower()

        titSim = jaccard_similarity(titA,titB)
        candidate_df.loc[i, 'title_similarity'] = titSim

        # 3-gram feature similarity
        match1 = []
        match2 = []
        try:
            for key in df['features'][idx1].keys():
                match1.append(df['features'][idx1][key])
            for key in df['features'][idx2].keys():
                match2.append(df['features'][idx2][key])
            str1 = ''      
            for match in match1:
                st1 = ''.join(word for word in match if word not in string.punctuation)
                st1 = st1.replace(' ','')
                st1 = st1.lower()
                str1 += st1
            str2 = ''  
            for match in match2:
                st2 = ''.join(word for word in match if word not in string.punctuation)
                st2 = st2.replace(' ','')
                st2 = st2.lower()
                str2 += st2
            sim_m_kv = jaccard_similarity(str1, str2)    
        except ZeroDivisionError:
            sim_m_kv = 0
        candidate_df.loc[i, 'key_similarity'] = sim_m_kv     

    return candidate_df

def jaccard_similarity(a, b):
    ''' input: 2 keys (strings), of matching key-value pairs
        return: 3-gram  jaccard similarity '''
    N = 3
    x = {a[i:i+N] for i in range(len(a)-N+1)}
    y = {b[i:i+N] for i in range(len(b)-N+1)}
    intersection = x.intersection(y)
    union = x.union(y)
    return float(len(intersection)) / len(union)

def logistic_regression(candidates_df):
    X = candidates_df['signature_similarity','']
    y = candidates_df['duplicate']

def main():
    print("Loading data...")
    
    with open("D:/Studie/23-24/Blok 2/Computer Science/Personal Assignment/TVs-all-merged (1)/TVs-all-merged.json", 'r') as read_file:
        data = json.load(read_file)
    read_file.close()
    df = create_df(data)
    
    print("Creating product representations...")
    
    #Product representations are model words in the title and brands in the titles
    df['title'] = normalize_list(df['title'])
    TV_brands_1 = ["Bang & Olufsen","Continental Edison","Denver","Edenwood","Grundig","Haier","Hisense","Hitachi","HKC","Huawei","Insignia","JVC","LeEco","LG","Loewe","Medion","Metz","Motorola","OK.","OnePlus","Panasonic","Philips","RCA","Samsung","Sceptre","Sharp","Skyworth","Sony","TCL","Telefunken","Thomson","Toshiba","Vestel","Vizio","Xiaomi","Nokia","Engel","Nevir","TD Systems","Hyundai","Strong","Realme","Oppo","Metz Blue","Asus","Amazon","Cecotec","Nilait","Daewoo","insignia","nec","supersonic","viewsonic","Element","Sylvania","Proscan","Onn","Vankyo","Blaupunkt","Coby","Kogan","RCA","Polaroid","Westinghouse","Seiki","Insignia","Funai","Sansui","Dynex","naxa"]
    TV_brands_2 = ['Philips', 'Samsung', 'Sharp', 'Toshiba', 'Hisense', 'Sony', 'LG', 'RCA', 'Panasonic', 'VIZIO', 'Naxa', 'Coby', 'Vizio', 'Avue', 'Insignia', 'SunBriteTV', 'Magnavox', 'Sanyo', 'JVC', 'Haier', 'Venturer', 'Westinghouse', 'Sansui', 'Pyle', 'NEC', 'Sceptre', 'ViewSonic', 'Mitsubishi', 'SuperSonic', 'Curtisyoung', 'Vizio', 'TCL', 'Sansui', 'Seiki', 'Dynex']
    TV_brands = normalize_list(list(set(TV_brands_1) | set(TV_brands_2)))
    product_representations = product_representation(df['title'],TV_brands)

    print("Creating binary matrix...")

    bin_matrix, tokens = binary_matrix(product_representations)
    
    print("Min-hashing...")

    signatures = minhash(100,bin_matrix)
    
    print("LSH...")

    LSH_candidate_pairs = LSH(signatures, 0.9)

    print("Adding title model ID candidate pairs...")

    title_modelIDs = get_modelID(df)
    modelID_pairs = get_title_modelID_pairs(title_modelIDs,df)

    candidate_pairs = list(set(LSH_candidate_pairs) | set(modelID_pairs))

    print("Scalability solution evaluation:")

    #candidate_evaluation(candidate_pairs,df)

    print("Logistic regression...")

    model_df = get_model_df(df, candidate_pairs, signatures, modelID_pairs, TV_brands)

    print(model_df.head())


    
if __name__ == "__main__":
    main()