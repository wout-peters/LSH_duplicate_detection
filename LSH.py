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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time


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

def product_representation(df,TV_brands,modelIDs):
    '''
    Creates a list of product representations
    Includes model words from the titles, and TV brand names in the title, and model words in the values
    '''
    representations = []

    for TV, row in df.iterrows():
        TV_rep = []
        title_list = df['title'][TV].split()
        for string in title_list:
            if is_model_word(string):
                TV_rep.append(string)
            if string in TV_brands:
                TV_rep.append(string)       
            if string in modelIDs:
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
    numHashFunc, numTV = signature_matrix.shape    
    nBuckets = find_next_prime(numTV)
    nBands = get_b(numHashFunc,thres)
    hash_buckets = initialize_hash_bucket(nBands, nBuckets)
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
                        #A = signature_matrix[:,pair[0]]
                        #B = signature_matrix[:,pair[1]]
                        #similarity = jaccard_score(A,B,average='macro')
                        #if similarity > thres, add to candidates  
                        #if similarity >= thres:
                            candidates.add(pair)            
    
    return list(candidates)        

def candidate_evaluation(candidates, df):
    '''
    Computes Pair Quality, Pair Completeness, F1_star, and Fraction of Comparisons given a list of candidate pairs
    '''
    true_duplicates = all_pairs(df) 
    candidates = [set(tuple_a) for tuple_a in candidates]
    result = [tuple_a for tuple_a in candidates if any(set(tuple_a).issubset(set_b) for set_b in true_duplicates)]

    def generate_2_combinations(input_set):
        return list(itertools.combinations(input_set, 2))
    combinations_list = [generate_2_combinations(set_b) for set_b in true_duplicates]
    flattened_combinations = [combination for sublist in combinations_list for combination in sublist]

    number_comparisons = len(candidates)
    true_duplicates_found = len(result)
    true_num_duplicates = len(flattened_combinations)
    possible_comparisons = math.factorial(len(df)) // (2 * math.factorial(len(df) - 2))

    PQ = true_duplicates_found / number_comparisons
    PC = true_duplicates_found / true_num_duplicates
    F1_star = (2*PQ*PC) / (PQ + PC)
    fraction_comparisons = number_comparisons / possible_comparisons

    return [PQ,PC,F1_star,fraction_comparisons]

def all_pairs(df):
    '''
    Returns a dictionary of model ID keys, with as values TV indices that have that model ID 
    '''
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
    '''
    Returns a list of mostly Model IDs, by filtering the words in the titles by some efficient rules.
    The list that is returned contains (unique) Model IDs, but is lightly contaminated with other words.
    '''
    modelID_set = set()
    for i, row in df.iterrows():
        title_split = df['title'][i].split()
        title_split_unique = set(title_split)

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
            split = result_string.split()
            modelID_set.update(split)
        else:
            modelID_set.add(f'{i}')

    return list({entry for entry in modelID_set if not entry.isnumeric()})
    
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

def get_model_df(df,candidate_pairs,signature,title_modelIDs,TV_brands):
    '''
    Constructs dataframe used for training/testing duplicate detection method.
    Duplicate detection method is trained on LSH output of train set.
    '''
    candidate_df = pd.DataFrame(columns = ['duplicate', 'idx_tv1', 'idx_tv2', 'same_shop', 'same_brand', 'modelID_pair', 'signature_similarity', 'title_similarity', 'key_similarity'])

    for i in range(len(candidate_pairs)):
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
        candidate_df.loc[i, 'modelID_pair'] = 0
        for word in df['title'][idx1]:
            if word in title_modelIDs:
                candidate_df.loc[i, 'modelID_pair'] += 1
        for word in df['title'][idx2]:
            if word in title_modelIDs:
                candidate_df.loc[i, 'modelID_pair'] += 1

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
        #match1 = []
        #match2 = []
        #try:
        #    for key in df['features'][idx1].keys():
        #        match1.append(df['features'][idx1][key])
        #    for key in df['features'][idx2].keys():
        #        match2.append(df['features'][idx2][key])
        #    str1 = ''      
        #    for match in match1:
        #        st1 = ''.join(word for word in match if word not in string.punctuation)
        #        st1 = st1.replace(' ','')
        #        st1 = st1.lower()
        #        str1 += st1
        #    str2 = ''  
        #    for match in match2:
        #        st2 = ''.join(word for word in match if word not in string.punctuation)
        #        st2 = st2.replace(' ','')
        #        st2 = st2.lower()
        #        str2 += st2
        #    sim_m_kv = jaccard_similarity(str1, str2)    
        #except ZeroDivisionError:
        #    sim_m_kv = 0
        #candidate_df.loc[i, 'key_similarity'] = sim_m_kv     

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

def logistic_regression(df_train, df_test):
    '''
    Returns predictive performance of a hyperparameter-optimized logistic regression on the test data, which was trained on the train data.
    '''
    X_train = df_train[['same_shop', 'same_brand', 'modelID_pair', 'signature_similarity', 'title_similarity', 'key_similarity']]
    X_test = df_test[['same_shop', 'same_brand', 'modelID_pair', 'signature_similarity', 'title_similarity', 'key_similarity']]

    label_encoder = LabelEncoder()
    df_train['duplicate'] = label_encoder.fit_transform(df_train['duplicate'])
    df_test['duplicate'] = label_encoder.fit_transform(df_test['duplicate'])
    y_train = df_train['duplicate']
    y_test = df_test['duplicate']
    
    model = LogisticRegression(random_state=0)    
    param_grid_lr = {'C': np.logspace(-3, 3, 7)}
    grid_lr = GridSearchCV(model, param_grid_lr, cv=3)
    fitgrid_lr = grid_lr.fit(X_train, y_train)
    y_pred = fitgrid_lr.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return f1, conf_matrix

def run_LSH(threshold,num_hash_func,df):
    TV_brands_1 = ["Bang & Olufsen","Continental Edison","Denver","Edenwood","Grundig","Haier","Hisense","Hitachi","HKC","Huawei","Insignia","JVC","LeEco","LG","Loewe","Medion","Metz","Motorola","OK.","OnePlus","Panasonic","Philips","RCA","Samsung","Sceptre","Sharp","Skyworth","Sony","TCL","Telefunken","Thomson","Toshiba","Vestel","Vizio","Xiaomi","Nokia","Engel","Nevir","TD Systems","Hyundai","Strong","Realme","Oppo","Metz Blue","Asus","Amazon","Cecotec","Nilait","Daewoo","insignia","nec","supersonic","viewsonic","Element","Sylvania","Proscan","Onn","Vankyo","Blaupunkt","Coby","Kogan","RCA","Polaroid","Westinghouse","Seiki","Insignia","Funai","Sansui","Dynex","naxa"]
    TV_brands_2 = ['Philips', 'Samsung', 'Sharp', 'Toshiba', 'Hisense', 'Sony', 'LG', 'RCA', 'Panasonic', 'VIZIO', 'Naxa', 'Coby', 'Vizio', 'Avue', 'Insignia', 'SunBriteTV', 'Magnavox', 'Sanyo', 'JVC', 'Haier', 'Venturer', 'Westinghouse', 'Sansui', 'Pyle', 'NEC', 'Sceptre', 'ViewSonic', 'Mitsubishi', 'SuperSonic', 'Curtisyoung', 'Vizio', 'TCL', 'Sansui', 'Seiki', 'Dynex']
    TV_brands = normalize_list(list(set(TV_brands_1) | set(TV_brands_2)))
    title_modelIDs = normalize_list(get_modelID(df))

    product_representations = product_representation(df,TV_brands,title_modelIDs)
    bin_matrix, tokens = binary_matrix(product_representations)
    signatures = minhash(num_hash_func,bin_matrix)
    LSH_candidate_pairs = LSH(signatures, threshold)
    return LSH_candidate_pairs, signatures, title_modelIDs, TV_brands

def get_data():
    with open("D:/Studie/23-24/Blok 2/Computer Science/Personal Assignment/TVs-all-merged (1)/TVs-all-merged.json", 'r') as read_file:
        data = json.load(read_file)
    read_file.close()
    df = create_df(data)
    df['title'] = normalize_list(df['title'])
    return df

def get_index(cand, indices):
    new_candidates = []
    for pair in cand:
        new_pair = []
        new_pair.append(indices[pair[0]])
        new_pair.append(indices[pair[1]])
        new_candidates.append(new_pair)
    return new_candidates

def main():
    df = get_data()
    performance_df = pd.DataFrame(columns = ['threshold', 'PQ', 'PC', 'F1_star', 'Fraction_of_comparisons', 'F1'])
    i = 0
    for t in np.arange(1.0, 0.0, -1):
        print(f"Threshold value:", t)
        tic = time.perf_counter()
        PQ = 0
        PC = 0
        F1_star = 0
        Fraction_of_comparisons = 0
        F1 = 0
        for bootstrap in range(5):
            indices_train, indices_test = train_test_split(df.index, test_size=0.37, random_state=42)
            train_candidates, train_signatures, train_title_modelIDs, TV_brands = run_LSH(t, 100, df.loc[indices_train])
            test_candidates, test_signatures, test_title_modelIDs, TV_brands = run_LSH(t, 100, df.loc[indices_test])
            print(len(train_candidates))
            #Performance: PQ, PC, F1_star, Fraction of Comparisons
            df_idx_train_candidates = get_index(train_candidates, indices_train)
            df_idx_test_candidates = get_index(test_candidates, indices_test)
            LSH_performance = candidate_evaluation(df_idx_test_candidates, df.loc[indices_test])
            PQ += LSH_performance[0]
            PC += LSH_performance[1]
            F1_star += LSH_performance[2]
            Fraction_of_comparisons += LSH_performance[3]
            #Logistic regression: 
            train_df = get_model_df(df.loc[indices_train], df_idx_train_candidates, train_signatures, train_title_modelIDs, TV_brands)
            test_df = get_model_df(df.loc[indices_test], df_idx_test_candidates, test_signatures, test_title_modelIDs, TV_brands)
            F1_, confusion = logistic_regression(train_df, test_df)
            F1 += F1_
        performance_df.loc[i, 'threshold'] = t
        performance_df.loc[i, 'PQ'] = PQ/5
        performance_df.loc[i, 'PC'] = PC/5
        performance_df.loc[i, 'F1_star'] = F1_star/5
        performance_df.loc[i, 'Fraction_of_comparisons'] = Fraction_of_comparisons/5
        performance_df.loc[i, 'F1'] = F1/5
        toc = time.perf_counter()
        print(f"Elapsed time at threshold {t} is {toc - tic:0.4f} seconds")
        i += 1
    
    print(performance_df)
    csv_file_path = '"C:\\Users\\Wout Peters\\Documents"'        
    df.to_csv(csv_file_path, index = False)

    
if __name__ == "__main__":
    main()