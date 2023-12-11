# LSH_duplicate_detection

Note: the code used was written in collaboration with Sjoerd Bommer (621763)

This page contains the code of my solution to the assignment of Computer Science for Business Analytics, given at the Erasmus School of Economics. \
For the assignment, students were asked to create a scalable duplicate detection method, using Locality Sensitive Hashing. The dataset used contains web scraped information of 1624 televisions, from four different Web shops: Amazon.com, Newegg.com, Best-Buy.com, and TheNerds.net. Each TV has the following data: Shop name, Title, a set of Features, and the ModelID. As the ModelID is often not explicitly stated in the product features, the ModelID can not be used directly for the duplicate detection, and is rather used for evaluation purposes, as equal ModelIDs correspond to duplicate products.\
\
The structure of the code is as follows: 

First, the data is loaded, and pre-processed. In particular, we normalize the titles by setting everything to lower cases. Additionally, we replace all variations of inch (Inch, ", -inch, inches, etc.) by 'inch'. We do the same for variations of hz. Also, we delete all kinds of punctuation from the titles. \
Then, we create product representations that are used for LSH. From the normalized titles, we include in the product representations all "Model Words", words that contain both letters and numbers. We also have a list of all TV brands, and we go through all titles word-by-word, and add any brands that we find in the title to the product representations. \
Then, we transform these product representations for all TVs into a binary matrix, containing all 'tokens' from the product representations as rows, and TVs as columns. This binary matrix is then transformed into a signature matrix using Minhashing. Minhashing reduces large binary matrices into smaller signature matrices using random hash functions. We use 100 random hash functions of the form (a*x + b) % P, where x is the input, a and b are (unique) random integers between 0 and 100, and P is the first prime larger than the number of tokens in the binary matrix. This way, we ensure that each hash function is unique and that there is a sufficient number of hash buckets. To implement minhashing, we use an efficient algorithm that was proposed in the lectures. \
The resulting signature matrix is then used to accurately and quickly compare the (Jaccard) similarity of two TVs, using LSH. The output of LSH is a list of candidate pairs, which all have the property that the Jaccard similarity of their signature matrices is larger than a threshold t. Furthermore, LSH divides the signature matrix M into b bands, each containing r rows, and each candidate pair also has the property that both candidates in the pair hashed to the same bucket for at least one band. Obviously, it should hold that b * r = nRows(M), but we use an approximation such that the last band can contain less than r rows. Furthermore, in the lecture, it was derived that it is optimal to use the relation t = (1/b) ^ (1/r). Given t, we take b and r to satisfy this relationship as good as possible. \ 
As the performance of pure LSH is rather poor, we use an additional scalability approach in parallel to LSH. Inspecting the data, I found that almost all titles contain the modelIDs of the TVs. We aim to exploit this information optimally. We read through all the titles of the TVs, and delete all words that contain only letters, only numbers, or other specific combinations of letters and numbers that occur often, but can not be a ModelID (720p, 4k, 3d, etc.). The resulting words are mostly ModelIDs. Then, we go through all TVs, and check whether there is a match between a word in the ModelID list we created, and a word in the title. If so, the index of that TV is stored in a bucket with the (presumably) ModelID as key. We keep key-value pairs that have at most 4 indices matched with the key, as it is very unlikely that one store has duplicate products, and there are 4 stores in the sample. The resulting duplicate pairs are then added to the candidate pairs produced by LSH. \
Given the duplicate pairs, for varying values of t, we compute Pair Quality, Pair Completeness, F1_star (the harmonic mean of PQ and PC), and the fraction of comparisons. 

Then, we use LogisticRegression as a classification algorithm to detect true candidate pairs out of the candidate pairs. The candidate pairs are split into a train and test set for 5 bootstraps, with a 63%/37% train/test split. The entries are pairs of TVs, and the dependent variable indicates whether this pair of TVs is a duplicate (1) or not (0), which is constructed by comparing the true ModelIDs (which was not used for training, but remained available in the data but hidden to the models). The features used to explain and then predict true duplicates are: 'Same shop', 'Same brand', 'Model ID Pair' (whether the pair was made candidate using the ModelID approximation procedure), 'Signature Similarity' (Jaccard similarity of signature vectors), 'Title Similarity' (3-gram Jaccard similarity of the titles), and 'Key Similarity' (3-gram Jaccard similarity of the feature values). The hyperparameter C, which controls overfitting, was optimized using GridsearchCV. The algorithm is trained on the training data, and tested on the test data, on which the F1-score is calculated. The F1-score is reported over all bootstraps and thresholds, and reported versus the fraction of comparisons. 
