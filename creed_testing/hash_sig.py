import re
import numpy as np
import hashlib
from collections import defaultdict

######## Cleaning string to remove special charcters and normalize text ########
def clean_text(text):
    #Regular expression that 'a-zA-Z0-9" detects all uppercase lowercase and numbers, and replaces anything that is not that with nothing
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip().split()\
    

###### Fucntion to generate hashfuntions ######
def hash_funct (seed, x):
    return int(hashlib.md5((str(seed) + x).encode()).hexdigest(),16)


###### Creating mini hash signature ###### Tokens: set of words extracted from the cleaned text string --- Num_hashes: num of hashes to use... more =  more accurate but more computation time
def minhash_sig(tokens, num_hashes = 100):
    signature = [] #stroing signatures
    for seed in range(num_hashes): #generating hash functions
        min_hash = min(hash_funct(seed, token) for token in tokens) # computes unique hash value for each token, and we take the min
        signature.append(min_hash)
    return signature


###### Fucntion to compute Jaccard similarity estimate using Minhash signatures ######
def minhash_Jaccard(sig_1, sig_2):
    return np.mean(np.array(sig_1) == np.array(sig_2))


minhash_db = {}


# Sample dataset
drinks = [
    "White claw orange", 
    "White claw grape", 
    "Coors Light", 
    "Coors Lite",
    "Budweiser", 
    "Budweiser Original",
    "Jack Daniels Whiskey", 
    "Jack Daniel’s Whisky"
]

# Compute MinHash signatures for all drinks
for drink in drinks:
    tokens = set(clean_text(drink))
    minhash_db[drink] = minhash_sig(tokens)

#checking for duplciates
query = "coors lite"
query_sig = minhash_sig(set(clean_text(query)))

#going through db

for drink, sig in minhash_db.items():
    score = minhash_Jaccard(query_sig, sig)
    print(f"Similarity with {drink}: {score:.2f}")
    if score > 0.7:
        print("-> Possible duplicate detected!\n")
