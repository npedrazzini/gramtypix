import re
from unidecode import unidecode
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def optimal_components_gmm(tfidf_matrix, random_state, max_components=3):
    lowest_bic = np.infty
    optimal_n_components = 1

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(tfidf_matrix)
        bic = gmm.bic(tfidf_matrix)
        if bic < lowest_bic:
            lowest_bic = bic
            optimal_n_components = n_components

    return optimal_n_components

# optimal_components = optimal_components_gmm(tfidf_matrix)
# Use optimal_components for GMM


def optimal_clusters_kmeans_agglomerative(tfidf_matrix, random_state,max_clusters=3):
    best_score = -1
    optimal_k = 2

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,n_init='auto')
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)

        if score > best_score:
            best_score = score
            optimal_k = n_clusters

    return optimal_k



def find_adv_head(text, subordinators,nlp):    
    # Process the input text with SpaCy
    doc = nlp(text)

    # Convert the subordinator to lowercase for case-insensitive matching
    subordinators = [sub.lower() for sub in subordinators]

    # Extract token:head information from the processed text
    result = []
    for token in doc:
        # Check if the token text (lowercased) matches the subordinator
        if token.text.lower() in subordinators:
            result.append(token.head.text)

    return result

def find_adv_head_absolutes_xadvs(text, subordinator):    
    # Process the input text with SpaCy
    # Regular expression to find 'ss' or 'ds' followed by words not in parentheses
    pattern = r'\b(absoluteadv|xadv)\b \((?:NOMATCH)\) (\w+)'
    
    re.findall(pattern, text)

    # Convert the subordinator to lowercase for case-insensitive matching
    subordinator = subordinator.lower()

    # Extract token:head information from the processed text
    result = []
    for token in doc:
        # Check if the token text (lowercased) matches the subordinator
        if token.text.lower() == subordinator:
            result.append(token.head.text)

    return result

def transform_targ(row, subjunctions, nlp):
    print('checking head')
    # Process the sentence with spaCy
    if row['targ'] is None:
        print('none line:', row['targ'])
        return None  # Skip this row
    else:
        doc = nlp(row['context'])

        # Split the 'targ' column into pairs
        targ_pairs = str(row['targ']).split(') ')
        transformed_pairs = []

        # Iterate over subjunctions and apply the transformation
        head_of_subjunction_position = []
        for subjunction in subjunctions:
            # Check if the parallel to the subjunction is 'NOMATCH'
            subjunction_nomatch = False
            for pair in targ_pairs:
                if pair.startswith(subjunction + ' ('):
                    subjunction_nomatch = pair.strip(')').endswith('(NOMATCH')
                    break

            # If the parallel to the subjunction is 'NOMATCH', find the head of the subjunction
            if subjunction_nomatch:
                for token in doc:
                    if token.text.lower() == subjunction:
                        head_of_subjunction_position.append(token.head.i)
                        break
        print('positions of heads:',head_of_subjunction_position)
            # Iterate over pairs in 'targ' and replace the head of the subjunction with 'advhead'
        current_position = 0
        for pair in targ_pairs:
            print('position',current_position)
            parts = pair.split(' (')
            if len(parts) != 2:
                # print(f"Error parsing pair: '{pair}' in row: {row['sent_id']}")
                continue

            word, translation = parts
            translation = translation.rstrip(')')

            if current_position in head_of_subjunction_position:
                print('translation of head',translation)
                if translation != 'NOMATCH':
                    transformed_pairs.append(f'advhead:{translation}')
                else:
                    transformed_pairs.append('advhead:')
                # head_of_subjunction_position = None  # Prevent further replacements
            else:
                print('translation',translation)
                if translation == 'NOMATCH':
                    transformed_pairs.append(f"{word}:")
                else:
                    transformed_pairs.append(f"{word}:{translation}")

            current_position += 1
    return '\t'.join(transformed_pairs)


def transform_sentence_find_head(sentence, headword, exclude_list):
    pattern = re.compile(r'(\w+)\s*\((\w*|NOMATCH)\)')
    transformed_pairs = []

    # Transform sentence into pairs
    sentence_pairs = [(match.group(1), match.group(2) if match.group(2) != 'NOMATCH' else '') for match in pattern.finditer(sentence)]

    # Flag to indicate if 'when' followed by a word in exclude list is found
    when_exclude_found = False

    # Iterate over pairs to transform sentence
    for word, translation in sentence_pairs:
        if word == 'when' and translation in exclude_list:
            when_exclude_found = True

        if word == headword and not when_exclude_found:
            transformed_pairs.append(f"advhead:{translation}")
        else:
            transformed_pairs.append(f"{word}:{translation}")

    return '\t'.join(transformed_pairs)

def transform_sentence_find_head_absolutes_xadvs(sentence, headword, exclude_list):
    pattern = re.compile(r'(\w+)\s*\((\w*|NOMATCH)\)')
    transformed_pairs = []

    # Transform sentence into pairs
    sentence_pairs = [(match.group(1), match.group(2) if match.group(2) != 'NOMATCH' else '') for match in pattern.finditer(sentence)]

    # Flag to indicate if 'when' followed by a word in exclude list is found
    when_exclude_found = False

    # Iterate over pairs to transform sentence
    for word, translation in sentence_pairs:
        if word == 'when' and translation in exclude_list:
            when_exclude_found = True
        if word == headword and not when_exclude_found:
            transformed_pairs.append(f"advhead:{translation}")
        else:
            transformed_pairs.append(f"{word}:{translation}")

    return '\t'.join(transformed_pairs)

def transform_sentence(sentence):
    pattern = re.compile(r'(\w+)\s*\((\w*|NOMATCH)\)')
    
    transformed_pairs = []
    for match in pattern.finditer(sentence):
        word = match.group(1)
        translation = match.group(2)

        # If translation is 'NOMATCH', leave it blank
        if translation == 'NOMATCH':
            translation = ''

        transformed_pairs.append(f"{word}:{translation}")

    return '\t'.join(transformed_pairs)

def remove_accents(input_str):
    if isinstance(input_str, str):
        return unidecode(input_str)
    else:
        return input_str 
# def extract_head_counterpart(row):
#     '''
#     Extracts the counterpart of a verb in an adverbial clause from a row of data.

#     This function specifically looks for a counterpart to the verb ('head') in an adverbial 
#     clause (e.g., 'when') in a target language. It processes the 'adv_head' and 'targ' 
#     fields of the row, where 'targ' contains pairs of words and their counterparts.

#     :param row: A dictionary or similar structure containing the data of a single row.
#     :return: The counterpart word in the target language if found, otherwise None.
#     '''
#     try:
#         # Extract 'adv_head' field from the row and convert it to a string.
#         adv_head = str(row['adv_head'])

#         # Extract 'targ' field from the row, remove accents, and convert it to a string.
#         # 'targ' is expected to contain pairs of words and their counterparts.
#         targ = remove_accents(str(row['targ']))

#         # Use regular expression to find all pairs in the 'targ' string.
#         # Each pair is expected to be in the format "word (counterpart)".
#         words_and_counterparts = re.findall(r'(\S+)\s*\(([^)]+)\)', targ)

#         # Iterate over the extracted pairs and find the first pair where the word matches 'adv_head'.
#         # This uses a generator expression to efficiently search for the matching pair.
#         counterpart = next((counterpart for word, counterpart in words_and_counterparts if word == adv_head), None)

#         # Return the found counterpart. If no match is found, None is returned.
#         return counterpart

    # except Exception as e:
    #     # If any error occurs (e.g., key errors, regex issues), print the error message and return None.
    #     print(f"Error: {e}")
    #     return None


def extract_head_counterpart(row):
    # Extract the target word from the 'eng-29' column
    target_word = row['eng-29']

    # Look for the first occurrence of the target word in 'transformed_targ' and get the next 'advhead:X'
    if target_word and row['transformed_targ']:
        parts = row['transformed_targ'].split('\t')
        found_target = False
        for part in parts:
            print(part)
            print(target_word)
            if part.startswith(str(target_word) + ':'):
                found_target = True
            if found_target and part.startswith('advhead:'):
                return part.split(':')[1]  # Return the word after 'advhead:'
    else:
        return None

def produce_labels_stopwords(pairs, stopwords_source):
    '''
    Process a given pair to determine if it contains a stopword and assign a label accordingly.
    This is to be used only within the source_target_stopwords_assoc function
    to find the most likely forms corresponding to a given list of stopwords in the source language

    :param pairs: A string in the format "key:value" where 'key' is a potential stopword.
    :param stopwords_source: A list of stopwords to check against.
    :return: A list of tuples. Each tuple contains a label (1 or 0) and the second part of the pair.

    The function checks if the first part of the pair (before the colon) is in the list 
    'stopwords_source'. If it is a stopword, it returns a tuple with label 1, otherwise 0.
    This label helps in identifying whether the word is a stopword or not.
    '''
    key, value = pairs.split(':')
    
    if key in stopwords_source:
        # If the key is a stopword, return a tuple with label 1
        return [(1, value)]
    else:
        # Otherwise, return a tuple with label 0
        return [(0, value)]
        
def produce_labels(pairs, target_words):
    '''
    This function processes a string 'pairs' to determine if it contains the target word(s). 

    :param pairs: A string in the format "key:value". The 'key' is checked against the target words.
    :param target_words: A collection of words (list, set, etc.) that are being targeted for identification.
    :return: A list containing a single tuple. The tuple consists of a label (1 or 0) and the value part of the pair.

    '''
    # Split the 'pairs' string into 'key' and 'value' based on the colon ':'.
    key, value = pairs.split(':')

    # Check if the 'key' part is in the list of target words.
    if key in target_words:
        # If the 'key' is a target word, return a tuple with label 1 and the 'value'.
        return [(1, value)]
    else:
        # If the 'key' is not a target word, return a tuple with label 0 and the 'value'.
        return [(0, value)]
