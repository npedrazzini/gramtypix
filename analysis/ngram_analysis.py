
import pandas as pd
import numpy as np
import codecs
import math
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2

from utils.text_processing import produce_labels, produce_labels_stopwords

class AssocAnalysis(object):
    def __init__(self, X, Y, feature_names):
        '''
        Constructor for the AssocAnalysis class.

        :param X: 2D array-like, [n_samples, n_features]
                  Training set - a set of feature vectors.
        :param Y: array-like, shape = [n_samples]
                  Target values (class labels in classification).
        :param feature_names: List of feature names.
        '''
        self.X = X  # Feature vectors
        self.Y = Y  # Target values
        self.feature_names = feature_names  # Feature names

    def extract_features_fdr(self, file_name, N, alpha=5e-2):
        '''
        Perform feature selection using False Discovery Rate (FDR).

        :param file_name: Name of the file where results will be written.
        :param N: Number of top features to select.
        :param alpha: Threshold for the FDR correction.
        '''
        # The SelectFdr class is used for feature selection, using a false discovery rate approach.
        # chi2 computes the chi-squared stat between each feature and the target.
        # This is a filter method for feature selection.
        selector = SelectFdr(chi2, alpha=alpha)
        
        # Fit the model and transform the data, selecting the most significant features.
        selector.fit_transform(self.X, self.Y)

        # Compute scores for each feature and filter out NaN values.
        # scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if not math.isnan(s)}
        scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if not math.isnan(s) and selector.pvalues_[i] < alpha}

        # Sort the scores in descending order and select the top N features.
        # scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)[0:N]
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)
        # Open a file to write the feature selection results.
        f = codecs.open(file_name, 'w')

        # Calculate the total count of positive and negative instances in the dataset.
        c_1 = np.sum(self.Y)  # Count of positive instances
        c_0 = len(self.Y) - c_1  # Count of negative instances

        # Write the header line in the file.
        f.write('\t'.join(['feature', 'score', 'p-value', 'c11', 'c10', 'c01', 'c00']) + '\n')

        # Convert sparse matrix to dense array if necessary.
        self.X = self.X.toarray()

        pos_scores = []  # List to store positive scores

        # Iterate through the top N features to calculate additional statistics.
        for w, score in scores:
            # Iterate over the top N features and their corresponding scores.
            # 'w' is the feature name and 'score' is its associated score and p-value.

            # Extract the feature column for the current feature 'w'.
            feature_array = self.X[:, list(self.feature_names).index(w)]
            
            # 'pos' is a list that contains the values of the feature 'w' for all samples where the target Y is 1 (positive class).
            # This is done by iterating over all samples and selecting the value of the feature 'w' if the corresponding target value is 1.
            pos = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 1]
            
            # Similarly, 'neg' is a list that contains the values of the feature 'w' for all samples where the target Y is 0 (negative class).
            # This is done by iterating over all samples and selecting the value of the feature 'w' if the corresponding target value is 0.
            neg = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 0]


            # Count the occurrences of each class for the current feature.
            c11 = np.sum(pos)  # True positives
            c01 = c_1 - c11    # False negatives
            c10 = np.sum(neg)  # False positives
            c00 = c_0 - c10    # True negatives

            # Adjust the score based on the counts.
            s = score[0]
            # Extract the actual chi-squared score from the score tuple. 
            # The 'score' tuple contains the chi-squared statistic and its corresponding p-value.

            if c11 > ((1.0 * c11) * c00 - (c10 * 1.0) * c01):
                # This condition checks if the observed frequency of the feature in the positive class (c11)
                # is greater than the expected frequency under the assumption of independence.
                # If this condition is true, it negates the score. This can be used to identify features
                # that are more frequent in the positive class than expected.
                s = -s

            s = np.round(s, 2)
            # Round the score to two decimal places for better readability and to simplify further analysis.

            # Store and write the positive scores.
            if s > 0:
                # Append the feature name, score, p-value, and contingency table values to the 'pos_scores' list
                # if the score is positive. This implies that the feature is positively correlated with the target.
                pos_scores.append([str(w), s, score[1], c11, c10, c01, c00])

            # Write the results to a file.
            # For each feature, write its name, score, p-value, and the counts from the contingency table.
            f.write('\t'.join([str(w), str(s), str(score[1])] + [str(x) for x in [c11, c10, c01, c00]]) + '\n')

        # Close the file after writing all the data.
        f.close()

        return pos_scores


def source_target_stopwords_assoc(tagged_file, outputdir, stopwords_source, subordinator, language):
    '''
    Analyze a target text file to identify and collect stopwords in the target language from 
    a list of stopwords in the source language.
    Potentially overkill - to be tested whether this step could be avoided and substituted for a simple target-word search

    :param tagged_file: Path to the text file that needs to be analyzed.
    :return: Results from the AssocAnalysis, identifying significant features (potential stopwords).

    This function reads a file, processes its contents to flag stopwords, 
    and then performs a chi-squared analysis to identify significant features.
    '''
    stopwords_target = []

    # Initialize a TfidfVectorizer for word-level feature extraction.
    tfvec = TfidfVectorizer(use_idf=False, ngram_range=(1, 1), norm=None, stop_words=stopwords_target, lowercase=True, binary=False)

    tagged_word_reduced = []
    # Open the file using codecs for proper UTF-8 encoding handling, especially for non-ASCII characters.
    with codecs.open(tagged_file, 'r', 'utf-8') as file:
        for line in file:
            words = line.split()
            for pairs in words:
                # Process each word with produce_labels_stopwords and extend the results to tagged_word_reduced.
                tagged_word_reduced.extend(produce_labels_stopwords(pairs,stopwords_source))
    
    # Filter out words that are in the target stopwords list.
    tagged_word_reduced = [(index, word) for index, word in tagged_word_reduced if word not in stopwords_target]

    if len(tagged_word_reduced) > 1:
        # Extract words for TF-IDF transformation and target labels.
        corp = [item[1] for item in tagged_word_reduced if len(item[1]) > 0]
        X = tfvec.fit_transform(corp)
        Y = [item[0] for item in tagged_word_reduced if len(item[1]) > 0]

    # Retrieve feature names and perform chi-squared analysis.
    feature_names = tfvec.get_feature_names_out()
    CHA = AssocAnalysis(X, Y, feature_names)
    # The results of the chi-squared analysis are saved to a file and returned.
    res = CHA.extract_features_fdr(f'{outputdir}{language}-{subordinator}-stopwords.txt', 100)
    
    return res

def WordAssoc(tagged_file, outputdir, target_words, stopwords_source, stopwords_target, subordinator, language):
    # Part 1: Word-Level Association
    # Initialize a TfidfVectorizer for word-level feature extraction.
    # Configuration: no IDF, unigrams only, exclude stopwords, all lowercase, and no binary weighting.
    tfvec = TfidfVectorizer(use_idf=False, ngram_range=(1, 1), norm=None, stop_words=stopwords_target, lowercase=True, binary=False)
    
    # Read the input file, process each line to extract word-level features, and remove source stopwords.
    # 'tagged_word_reduced' will be a list of tuples (index, word).
    tagged_word_reduced = []
    with codecs.open(tagged_file, 'r', 'utf-8') as file:
        for line in file:
            words = line.split()
            for pair in words:
                # Process each pair and add the result to tagged_word_reduced
                tagged_word_reduced.extend(produce_labels(pair, target_words))

    # Filter out words that are in the target stopwords list from 'tagged_word_reduced'.
    tagged_word_reduced = [(index, word) for index, word in tagged_word_reduced if word not in stopwords_target]
    
    # Check if the list 'tagged_word_reduced' contains more than one item for further processing.
    if len(tagged_word_reduced) > 1:
        # Extract the words (second element of each tuple) for TF-IDF transformation.
        corp = [item[1] for item in tagged_word_reduced if len(item[1]) > 0]
        # Transform the list of words into a TF-IDF matrix.
        X = tfvec.fit_transform(corp)
        # Extract the indices (first element of each tuple) as target labels.
        Y = [item[0] for item in tagged_word_reduced if len(item[1]) > 0]

    # Retrieve the feature names (words) from the TF-IDF vectorizer.
    feature_names = tfvec.get_feature_names_out()
    # Create a AssocAnalysis object for chi-squared statistical analysis.
    CHA = AssocAnalysis(X, Y, feature_names)
    # Perform the chi-squared analysis and write the top 200 features to a file.
    res = CHA.extract_features_fdr(f'{outputdir}{language}-{subordinator}-word-assoc.txt', 200)
    
    return res

def NGramAssoc(tagged_file, outputdir, target_words, stopwords_source, stopwords_target, subordinator, language):
    # Part 2: Character n-Gram Level Association
    # Reinitialize TfidfVectorizer for character n-gram level feature extraction.
    # Configuration: no IDF, n-grams from 2 to 8 characters, character analyzer, all lowercase, no binary weighting, and a minimum document frequency of 10.
    tfvec = TfidfVectorizer(use_idf=False, min_df=10, ngram_range=(2, 8), analyzer='char', lowercase=True, binary=False)
    print('tfvec initialized')
    # Read the input file again, this time processing each line for character n-gram analysis and adding the head the adverbial as a potential n-gram candidate
    tagged_word_reduced = []

    # Open the file again and process each line for character n-gram level associations.
    print(f'now reading {tagged_file}')
    with codecs.open(tagged_file, 'r', 'utf-8') as file:
        for line in file:
            if line is not None:
                for pair in line.split():
                    # Process each pair for character n-gram analysis.
                    tagged_word_reduced.extend(produce_labels(pair, target_words))
    print(f'{tagged_file} read')
    # Filter out character n-grams that are in the updated stopwords list.
    tagged_word_reduced = [(index, word) for index, word in tagged_word_reduced if word not in stopwords_target]
    print('tagged_word_reduced: done')
    # Check if the list 'tagged_word_reduced' contains more than one item for further processing.
    if len(tagged_word_reduced) > 1:
        print('tagged_word_reduced more than 1')
        # Prepare the character n-grams for TF-IDF transformation, adding '$' and '@' as markers at the start and end.
        corp = ['$' + item[1].strip() + '@' for item in tagged_word_reduced if len(item[1]) > 0]
        # Transform the list of character n-grams into a TF-IDF matrix.
        X = tfvec.fit_transform(corp)
        # Extract the indices (first element of each tuple) as target labels.
        Y = [item[0] for item in tagged_word_reduced if len(item[1]) > 0]
    
    # Retrieve the feature names (character n-grams) from the TF-IDF vectorizer.
    # Perform chi-squared analysis for character n-gram level features.
    feature_names = tfvec.get_feature_names_out()
    print('feature_names:', feature_names)
    CHA = AssocAnalysis(X, Y, feature_names)
    res = CHA.extract_features_fdr(f'{outputdir}{language}-{subordinator}-char-ngram-assoc.txt', 200)

    # Return the results of the character n-gram level analysis.
    return res
