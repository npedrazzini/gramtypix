#!/usr/bin/env python
# coding: utf-8
# # Mapping NULL constructions cross-linguistically

import pandas as pd
import numpy as np
from glob import glob
import yaml
import spacy
import spacy.cli

# Check if the model is already downloaded
if not spacy.util.is_package("en_core_web_sm"):
    # If not, download the model
    spacy.cli.download("en_core_web_sm")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from utils.text_processing import transform_sentence, transform_targ, remove_accents, extract_head_counterpart,optimal_clusters_kmeans_agglomerative,optimal_components_gmm
from analysis.ngram_analysis import WordAssoc, NGramAssoc, source_target_stopwords_assoc

# Load the English language model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# ------------------- Import configs --------------------

with open("./src/config.yaml", "r") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# --- Relevant vars
## -- General
source_tokens = configs['ngramsearch']['words']
sourcelang = configs['ngramsearch']['sourcelang']
targetlangs = configs['ngramsearch']['targetlangs']
advdf_path = configs['ngramsearch']['advdf_path']
alignments_parent_path = configs['ngramsearch']['alignments_parent_path']
stopwords_source = configs['ngramsearch']['stopwords_source']
outputdir = configs['ngramsearch']['outputdir']

stopwords_target = []

# Open a file to write output related to ngrams

source_tokenname = '-'.join(source_tokens)

targetlangswithcode = []
with open(f'{outputdir}{source_tokenname}-ngrams-details.txt','w') as outtxtgrams:
    # Load the dataframe containing alignments
    advdf = pd.read_csv(advdf_path,dtype=str)
    advdf = advdf.applymap(remove_accents)
    
    sourcelangcode = advdf.filter(like=sourcelang).columns[0]
    # advdf = advdf[advdf[sourcelangcode] == source_token]
    advdf = advdf[advdf[sourcelangcode].isin(source_tokens)]

    # Collect all alignment CSV files from the specified folder
    alignments_paths = sorted(glob(f'{alignments_parent_path}/*.csv'))

    # Iterate through each alignment file
    for alignments_path in alignments_paths:
        language = alignments_path.split('/')[-1].split('-')[0]
        lang_name_with_code = alignments_path.split('/')[-1].split('.csv')[0]

        if language in targetlangs:
            print(lang_name_with_code)
            # try:
            print(lang_name_with_code)
            # Read the current alignment file
            df_with_parall = pd.read_csv(alignments_path,dtype=str)
            df_with_parall = df_with_parall.applymap(remove_accents)

            # Merge the alignment data with the {words}-dataframe based on 'sent_id' and 'context'
            df_with_parall = advdf.merge(df_with_parall, how='left',on=['sent_id','context'])
#####
            # Process each row to transform the sentence based on the target and headword
            with open(f'{outputdir}{lang_name_with_code}-{source_tokenname}.txt', 'w') as outtxt:
                for index, row in df_with_parall.iterrows():
                    if not pd.isna(row['context']):
                        target = str(row['targ'])
                        output_sentence = transform_sentence(str(target))
                        outtxt.write(output_sentence + '\n')

            # Define target words for NGram alignment
            target_words=source_tokens
#####
            # Collect stopwords in the target language
            res = source_target_stopwords_assoc(f'{outputdir}{lang_name_with_code}-{source_tokenname}.txt', outputdir, stopwords_source, source_tokenname, lang_name_with_code)
            
            # Now read in the file with stopwords matches
            df_sw= pd.read_csv(f'{outputdir}{lang_name_with_code}-{source_tokenname}-stopwords.txt',sep='\t')

            # Initialize stopwords_target if not provided by the user
            if 'stopwords_target' not in locals():
                stopwords_target = []
            
            stopwords_target = list(df_sw.sort_values(by='p-value')['feature'][0:len(stopwords_source)])

            res=WordAssoc(f'{outputdir}{lang_name_with_code}-{source_tokenname}.txt',outputdir,target_words,stopwords_source,stopwords_target,source_tokenname,lang_name_with_code)
            # Read the results of the chi-squared analysis from the file.
            
            df = pd.read_csv(f'{outputdir}{lang_name_with_code}-{source_tokenname}-word-assoc.txt', sep='\t')

            # Add likely full words as new stopwords when looking for n-grams.

            # Rule 1: p-value of 0
            rule1 = df[df['p-value'] == 0]

            # Rule 2: p-value with at least 50 zeros
            rule2 = df[df['p-value'].apply(lambda x: int(str(x).split('e-')[1]) if 'e-' in str(x) and '.' in str(x).split('e-')[0] else 0) >= 50]

            # Rule 3: p-value up to 40 zeros and c10 = 0
            rule3 = df[(df['p-value'].apply(lambda x: 40 < int(str(x).split('e-')[1]) <= 50 if 'e-' in str(x) and '.' in str(x).split('e-')[0] else False)) & (df['c10'] == 0)]

            # Combine all rules
            filtered_df = pd.concat([rule1, rule2, rule3]).drop_duplicates()
            source_token_words = filtered_df['feature'].tolist()

            stopwords_target.extend(source_token_words)
            print('stopwords target:', stopwords_target)

            exclude_list = stopwords_target

            # Process each row to transform the sentence based on the target and headword
            with open(f'{outputdir}{lang_name_with_code}-{source_tokenname}-with-adv-head.txt', 'w') as outtxt:
                for index, row in df_with_parall.iterrows():
                    if not pd.isna(row['context']):
                        output_sentence = transform_targ(row,source_tokens,nlp)
                        outtxt.write(output_sentence + '\n')

            # Apply the transformation
            df_with_parall['transformed_targ'] = df_with_parall.apply(lambda row: transform_targ(row, source_tokens, nlp), axis=1)
            
            print('ADV head file generated')

            target_words.append('advhead')
            print('targetwords',target_words)
            res=NGramAssoc(f'{outputdir}{lang_name_with_code}-{source_tokenname}-with-adv-head.txt',outputdir,target_words,stopwords_source,stopwords_target,source_tokenname,lang_name_with_code)

            print('Ran the ngramassoc function')
            
            # Your DataFrame
            df = pd.read_csv(f'{outputdir}{lang_name_with_code}-{source_tokenname}-char-ngram-assoc.txt', sep='\t')

            # Only consider ngrams occurring at the end of words (signaled by '@')
            # Note that this is experimental, and by no means the best approach. Needs to be tested systematically against, e.g. templatic languages
            
            df['feature'] = df['feature'].astype(str)
            # df = df[(df['score'] > 1) & (df['feature'].str.contains('@'))].sort_values(by=['c11'], ascending=False).head(20)
            df = df[(df['score'] > 5) & (df['feature'].str.contains('@')) & (~df['feature'].str.startswith('$'))]

            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 8))
            tfidf_matrix = vectorizer.fit_transform(df['feature']).toarray()  # Convert to dense array

            # Number of clusters/components

            # K-Means clustering with different initialization methods
            random_state = 42  # Random state for reproducibility
            np.random.seed(random_state)

            optimal_k = optimal_clusters_kmeans_agglomerative(tfidf_matrix,random_state)

            kmeans_random = KMeans(n_clusters=optimal_k, init='random', random_state=random_state,n_init='auto')
            df['cluster_kmeans_random'] = kmeans_random.fit_predict(tfidf_matrix)

            kmeans_kmeans_plus_plus = KMeans(n_clusters=optimal_k, init='k-means++', random_state=random_state,n_init='auto')
            df['cluster_kmeans_kmeans++'] = kmeans_kmeans_plus_plus.fit_predict(tfidf_matrix)

            # DBSCAN clustering
            dbscan = DBSCAN(eps=1, min_samples=3)
            df['cluster_dbscan'] = dbscan.fit_predict(tfidf_matrix)

            # Agglomerative clustering
            agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
            df['cluster_agglomerative'] = agglomerative.fit_predict(tfidf_matrix)

            optimal_components = optimal_components_gmm(tfidf_matrix,random_state)

            # GMM clustering with fixed number of components
            gmm = GaussianMixture(n_components=optimal_components, random_state=random_state)
            df['cluster_gmm'] = gmm.fit_predict(tfidf_matrix)

            # Print cluster assignments
            for algorithm, cluster_columns in [('K-Means (Random)', 'cluster_kmeans_random'),
                                            ('K-Means (k-means++)', 'cluster_kmeans_kmeans++'),
                                            ('DBSCAN', 'cluster_dbscan'),
                                            ('Agglomerative', 'cluster_agglomerative'),
                                            ('GMM', 'cluster_gmm')]:
            # for algorithm, cluster_columns in [('DBSCAN', 'cluster_dbscan')]:
                print(f"\n{algorithm} Clustering:")
                for cluster_number in sorted(df[cluster_columns].unique()):
                    # Select rows corresponding to the current cluster number
                    cluster_rows = df[df[cluster_columns] == cluster_number]

                    # Print the values in df['feature'] as a list for the current cluster
                    feature_list = cluster_rows['feature'].tolist()
                    print(f'Cluster {cluster_number}: {feature_list}')
            df_with_parall.to_csv(f'dfheadwithparallel{lang_name_with_code}.csv')
            # Apply the function to create a new column
            
            df_with_parall['adv_head_transl'] = df_with_parall.apply(extract_head_counterpart, axis=1)

            ngrams_clusters = []
            full_words = [stopwords_target]
            print('fullwords included:',full_words)

            for name, group in df[df['cluster_dbscan'] >= 0].groupby('cluster_dbscan'):
                # print(f'Cluster {name}')
                cluster = []
                # print(group['feature'])
                for ngram in group['feature']:
                    if '$' in ngram:
                        newngram = ngram.split('$')[1].split('@')[0]
                        # print(f'ngram {newngram} is full word')
                        full_words.append(newngram)
                        # cluster.append(newngram)
                    else:
                        newngram = ngram.split('@')[0]
                        cluster.append(newngram)
                ngrams_clusters.append(cluster)

            outtxtgrams.write(lang_name_with_code)
            outtxtgrams.write('\n')
            outtxtgrams.write('\n')
            for iclu in range(len(ngrams_clusters)):
                ngramcurrent = ', '.join(ngrams_clusters[iclu])
                clustern = iclu + 1
                outtxtgrams.write(f'ngram_{clustern}: {ngramcurrent}')
                outtxtgrams.write('\n')

            print('adding column')
            newcol = []

            print('final source_token_words:',source_token_words)
            print('final full_words:',full_words)
            for index,row in df_with_parall.iterrows():
                if str(row[lang_name_with_code]) in source_token_words:
                    newcol.append(str(row[lang_name_with_code]))
                    print(f'IF1 appending {str(row[lang_name_with_code])}')
                elif row[lang_name_with_code] == 'NOMATCH':
                    anytrue = []
                    for cluster_values in ngrams_clusters:
                        ends_with_any = any(str(row['adv_head_transl']).endswith(value) for value in cluster_values)
                        anytrue.append(ends_with_any)
                    if True in anytrue:
                        indexoftrue = anytrue.index(True)
                        foundcluster = ngrams_clusters[indexoftrue]
                        wordtoprint = str(row['adv_head_transl'])
                        newcol.append(f'ngram_{indexoftrue + 1}')
                        print(f'IF4 appending ngram_{indexoftrue + 1} for {wordtoprint}')
                    else:
                        newcol.append('NOMATCH')
                        wordtoprint = str(row['adv_head_transl'])
                        print(f'IF4 appending NOMATCH for {wordtoprint}')
                else:
                    anytrue = []
                    for cluster_values in ngrams_clusters:
                        if str(row[lang_name_with_code]) not in cluster_values:
                            ends_with_any = any(str(row[lang_name_with_code]).endswith(value) for value in cluster_values)
                            anytrue.append(ends_with_any)
                        else:
                            anytrue.append('fullwordnotngram')
                    if True in anytrue:
                        indexoftrue = anytrue.index(True)
                        foundcluster = ngrams_clusters[indexoftrue]
                        newcol.append(f'ngram_{indexoftrue + 1}')
                        wordtoprint = str(row[lang_name_with_code])
                        print(f'IF5 appending ngram_{indexoftrue + 1} for {wordtoprint}')
                    elif 'fullwordnotngram' in anytrue:
                        newcol.append(str(row[lang_name_with_code]))
                        print(f'IF5 appending {str(row[lang_name_with_code])}')
                    else:
                        anytrue = []
                        for cluster_values in ngrams_clusters:
                            ends_with_any = any(str(row['adv_head_transl']).endswith(value) for value in cluster_values)
                            anytrue.append(ends_with_any)
                        if True in anytrue:
                            indexoftrue = anytrue.index(True)
                            foundcluster = ngrams_clusters[indexoftrue]
                            newcol.append(f'ngram_{indexoftrue + 1}')
                            wordtoprint = str(row['adv_head_transl'])
                            print(f'IF6 appending ngram_{indexoftrue + 1} for {wordtoprint}')
                        else:
                            if str(row[lang_name_with_code]) not in stopwords_target:
                                newcol.append(str(row[lang_name_with_code]))
                                wordtoprint = str(row[lang_name_with_code])
                                print(f'IF7 appending {wordtoprint}')
                            else:
                                newcol.append('NOMATCH')
                                print(f'IF7 appending NOMATCH')
            advdf[lang_name_with_code] = newcol
            targetlangswithcode.append(lang_name_with_code)

advdf.to_csv(f'{outputdir}{source_tokenname}_withgrams.csv',index=False)

selected_columns = advdf.loc[:, [col for col in advdf.columns if any(lang in col for lang in targetlangswithcode)]].columns
selected_columns = ['sent_id','context','eng-29'] + list(selected_columns)
advdf.to_csv(f'{outputdir}{source_tokenname}_withgrams_selectedcols.csv',index=False,columns=selected_columns)
