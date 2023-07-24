# Topic_modeling

## Part 1 :
### Extract data :
- 0/ Create empty df
- 1/ Extract corpus (Web scrapping on XML) and 1st article cleaning (regex)
- 2/ Add extracted data in the df and save the df



## Part 2 :
### Pre-processing :
- 0/ Open dataset
- 1/ Select data for training (aka, training dataset)
- 2/ remove stopword and Lemmatization


### Building word dictionnary :
- 0/ Creating term dictionary of corpus, where each unique term is assigned an index
- 1/ Filter terms which occurs in less than 4 articles (aim to reduce overfitting) & more than 40% of the articles (aim to reduce underfitting)
- 2/ List of few words which are removed from dictionary as they are content neutral
- 3/ Analyse The most frequent words with their respective frequencies


### Feature extraction (Bag of Words) : 
- 1/ Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.


### LDA model training :
- 0/ Naiv modeling
- 1/ Optimize hyper parameters
- 2/ Ajust the model with best hyper parameters 
- 3/ Analyse the result of all topics
- 4/ Save the model and the list of topics



## Part 3 :
### Load the LDA model :


### Open test data :


### Create functions for Pre-processing on test set :
- 0/ Clean article with regex
- 1/ Drop empty and useless article
- 2/ Clean article with stop word and lemmatization


### Document clustering :
- 0/ Clustering articles
- 1/ Analyse the result of the clustering


### Theme extraction :
- 0/ Give an article and obtain the topic of article


### Document exploration :
- 0/ Give a topic and obtain a list of articles (from test) which are the most probably belong to the mentionned topic
