# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

## Objective
Practise basic NLP tasks: text preprocessing, Bag-of-Words (BoW) features and training a traditional ML classifier to predict sentiment.

## Dataset
UCI *Sentiment Labelled Sentences* (Amazon, IMDB and Yelp reviews).
- 2,748 sentences, labelled 1 (positive) or 0 (negative)
- Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

## Tools & Libraries
Python, Pandas, NumPy, NLTK / scikit-learn, Matplotlib, Seaborn

## Method
1. Loaded the three review files and combined them into one DataFrame
2. Preprocessing: lowercase, removed punctuation and special characters, removed extra whitespace, created a `clean_text` column
3. Explored the data: positive/negative examples and the most frequent words (stop-words removed)
4. Split into 80% train / 20% test (stratified)
5. Created a Bag-of-Words representation (unigrams + bigrams, `min_df=2`)
6. Trained Logistic Regression and Linear SVM
7. Evaluated with Accuracy, Precision, Recall, F1 and a confusion matrix
8. Tested the model with new sentences

## Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic