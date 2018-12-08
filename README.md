# Quora Insincere Question Classification Problem

Problem and Data: 
Problem: https://www.kaggle.com/c/quora-insincere-questions-classification#evaluation

We are given a binary classification problem. We have to classify the questions as insincere or neutral(insincere being class 1 and neutral being class 0). 
Data provided is highly skewed towards one class. (6.2% for insincere question  and 93.8%  for neutral question).
Data consists of 1.31 M rows of questions and their classification of being 1 or 0. 
We are provided with pre trained embedding models which were allowed to be used for this problem.

Pre-Processing:  

As the data was highly skewed I explored 2 options: weighting classes and under sample majority class; Weighting examples was discarded as there was huge amount of data, we could work with subsampling to get good accuracy with high performance. After balancing the 2 classes, the textual data had to undergo 2 steps for vectorization. Firstly, all the unwanted and uninformative data such as html tags, non-letters etc. were removed. There was an option of keeping the stop words or removing stop words. Keeping the stop words help gain higher performance measure of models compared to removing the stop words. Then each question was vectorized to 300 dim vector using W2V pre trained model. Explored the option of using tf-idf and performance was better with w2v. The data is now transformed into a matrix of no_of_rows x 300. The data was then split into train, dev and test splits (Used 0.8-0.1-0.1 with cross validation as 0.1 splits yielded good amount of data for validation and test).

Modelling: 

This is now a binary classification problems in high dimensions. We cannot reduce dimensions in our problem as we are not aware of the w2v feature words being used. Any binary classifier is expected to perform well on this data. My exploration was to find the best such classifier which could be used for prediction. For each of the models used, I have tried to evaluate on val data and tuned the hyper parameters based on that. Each model was assessed for overfitting/underfitting and tuned to generalize. Each model performance was measured with F1 score as the key aspect of this problem lies in the minority class classification. Used 80% of data for training, 10% validation and 10% testing. This was sufficient data as we have been provided with huge dataset.

Running this File:
Download all the files and QIQC.ipynb to play arund the models which are built in Models.py 
It consists of preprocessed data in the inp directory. One can also find the data in data section of Kaggle competition link.
