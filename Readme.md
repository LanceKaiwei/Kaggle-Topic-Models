#A Real-world Introduction to Topic Modeling and Text Mining

This is my final group project from data science class I took in spring 2015 at the University of Minnesota.  
It was taken from a past Kaggle competition, [Facebook Recruiting III - Keyword Extraction](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction)

##Preprocessing

There are embedded newline characters in the training set. To remove these we ran:

    tr '\n' ' ' < Train.csv | tr '\r' '\n' > Train.clean.csv

The original Train.csv file from Kaggle may not fit in memory. As the items are already randomized, you can just cut a number of lines from them.

    # 4,525,646 questions + the header
    head -n 4525647 Train.clean.csv > training.csv
    # Copy the header
    head -n 1 Train.clean.csv > testing.csv
    # Append the rest (1,508,547 questions) of the data
    tail -n 1508547 Train.clean.csv >> testing.csv

##Results

Results rounded to the nearest percent.
SGD Classifier: 11%
Latent Semantic Indexing & SGD Classifier: 13%
Latent Dirichlet Analysis & SGD Classifier: 15%
By exploring the forums at Kaggle it was discovered that the testing data set contained about 50% questions exactly the same as the training set. Armed with this knowledge most competitors were seeing approximately a 50% increase in scores compared to scores tested against a non-overlapping validation set. 

