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
