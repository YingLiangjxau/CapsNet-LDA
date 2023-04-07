# CapsNet-LDA
This is the implementation for our paper:
>CapsNet-LDA: predicting lncRNA-disease associations using attention mechanism and capsule network based on multi-view data
## Step 
>step 1: calculate lncRNA similarity matrices and disease similarity matrices based on lncRNA-disease adjacency matrix (LDA.csv).  
step 2: obtain positive samples (e.g. PDCSLCS.csv) and negative samples (e.g. NDCSLCS.csv) based on similarity matrices and lncRNA-disease adjacency matrix.  
step 3: run autoencoder.py to get low-dimensional representations of samples under three views.  
step 4: run process.py to get integrated representations of samples and the labels of samples.  
step 5: run CapsNet-LDA.py to get the prediction results.  
