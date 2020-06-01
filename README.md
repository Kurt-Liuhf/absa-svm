# ABSA-SVM
This is the code of SVMs for Aspects Based Sentiment Analysis.

## Requirements
- python 3.x 
- sklearn 
- thundersvm 
- numpy 
- hyperopt
- matplotlib
- standfordcorenlp 

## Quick Start
### Data Preporcessing
You can run `python dataset.py` to process the two ABSA datasets. Notice that the path of the dataset is needed to be specified and the flag `is_preprocessed` should be `False` if you want to process the data from scratch.
### Training of the Model
The file `search_feature_comb.py` is used for searching the best features and parameters for our model. So you can run `python search_feature_comb.py` after the data preprocessing.
### Model Visualization
The codes for model visualization are stored in the folder named 'visualization'. `t-sne.py` is used for clustering model visualization by employing T-SNE. In addition, files named `svm_visualize_*.py` are for visualizing the trained SVMs in our model.
