## 1. Introduction ##
This branch stores the code that implements Multiple-Berts of our paper.
Please notice that most of the code (more than 99%) comes from [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)

## 2. How to run ##
1. The command `python3 train.py --dataset restaurant --CID $1 --num_epoch $2` is used for training the bert-models, where `$1` refers to the cluster id and `$2` refers to number of epoch (15 - 20 is enough) that bert trains. The evaluation of *inference* is already included in the `train.py`.
2. After finish training the model, run `f1.py` to get the evaluation metrics.
