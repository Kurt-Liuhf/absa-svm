# ABSA_RLSVM

## 1：数据集信息

### 我们与baselines数据集的异同

每列中相同数字代表使用的数据集相同

| Models        | Laptop14 | Restaurant14 | Restaurant15 | Restaurant16 | Twitter | MAMS |
| ------------- | -------- | ------------ | ------------ | ------------ | ------- | ---- |
| DPL-BERT      | 1        | 1            | /            | /            | /       | /    |
| dotGCN-BERT   | 2        | 2            | 1            | 1            | 1       | 1    |
| TF-BERT       | 1        | 1            | /            | /            | 2       | /    |
| TGCN-BERT     | 2        | 3            | 2            | 2            | 1       | 1    |
| SARL          | 2        | 2            | 1            | 1            | 1       | /    |
| LSA-X-DeBERTa | 2        | 2            | 3            | 3            | /       | 2    |
| Ours          | 3        | 2            | 4            | 4            | 1       | 1    |

> DPL-BERT: Yiming Zhang, Min Zhang, Sai Wu, and Junbo Zhao. 2022. Towards Unifying the Label Space for Aspect-and Sentence-based Sentiment Analysis. In Findings of the Association for Computational Linguistics: ACL 2022. 20–30.
>
> dotGCN-BERT: Chenhua Chen, Zhiyang Teng, Zhongqing Wang, and Yue Zhang. 2022. Discrete opinion tree induction for aspect-based sentiment analysis. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2051–2064.
>
> TF-BERT: Mao Zhang, Yongxin Zhu, Zhen Liu, Zhimin Bao, Yunfei Wu, Xing Sun, and Linli Xu. 2023. Span-level Aspect-based Sentiment Analysis via Table Filling. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 9273–9284.
>
> TGCN-BERT: Yuanhe Tian, Guimin Chen, and Yan Song. 2021. Aspect-based sentiment analysis with type-aware graph convolutional networks and layer ensemble. In Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: human language technologies. 2910–2922.
>
> SARL-RoBERTa, SARL-RoBERTa_{large} and SARL-DeBERTa: Bo Wang, Tao Shen, Guodong Long, Tianyi Zhou, and Yi Chang. 2021. Eliminating sentiment bias for aspect-level sentiment classification with unsupervised opinion extraction. arXiv preprint arXiv:2109.02403 (2021).
>
> LSA-X-DeBERTa: Heng Yang and Ke Li. 2021. Improving Implicit Sentiment Learning via Local Sentiment Aggregation. arXiv preprint arXiv:2110.08604 (2021).

### 我们使用的数据集来源：

Laptop14[4]，Restaurant14[4]

数据仓库：https://github.com/Pinlong-Zhao/SDGCN/tree/master

Restaurant15[1]，Restaurant16[1]：和论文中数据不一致是因为去除了重复数据
数据仓库：https://github.com/kevinscaria/InstructABSA/tree/main/Dataset

Twitter[2]:
数据仓库：http://goo.gl/5Enpu7

MAMS[3]：
数据仓库：https://github.com/siat-nlp/MAMS-for-ABSA/tree/master/data/MAMS-ATSA/raw

> [1]Scaria K, Gupta H, Goyal S, et al. Instructabsa: Instruction learning for aspect based sentiment analysis[J]. arXiv preprint arXiv:2302.08624, 2023.
>
> [2] Li Dong, Furu Wei, Chuanqi Tan, Duyu Tang, Ming Zhou, and Ke Xu. 2014. Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 49–54, Baltimore, Maryland. Association for Computational Linguistics.
>
> [3] Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, and Min Yang. 2019. A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6280–6285, Hong Kong, China. Association for Computational Linguistics.
>
> [4]Zhao, Pinlong, Linlin Hou, and Ou Wu. "Modeling sentiment dependencies with graph convolutional networks for aspect-level sentiment classification." Knowledge-Based Systems 193 (2020): 105443.

## 2：代码描述

### **环境配置**

```
conda env create -f absa_svm.yml
```

### **运行步骤**

#### **依赖**

在项目根目录下新建目录stanfordmodel，从[该网站](https://stanfordnlp.github.io/CoreNLP/history.html)下载stanfordcorenlp3.9.2解压到stanfordmodel目录，并修改”stanford_nlp.py“以及”file_utils.py“文件中相关路径。

#### **数据处理**

6个数据集均已经处理完毕，分别为laptop，rest，rest15，rest16，twitter，MAMS，如果想重新处理，请运行`python dataset.py`。（注意：仓库里有一个laptopsours，这个是论文中所使用的数据集。在rebuttal阶段，重新跑了第一个表中的第“2”类数据集，即仓库中的laptop数据集和dotGCN-BERT、TGCN-BERT等一致）

#### **训练模型**

文件“search_feature_comb.py”用于搜索模型的最佳特征和参数。因此，您可以在数据预处理后运行`python search_feature_comb.py`，参数可以使用`python search_feature_comb.py -h`查看。

例如训练laptop:

```python
python search_feature_comb.py -n 0 -p no_fc_rl -m 49 -s 0 -e 48 -T -g 0
```

-n 代表对第“0”个数据集进行超参数搜索（laptop，rest，twitter，rest15，rest16，MAMS分别对应着0~5），-p 代表结果保存的目录，-m 代表数据集被分成的子数据集数量（可以看数据集`parsed_data`目录），-s 和 -e代表了运行子数据集的开始和结束索引（可以在不同机器上运行，例如分成2份[0, 24], [25,48]在两台服务器上运行），-T 代表使用Thundersvm， -g 代表使用的显卡索引

可以先使用上面的命令跑一遍所有的子数据集，然后再运行下面的命令：

```python
python search_feature_comb.py -n 0 -p fc_no_rl -m 49 -s 0 -e 48 -T -f -g 0
```

-f 代表使用自动特征构造

然后将两次运行的结果进行对比，选取较好的留下，放在best49目录中（49需要根据子数据集的数量改变）。

然后执行下面的命令进行RL：

```python
python search_feature_comb.py -n 0 -p fc_rl -m 49 -s 0 -e 48 -T -R -g 0
```

运行完之后，与best49目录下的结果进行对比，替换best49目录下的不好的结果。

每个数据集`optimal_results`目录下都有一个`f1.ipynb`文件，可以运行它得到结果整合后的Acc和F1。

#### **验证结果**

运行 `python search_feature_comb.py -n 0 -p best -m 49 -V `