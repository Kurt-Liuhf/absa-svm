# ABSA-SVM
## 环境配置
`conda env create -f absa_svm.yml`
## 运行步骤
### 依赖
在项目根目录下新建目录stanfordmodel，从[该网站](https://stanfordnlp.github.io/CoreNLP/history.html)下载stanfordcorenlp3.9.2解压到stanfordmodel目录，并修改”stanford_nlp.py“以及”file_utils.py“文件中相关路径。
### 数据处理
6个数据集均已经处理完毕，分别为laptop，rest，rest15，rest16，twitter，MAMS，如果想重新处理，请运行`python dataset.py`。
### 训练模型
文件“search_feature_comb.py”用于搜索模型的最佳特性和参数。因此，您可以在数据预处理后运行`python search_feature_comb.py`，参数可以使用`python search_feature_comb.py -h`查看。
例如训练laptop:
`python search_feature_comb.py -n 0 -p no_fc_rl -m 49 -s 0 -e 48 -T -f`
### 验证结果
运行 `python search_feature_comb.py -n 0 -p best -m 49 -V `

