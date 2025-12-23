# Plant Seedlings Classification

This is a competition from Kaggle: [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification).

## Project Goal

- 每组至少要实现 3 种方法（注：两种模型+集成方法也算三种方法）
- 撰写一篇实验报告，包括但不限于实验目的、数据集介绍、数据预处理、模型设计与选择、预测结果
- 提交模型在 kaggle 上运行得到的分数截图
- 代码中给出必要的注释，README 文件中介绍代码运行方式

## Original Problem Statement

Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

![Sample Images](https://storage.googleapis.com/kaggle-media/competitions/seedlings-classify/seedlings.png)

We're hosting this dataset as a Kaggle competition in order to give it wider exposure, to give the community an opportunity to experiment with different image recognition techniques, as well to provide a place to cross-pollenate ideas.


### Acknowledgments
We extend our appreciation to the Aarhus University Department of Engineering Signal Processing Group for hosting the [original data](https://vision.eng.au.dk/plant-seedlings-dataset/).

### Citation

[A Public Image Database for Benchmark of Plant Seedling Classification Algorithms](https://arxiv.org/abs/1711.05458)

## Dataset Description
You are provided with a training set and a test set of images of plant seedlings at various stages of grown. Each image has a filename that is its unique id. The dataset comprises 12 plant species. The goal of the competition is to create a classifier capable of determining a plant's species from a photo. The list of species is as follows:

- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet


## File descriptions

- `train` - the training set, with plant species organized by folder
- `test` - the test set, you need to predict the species of each image
- `sample_submission.csv` - a sample submission file in the correct format

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
