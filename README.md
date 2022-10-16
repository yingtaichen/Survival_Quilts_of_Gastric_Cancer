# Survival Quilts of Gastric Cancer

Use the [Survival Quilts](https://github.com/chl8856/SurvivalQuilts) to model and predict the survival curve of Gastric Cancer, and validate models by the China National Cancer Center Gastric Cancer database.

## Thesis information
* Title: Application of Survival Quilts for prognosis prediction of gastrectomy patients based on the Surveillance, Epidemiology, and End Results (SEER) database and China National Cancer Center Gastric Cancer (NCCGC) database
* Authors: Lulu Zhao, Penghui Niu, Wanqing Wang, Xue Han1, Xiaoyi Luan1, Dongbing Zhao, Jidong Gao, Yingtai Chen

## Description of the code
This code shows a simple example of the Survival Quilts on SEER/NCCGC dataset.
* [Code/](Code/): The source code implementation of Survival Quilts, and the training and testing process of the model.
* [Dataset/](Dataset/): The dataset files are stored in this path, and the training set and test set are divided according to the ratio of 8:2. In addition, data from the China National Cancer Center are not available.
* [Results/Models](Results/Models): The training model files are stored in this folder. These models are too large to upload. If necessary, contact the author to obtain.
* [environment.yaml](environment.yaml): Conda environment information. Using the command `conda env create -n <env_name> -f environment.yaml` to import conda environment information.
* [Tutorial.ipynb](Tutorial.ipynb): Give an example of the use process of code and the validation results of four models on SEER/NCC datasets.