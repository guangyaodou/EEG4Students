# EEG4Students

## Overview
Using Machine Learning and Deep Learning to predict cognitive tasks from electroencephalography (EEG) signals has been a fast- developing area in Brain-Computer Interfaces (BCI). However, dur- ing the COVID-19 pandemic, data collection and analysis could be more challenging. The remote experiment during the pandemic yields several challenges, and we discuss the possible solutions. This paper explores machine learning algorithms that can run efficiently on personal computers for BCI classification tasks. The results show that Random Forest and RBF SVM perform well for EEG classification tasks. Furthermore, we investigate how to conduct such BCI experiments using affordable consumer-grade devices to collect EEG-based BCI data. In addition, we have developed the data col- lection protocol, EEG4Students, that grants non-experts who are interested in a guideline for such data collection. 

In this paper, we experiment the following Machine Learning Algorithms
```
Gradient Boosting
LDA
Nearest Neighbors
AdaBoost
Random forest
Linear SVM
RBF SVM
Decision Tree
Shrinkage LDA
```

You can see our hyperparameters choice in our code:

https://github.com/GuangyaoDou/EEG4Students/blob/510b12ee3f4e382a2b651cec4d697dc5577f62b5/TCR.py#L68-L78

## Access to our datasets
Our datasets can be found in this google drive [TCR](https://drive.google.com/drive/folders/1HbFBL04t7gUllcBQuwuItcTMkksijT8o?usp=sharing)

More specifically, you can find out the TXT files (raw data) [Here](https://drive.google.com/drive/folders/1Y3dFue0w96EFr9SJ6ehoBOg1FUvrpJKW?usp=sharing)

You can get access to the CSV version of the datasets (preprocessed data)[Here](https://drive.google.com/drive/folders/16yTV8UrN1K6gjd7y4Vm4wAT_HAoAs_gt?usp=sharing)

## Installation and Evaluation
EEG4Students requires the following packages to replicate our results:

Data processing and visualization:
```
pandas
matplotlib
numpy
```

Machine learning:
```
sklearn
```

We highly recommend using a conda environment for package management.

First, you should create and activate a conda environment
```
conda create -n EEG4Students python = 3.9

conda activate EEG4Students
```

You should download the sklearn, pandas, and matplotlib:
```
conda install scikit-learn
```
```
conda install pandas
```
```
conda install matplotlib
```

# Preprocessing Data

Most EEG recording applications have a toolset to convert the recording files to TXT or CSV files. Afterward, we can pick the subset of data we plan to use for further analysis; we recommend starting with the absolute value of the EEG signals.

Here is an example of how we preprocess the EEG signals:

1. Convert .muse file to .csv files by using the default app.
One example would look like this in the command line:
```
muse-player -f 2_66_0530.muse -C 2_66_0530.csv
```
Side note: This is time consuming. It takes about 10 minutes to convert 110M data.

2. Use the Java code ([Process1.java](Process1.java)) to convert .csv files to .txt files.
For example, you can convert the .csv files to .txt files using the absolute values of the EEG signals:
```
java Process1.java <Sub1_Ses1_rwt_FFT.csv > Sub1_Ses1_rwt_FFT.txt;
```
3. As mentioned in the paper, we remove consecutive 1.4 seconds of anomaly in the datasets by executing [preprocess.m](preprocess.m) ([preprocess.m](preprocess.m) markes these anomalies to all 0s). 
You can adjust the variable "plateau_threshold" to detect noise. 

https://github.com/GuangyaoDou/EEG4Students/blob/d1cf8c07ea57d45ba22982c22d26b67068d93740/preprocess.m#L6

Also, remember to change the subject ids from your experiments accordingly:

https://github.com/GuangyaoDou/EEG4Students/blob/d1cf8c07ea57d45ba22982c22d26b67068d93740/preprocess.m#L4

We provide our raw and preprocessed datasets in the google drive. You are welcome to
use these datasets directly to replicate or reproduce our results. 

# Data Analysis
The directory of the project should look like this:
```
├── Process1.java
├── TCR.py
├── data_preprocess
│   └── tcr_plateau_removed_data
│       ├── tcr_subject_10_session_1.csv
│       ├── tcr_subject_10_session_2.csv
       ……
│       ├── tcr_subject_9_session_5.csv
│       └── tcr_subject_9_session_6.csv
├── data_raw
│   ├── 105_tcr_s1.txt
│   ├── 105_tcr_s2.txt
     …….
│   └── 82_tcr_s6.txt
├── output
│   └── tcr
│       ├── RandomForest_data_distribution.jpg
│       ├── accuracy_runtime_classifier.csv
│       ├── algorithm_comparison_each_subject.jpg
│       ├── subject_17_heatmap.jpg
│       └── subject_9_heatmap.jpg
└── preprocess.m
```

Remeber, create a folder called "data_preprocess" in the same level as the 
[TCR.py](TCR.py). Inside the "data_preprocess", create a folder called
"tcr_plateau_removed_data" that contains all the preprocessed csv version of
your datasets. Lastly, create an "output" folder and a "tcr" folder inside it.

In the [TCR.py](TCR.py), we first exclude 
In order to evaluate our results, just simply run
```
python TCR.py
```
Side note: the subject's ids here have limited meanings. The "subject ids"
in the codes refer to the participants' position index of this list:
https://github.com/GuangyaoDou/EEG4Students/blob/d1cf8c07ea57d45ba22982c22d26b67068d93740/TCR.py#L19

You can adjust the remote participants id so that the output will generate
the corresponding virtual participant's heatmap:
https://github.com/GuangyaoDou/EEG4Students/blob/d1cf8c07ea57d45ba22982c22d26b67068d93740/TCR.py#L22

If you have any questions, please reach out to us: guangyaodou@brandeis.edu