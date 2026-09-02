# Machine Learning Journey

## About This Repository
Welcome to my Machine Learning workspace! As a graduate in Statistics and Computer Science, I created this repository to document my ongoing journey into the world of Machine Learning. 

This is a living, breathing project. Rather than just following tutorials, I am using this space to build algorithms from scratch, apply models to real datasets, and bridge the gap between pure statistical theory and applied data science. You will see this repository grow as I learn and implement new concepts.

## Journey Index
Here is how I am structuring my learning process. Click on any topic to explore the code and notes.

### 1. Data Preprocessing & EDA
  * *I created my own library of cleaning and caching processes during my [<img src="https://raw.githubusercontent.com/abdallahabukhalil/abdallahabukhalil/94a54f06f82e3e09834e4f23dfce14ae81d75499/assets/InsightHub%20Logo.svg" width="16"> InsightHub](https://github.com/InsightHubapp/InsightHub) project, with some enhancements that were not published in the project.*

  * [`Hot Encoding`](./The%20Journey/Preprocessing/Hot%20Encoding/main.py): Handling categorical variables and converting text data into a machine-readable format.

### 2. Supervised Learning (In Progress... 🚧)

  * **1. Regression**

    * **Linear Regression**
      * [`Main Model`](./The%20Journey/Supervised%20Learning/Regression/The%20Beginning,%20Linear%20Regression/Discover%20Data%20Science.py): My first steps in predictive modeling, applying Simple Linear Regression to analyze economic data.
    <br><br>

    * **Multiple Linear Regression**: *Theoretical concepts and mathematical foundations studied.*
    <br><br>

  * **2. Classification**

    * **Logistic Regression**
      * [`Main Binary Model`](./The%20Journey/Supervised%20Learning/Classification/Logistic%20Regression/main.py): A classification model built to predict employee retention and turnover using an HR analytics dataset.
      * [`Encoding Experiment`](./The%20Journey/Supervised%20Learning/Classification/Logistic%20Regression/salary_encoding_test.py): An experimental script testing the impact of different feature engineering techniques (Ordinal vs. One-Hot Encoding) on the model's accuracy. (Note: The result differs from case to case).
    <br><br>

    * **Multiclass Logistic Regression**: *Theoretical concepts and mathematical foundations studied.*
    <br><br>

    * **Decision Tree**
      * [`Main Model`](./The%20Journey/Supervised%20Learning/Classification/Decision%20Tree/main.py): The core Decision Tree classification model built to predict Titanic passenger survival. This script contains the complete pipeline, including advanced feature engineering (extracting deck levels, cabin sides, and calculating companion groups) prior to training the model.
      * [`Data Exploration`](./The%20Journey/Supervised%20Learning/Classification/Decision%20Tree/Decision_Tree.ipynb): A supporting Jupyter Notebook used for initial data visualization and understanding the dataset while writing the main script.
    <br><br>

    * **Support Vector Machine (SVM)**
      * [`Main Model`](./The%20Journey/Supervised%20Learning/Classification/Support%20Vector%20Machine/main.py): An SVC (Support Vector Classifier) model built to recognize handwritten digits, achieving an accuracy score of ~99.4% by tuning the `C` and `gamma` parameters.
      * [`Notebook Workspace`](./The%20Journey/Supervised%20Learning/Classification/Support%20Vector%20Machine/SVM.ipynb): A Jupyter Notebook used alongside the main script to inspect feature names, target labels, and test the model runs.
    <br><br>

    * **Random Forest**
      * [`Main Model`](./The%20Journey/Supervised%20Learning/Classification/Random%20Forest/main.py): A Random Forest model built to recognize digits. I adjusted settings like `n_estimators` and `max_depth` to reach an accuracy of ~96.39%.
    
### 3. Unsupervised Learning
  * *Soon...*

### 4. Model Evaluation & Tuning
  * *Soon...*

## Tools & Libraries
* **Language:** Python 3.x
* **Core Libraries:** Scikit-Learn, Pandas.
* **Concepts:** Statistical Modeling, Predictive Analytics.

## Author

<div align="center">

### Abdallah Abukhalil

*Data Analyst | Analytics Engineer | B.Sc in Stat & CS - Ain Shams University*

<a href="https://www.linkedin.com/in/abdallahabukhalil/">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.iconify.design/mdi:linkedin.svg?color=%23ffffff">
  <source media="(prefers-color-scheme: light)" srcset="https://api.iconify.design/mdi:linkedin.svg?color=%23000000">
  <img src="https://api.iconify.design/mdi:linkedin.svg?color=%23000000" width="32" alt="LinkedIn">
</picture>
</a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/abdallahabukhalil">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.iconify.design/mdi:github.svg?color=%23ffffff">
  <source media="(prefers-color-scheme: light)" srcset="https://api.iconify.design/mdi:github.svg?color=%23000000">
  <img src="https://api.iconify.design/mdi:github.svg?color=%23000000" width="32" alt="GitHub">
</picture>
</a>

</div>
