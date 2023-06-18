# Medicare-Provider-Fraud-Detection
The US spends over $4 trillion per year on healthcare, which is largely conducted by private providers and reimbursed by insurers. Overbilling, waste, and fraud are major concerns in the healthcare system and are estimated by the US Federal Bureau of Investigation to account for 3% to 10% of overall spending. Provider fraud occurs when healthcare providers misreport their claims to receive higher payments. In this work, I use large-scale claims data from Medicare, the US federal health insurance program for elderly adults and the disabled, to perform an exploratory data analysis. My goal is to identify patterns consistent with fraud or overbilling and understand characteristics associated with high suspiciousness of fraud. My proposed approach for fraud detection is supervised and involves developing a broad range of classifiers on labeled training data to help identify potential providers who might overbill insurers. I also provide reasoning and interpretable insights into the potentially suspicious behavior of flagged providers by analyzing important features. Additionally, I apply SMOTE techniques to handle imbalanced data and improve 1-fold lift recall for the minority/fraud class. The results can be used to guide investigations and auditing of suspicious providers for both public and private health insurance systems.
## Healthcare Provider Fraud overview
Healthcare fraud and abuse take many forms. Some of the most common types that providers deceive insurers through claims procedures are listed below:
* Phantom Billing. Providers billing for services not provided.
* Unnecessary Services. Providers administering (more) tests and treatments that are not medically necessary.
* Upcoding. Providers administering more expensive tests and equipments.
* Multiple-billing. Providers multiple-billing for services rendered.
* Unbundling. Providers unbundling or billing separately for laboratory tests to get higher reimbursements.
* False price reporting. Providers charging more than peers for the same services.
## Medicare claims dataset
The dataset in this project comes from Kaggle's website - [Healthcare Provider Fraud Dection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) by Rohit Anand Gupta. The dataset consists of four sub-datasets, as listed below.
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/d967824e-a734-47cc-aee7-a16661925770)
## Workflow
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/6b7a5707-10f0-4a5a-8682-1b3ea0d71c41)
## Data Preporcessing
Before discussing the extensive data analysis performed on the claims data, I would like to explain how the data was preprocessed. First, I handled missing values in the data by imputing them accordingly. I also enriched the data by creating new features. For example, in the beneficiaries dataset, the date of death was missing if a patient was alive. I created two new features: ‘if-alive’ and ‘Age’ using the date of birth and death.

Next, I label-encoded categorical data and scaled numerical data for uniform and efficient preprocessing. Here, I used RobustScaler to scale features because it is robust to outliers. The outliers in the data were kept, as they could provide important fraud indicator information and sometimes represent actual fraud transactions were being committed sometim.

Another important preprocessing step was resampling the imbalanced data to make class ratio to 1:1. Data imbalance is a typical scenario in many business problems like fraud detection, spam filtering, rare disease discovery, hardware fault detection, etc. In general, the minority/positive class is the main concern and is aimed to achieve the best results. If imbalanced data is not treated beforehand, most predictions will correspond to the majority class and treat minority class features as noise in the data and ignore them. This will result in high bias in the classifier model and degrade the model performance. Resampling data is one of the most common approaches to deal with an imbalanced dataset, including undersampling and oversampling. Here, I used two resampling techniques: 1) SMOTE, which randomly oversamples data by choosing one of K instances to interpolate new synthetic instances; and 2) SMOTE+ENN, which is a hybrid technique. ENN is an undersampling technique that removes the nearest neighbors of each majority class instance. Integrating ENN with SMOTE can clean overlapping data points for each class distributed in sample space and optimize the performance of classifier models while avoiding overfitting.

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/c5084b59-a48f-4eda-ab5e-b6b2e53dd2ea" width=80% height=80%>

## Exploratory Data Analysis
### 1. Class Labels
Let’s take a look at the provider class dataset. It is an imbalanced dataset where the target variable, “Potential Fraud,” has 90.6% of providers not fraudulent and 9.6% of providers fraudulent. Notice, at the model evlaluation section, we will use metrics like precision, recall, F1-score rather than the accuracy metric to understand the performance of the classifiers for correctly determining which provider is fraudulent, since class distribution is high skewed, the accuracy metric is biased and not preferable to evlauate the model performance.
<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/8983b33c-d1ba-4a33-8597-26fade422881" width=60% height=60%>

### 2. Beneficiary Basic Information Study

Before we proceed, let’s take a look at our patients. We notice that the majority of our beneficiaries belong to race 0. The percentage of gender 0 is larger than that of gender 1. Fifty percent of beneficiaries fall between the ages of 68 and 82 years old, with some outliers below 47 who are disabled. The chronic disease risk scores display a bell-like shape with a slight right tail. There is a positive relationship between the mean annual reimbursement amount and the chronic disease risk scores of beneficiaries.

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/bf76c4be-352d-455f-bef9-26d3a52f6fe6" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/6046aa3a-e186-4d31-af4e-3ce099989426" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/239c23cc-271f-4a15-8c9d-5f24250c085c" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/5edc2cdd-83be-4651-93fb-0322a3da170a" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/1fb613ce-a4b8-433c-83aa-fd229e5b0701" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/9c6c166c-feea-415c-8643-593f87bd09ec" width=60% height=60%>

### 3. fraud VS. Non-fraud Providers Study
Now, let’s explore the characteristics of fraudulent providers. We know that if the median line of a box plot lies outside the box of a comparison box plot, then there is likely to be a difference between the two groups. When comparing the box plots of characteristics, like claim numbers, beneficiary numbers, diagnose code numbers, and hospital duration variation, we find that there are obvious differences between fraudulent and non-fraudulent providers. 
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/a63335c5-ded3-4185-9f79-75dbe7087ae1)
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/dd327004-7eb8-49bd-a058-afc18c4aec8e)
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/c75aa2bf-0a27-481e-ac23-1232d68fe0ba)
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/05c862d4-683a-4af8-b367-d79febcf4032)

![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/edf782f1-6ece-4a76-8fc7-518f95492f37)
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/df28a561-8d1c-4fa6-a2cb-7c088c3ad9e7)

## Data Modeling

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/a5c0ae26-62c8-4da6-8927-2d5d9c228e00" width=60% height=60%>

### 1. Feature Selection
In this section, I will pipleline classifiers and feature selection method to build the base algorithms.  I choose RFE method for feature selection, and tune the hyperparameters of RFE to obtian the best number of features, as showed in the below,  the number 41 is the best number of features for both logistic regression and random forest algorithms.
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/bd7e8b12-1e8c-4509-a753-f58bc1737786)

### 2. Explore Base Algorithm
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/013ccd09-39b4-467a-a957-39a7b5b59c19)

### 3. Base Algorithm comparison
Logistic Regression, Support Vector Machine, and Random Forest classifiers generally perform well. While the mean performance of the Gradient Boosting classifier appears good, its F1_weighted score has a relatively larger variance compared to the others. This may result in less stable results.

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/6cd8af96-a576-4d05-9a55-99c53bf07a60" width=60% height=60%>

## Optimization Models
### 1. Tune Hyperparameters
<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/6fa5766a-5f8c-4d05-9d75-2c716b674ede" width=60% height=60%>

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/2f04f20e-ec8f-4cd9-a995-4d52e9fb1169" width=60% height=60%>

### 2 Learning Curves
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/797fb576-9a63-40e7-a9e7-8839524b3d8f)

### 3. Evaluation Metrics for Imbalanced Data
A comparative analysis was performed on the dataset using four classifier models: Logistic Regression, Random Forest, Linear Support Vector Machine, and Gradient Boosting. As discussed earlier, we will ignore the accuracy metric to evaluate the performance of the classifiers on this imbalanced dataset. Here, we are more interested in identifying potential fraudulent providers. Therefore, we will focus on metrics such as precision, recall, and F1-score to understand the performance of the classifiers in correctly determining which providers will commit fraud in the insurance claim process.
<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/54bcb2d7-53e2-49d6-a765-de212f4d0c03" width=60% height=60%>

![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/84c1f0d6-e33d-4072-b99f-a05ccb83f3bb)

### 4. Maximize Minority’s Recall Scores
From the above, it can be seen that all 4 classifier models were not able to generalize well on the minority class compared to the majority class. To minimize fraudulent providers incorrectly classified as non-faudulent providers (False Negative), we can use SMOTE related technolege to imporve recall scores of the minority class. As shown in below, after resampling, a clear surge in Recall is seen on the test data.

<img src="https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/aa10e721-aefe-40c1-9338-eac0d527e81f" width=80% height=80%>

### 5. Feature Importances
Machine Learning models are often black boxes, making their interpretation difficult. SHAP values (SHapley Additive exPlanations) is one method used to explain how each feature affects the model. The following plots show the main features affecting the prediction of the observations and the magnitude of the SHAP value for each feature. Here, I used shap__value[0], the SHAP values for class 0. The SHAP values for class 1 are symmetrical to them.

![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/ae920f28-fc1f-487e-8822-9d43c31d4bd3)

![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/7da6b9d9-3bdd-4eb8-bed5-53754a78279d)

## Conclusion
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/37255c3f-0780-486b-bb49-02317667a4bf)





