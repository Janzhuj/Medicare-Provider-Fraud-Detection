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

Another important preprocessing step was resampling the imbalanced data to make class ratio to 1:1. Data imbalance is a typical scenario in many business problems like fraud detection, spam filtering, rare disease discovery, hardware fault detection, etc. Generally, the minority/positive class is the main concern of the business and is aimed to achieve the best results. If imbalanced data is not treated beforehand, most predictions will correspond to the majority class and treat minority class features as noise in the data and ignore them. This will result in high bias in the classifier model and degrade the model performance. Resampling data is one of the most common approaches to deal with an imbalanced dataset, including undersampling and oversampling. Here, I used two resampling techniques: 1) SMOTE, which randomly oversamples data by choosing one of K instances to interpolate new synthetic instances; and 2) SMOTE+ENN, which is a hybrid technique. ENN is an undersampling technique that removes the nearest neighbors of each majority class instance. Integrating ENN with SMOTE can clean overlapping data points for each class distributed in sample space and optimize the performance of classifier models while avoiding overfitting.

![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/c5084b59-a48f-4eda-ab5e-b6b2e53dd2ea)
# Exploratory Data Analysis

