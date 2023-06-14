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
Before discussing the extensive data analysis performed on the claims data, I would like to explain how the data was preprocessed. First, handling missing values in the data. Missing information should be imputed accordingly. we also can enrich data informantion by creating new features. For example, in benificiaries dataset, the date of death was missing if a patient was alive,  and in inpatient dataset. I create two new features 'if-alive' and 'Age' using the date of birth and death. 
Next, categorical data was label-encoded for uniform and efficient preprocessing.

I decided to keep outliers in the data because they could provide key fraud indicator information. These outliers could represent transactions where actual fraud is being committed. For this reason, the data was robustly scaled before modeling. Another important preprocessing step was upsampling the data to reduce the imbalance and bring the fraud label ratio to 1:1. The data was processed using two upsampling techniques: SMOTE, which creates data randomly between two data points, and BorderlineSMOTE, which creates data along the decision boundary between the two classes. The performances of both techniques were compared.
![image](https://github.com/Janzhuj/Medicare-Provider-Fraud-Detection/assets/99841253/c5084b59-a48f-4eda-ab5e-b6b2e53dd2ea)
# Exploratory Data Analysis

