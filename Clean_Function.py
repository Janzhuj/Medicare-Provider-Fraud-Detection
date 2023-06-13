import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

def benef_new_feats(df):
    # Convert Date of Birth and Date of Death from String to Datetime format
    df['DOB'] = pd.to_datetime(df['DOB'] , format = '%Y-%m-%d')
    df['DOD'] = pd.to_datetime(df['DOD'],format = '%Y-%m-%d')
    
    # Create Age feature
    df['Age'] = round(((df['DOD'] - df['DOB']).dt.days)/365)    # Add feature 1
    df.Age.fillna(round(((pd.to_datetime('2009-12-01',format ='%Y-%m-%d')-df['DOB']).dt.days)/365),inplace=True)
    df['Age'] = df['Age'].astype(int)
    
    # Add column If_Alive and set value=1 if DOD value is null
    df['If_Alive'] = df['DOD'].apply(lambda val:1 if val != val else 0)      # Add feature 2

    # Create columns for total anual amounts of reimbursement and deductible      # Add feature 3
    #df['Tot_Risk'] = df['IPAnnualReimbursementAmt'] + df['OPAnnualReimbursementAmt']
    df['Tot_Reimbursed_Amt'] = df['IPAnnualReimbursementAmt'] + df['OPAnnualReimbursementAmt']
    df['Tot_Deductible_Amt'] = df['IPAnnualDeductibleAmt'] + df['OPAnnualReimbursementAmt']


    # Add a column to present patient risk scores with chronic diseases   ---Add feature 4
    df['Risk_score'] = df.iloc[:,10:21].sum(axis=1)


def benef_dummy_encode(df):
    # The columns for label encoding 
    col_list = ['Gender', 'Race']
    # Encoding Nominals
    df_encode = pd.get_dummies(df, columns=col_list)
    return df_encode

    
def benef_label_encode(df):

    # The columns for label encoding 
    col_list = ['Gender', 'Race', 'RenalDiseaseIndicator',
                'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
                'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
                'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
                'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                'ChronicCond_stroke']
    
    # Label Encoding each column 
    for col in col_list:
        label_encoder = preprocessing.LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])        
        

# https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-grouped-bars
# Add percentages on top of grouped bars function
def annotation_with_hue(ax, feature, Number_of_categories, hue_categories, x_pos=0, y_pos=0, fontsize = 12):
    a = [p.get_height() for p in ax.patches]
    patch = [p for p in ax.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
            x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 + x_pos
            y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() +y_pos
            ax.annotate(percentage, (x, y), size = 12)

def annotation_without_hue(ax, feature, x_pos=0, y_pos=0, fontsize = 12):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 + x_pos # To adjust the position of the percentage value
        y = p.get_y() + p.get_height() +y_pos # To adjust the position of the percentage value
        ax.annotate(percentage, (x, y), size = 12)

def annotation_index(ax, feature, Number_of_categories, hue_categories, x_pos=0, y_pos=0, fontsize = 10):
	a = [p.get_height() for p in ax.patches]
	patch = [p for p in ax.patches]
	for i in range(Number_of_categories):
		total = feature.value_counts().loc[i]
		for j in range(hue_categories):
			percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
			x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 + x_pos
			y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() +y_pos
			ax.annotate(percentage, (x, y), size = 10)


def in_out_null_values(df):

	# Remove columns with high missing values in both inpatient and outpatient datasets.
	df.drop(['ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6'], axis=1, inplace=True)    
    
	# Fill 0 for missing deductible amounts. It assumes that no deductible was paid

	df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(0)
  
	# Fill None if no codes were listed

	col_list = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
			   'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
			   'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
			   'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
			   'ClmProcedureCode_3','ClmAdmitDiagnosisCode']

	for col in col_list:
		df[col] = df[col].fillna('None')

	# Fill None for missing doctor code values

	for col in ('AttendingPhysician','OperatingPhysician','OtherPhysician'):
		df[col] = df[col].fillna('None')    

def in_new_feats(df):

	# Convert claim start/end and admission/discharge date dtype from String to Datetime format

	df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], format = '%Y-%m-%d')
	df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'], format = '%Y-%m-%d')
	df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], format = '%Y-%m-%d')
	df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], format = '%Y-%m-%d')

	# Create Hospital_Stay and Claim_Duration features and changing
	# dtype to int
    
	df['Claim_Duration'] = round(((df['ClaimEndDt'] - df['ClaimStartDt']).dt.days))  # Add feature 5
	df['Hospital_Duration'] = round(((df['DischargeDt'] - df['AdmissionDt']).dt.days))  # Add feature 6
	df['Claim_Duration'] = df['Claim_Duration'].astype(int)
	df['Hospital_Duration'] = df['Hospital_Duration'].astype(int)



	# Breaking down claim made month and year

	df['Claim_Start_Year'] = df['ClaimStartDt'].dt.year  # Add feature 7
	df['Claim_Start_Month'] = df['ClaimStartDt'].dt.month   # Add feature 8

	# Total claim amount

	df['Total_Claim_Amt'] = (df['InscClaimAmtReimbursed']+df['DeductibleAmtPaid'])  # Add feature 9

	# Insurance covered percentages

	#df['Insurance_Covered_Per'] = round((df['InscClaimAmtReimbursed']/df['Total_Claim_Amt'])*100, 2)  # Add feature 10
	#df['Insurance_Covered_Per'] = df['Insurance_Covered_Per'].fillna(0)
    
	# Add a new column if_inpatient --------Add feature 11
	df['if_inpatient']=1

	# Drop ClaimStartDt, ClaimEndDt, AdmissionDt and DischargeDt columns
	df.drop(['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt'], axis=1, inplace=True)



def out_new_feats(df):

	# Changing dtype for claim start/end and admission/discharge date
	# columns to datetime

	df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'])
	df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'])

	# Creating Hospital_Stay and Claim_Duration features and changing
	# dtype to int

	df['Claim_Duration'] = round(((df['ClaimEndDt'] - df['ClaimStartDt']).dt.days))
	df['Claim_Duration'] = df['Claim_Duration'].astype(int)

	# Breaking down claim made month and year

	df['Claim_Start_Year'] = df['ClaimStartDt'].dt.year
	df['Claim_Start_Month'] = df['ClaimStartDt'].dt.month

	# Total claim amount

	df['Total_Claim_Amt'] = (df['InscClaimAmtReimbursed']+df['DeductibleAmtPaid'])

	# Insurance covered percentages

	#df['Insurance_Covered_Per'] = round((df['InscClaimAmtReimbursed']/df['Total_Claim_Amt'])*100, 2)
	#df['Insurance_Covered_Per'] = df['Insurance_Covered_Per'].fillna(0)
    
	# Add a new column if_inpatient --------Add feature 11
	df['if_inpatient']=0

	# Drop ClaimStartDt, ClaimEndDt columns
	df.drop(['ClaimStartDt','ClaimEndDt'], axis=1, inplace=True)
    
    
def benef_group_feats(df):
	# Count of nuique number for each beneficiary

	benef_group = df[['BeneID','ClaimID', 'Provider','AttendingPhysician','OperatingPhysician','OtherPhysician', 'if_inpatient']]\
    .groupby(['BeneID','if_inpatient']).agg(['nunique']).reset_index()

	benef_group.columns=['Bene_ID', 'if_inpatient', 'Claim_nunique', 'Provider_nunique', 'Att_nunique', 'Op_nunique', 'Othr_nunique']

	benef_group['Physician_nunique'] = benef_group['Att_nunique'] + benef_group['Op_nunique'] + benef_group['Othr_nunique']
   
	return benef_group


    

def provider_group_feats(df):
	# Count of nuique number for each Provider

	provider_group = df[['Provider','BeneID','ClaimID','AttendingPhysician','OperatingPhysician','OtherPhysician', 
                         'ClmAdmitDiagnosisCode','ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',
                         'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9',
                         'ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3',
                         'if_inpatient']].groupby(['Provider','if_inpatient']).agg(['nunique']).reset_index()
    
	provider_group_2 = df[['Provider','Claim_Duration','Hospital_Duration','InscClaimAmtReimbursed','DeductibleAmtPaid',
                           'SC_Reimbursed_mean','diff_SC_Reimbursed',
                           'Age','Risk_score','if_inpatient']].groupby(['Provider','if_inpatient']).agg(['mean','std']).reset_index()

	provider_group_3 = df[['Provider','Claim_Duration','Hospital_Duration','InscClaimAmtReimbursed','DeductibleAmtPaid','Risk_score',
                           'Age','if_inpatient']].groupby(['Provider','if_inpatient']).agg(['max']).reset_index()
  

	provider_group = provider_group.merge(provider_group_2, how='inner',on=['Provider','if_inpatient'])\
    .merge(provider_group_3,how='inner',on=['Provider','if_inpatient'])

	provider_group.columns=['Provider','if_inpatient','Bene_nunique', 'Claim_nunique', 'Att_nunique','Op_nunique','Othr_nunique',
                            'Diag_nunique','CD1','CD2','CD3', 'CD4','CD5','CD6','CD7', 'CD8','CD9','CD10','CP1','CP2','CP3',
                            'Claim_Duration_mean','Claim_Duration_std','Hospital_Duration_mean','Hospital_Duration_std',
                            'Reimbursed_mean','Reimbursed_std','Deductible_mean','Deductible_std',
                            'SC_Reimbursed_mean','SC_Reimbursed_std','diff_SC_Reimbursed_mean','diff_SC_Reimbursed_std',
                            'Age_mean','Age_std','Risk_score_mean','Risk_score_std',
                            'Claim_Duration_max','Hospital_Duration_max','Reimbursed_max','Deductible_max','Risk_score_max','Age_max']

	provider_group['Physician_Cnt'] = provider_group['Att_nunique'] + provider_group['Op_nunique'] + provider_group['Othr_nunique']
    
	provider_group['Diag_Code_Cnt'] = provider_group['CD1'] + provider_group['CD2'] + provider_group['CD3'] +provider_group['CD4']+\
						provider_group['CD5'] + provider_group['CD6'] +provider_group['CD7'] + provider_group['CD8'] + \
						provider_group['CD9'] + provider_group['CD10']

	provider_group['Proc_Code_Cnt'] = provider_group['CP1'] + provider_group['CP2'] + provider_group['CP3']
    
	provider_group['Insurance_Covered_Per'] =round((provider_group['Reimbursed_mean']/
                                                    (provider_group['Reimbursed_mean']+provider_group['Deductible_mean']))*100, 2)

	return provider_group

