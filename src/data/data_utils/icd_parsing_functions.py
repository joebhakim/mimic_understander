import pandas as pd
import numpy as np


def get_code_categories(icd_df):
    # creates columns for icd categories

    # input: icd_df should have only patient_id, age (for charlson), and columns whose names start with DIA
    binned_icd_df = icd_df.copy()
    icd_only_df = binned_icd_df.drop('age',axis=1)
    icd_melt_sparse = icd_only_df.melt(id_vars='patient_id', value_name='icd10')
    icd_melt = icd_melt_sparse[~icd_melt_sparse['icd10'].isna()]

    codes_dict_score_one = {
        'myocardial infarction':
        ['^I21.*', '^I22.*', 'I25.2'],
        'congestive heart failure':
        ['I09.9',  'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5', 'I42.6', 'I42.7', 'I42.8', 'I42.9',
         '^I43.*', '^I50.*', 'P29.0'],
        'peripheral vascular disease':
        ['^I70.*', '^I71.*', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9',
         'Z95.8', 'Z95.9', ],
        'cerebrovascular accident':
        ['^G45.*', '^G46.*', 'H34.0', '^I60.*', '^I61.*', '^I62.*', '^I63.*', '^I64.*', '^I65.*', '^I66.*',
         '^I67.*', '^I68.*', '^I69.*', ],
        'dementia':
        ['^F00.*', '^F01.*', '^F02.*', '^F03.*', 'F05.1', '^G30.*', 'G31.1', ],
        'COPD':
        ['I27.8', 'I27.9', '^J40.*', '^J41.*', '^J42.*', '^J43.*', '^J44.*', '^J45.*', '^J46.*', '^J47.*',
         '^J60.*', '^J61.*', '^J62.*', '^J63.*', '^J64.*', '^J65.*', '^J66.*', '^J67.*', 'J68.4', 'J70.1',
         'J70.3', ],
        'connective tissue / rheumatic disease':
        ['^M05.*', '^M06.*', 'M31.5', '^M32.*', '^M33.*',
         '^M34.*', 'M35.1', 'M35.3', 'M36.0', ],
        'peptic ulcer disease':
        ['^K25.*', '^K26.*', '^K27.*', '^K28.*', ],
        'mild liver disease':
        ['^B18.*', 'K70.0', 'K70.1', 'K70.2', 'K70.3', 'K70.9', 'K71.3', 'K71.4', 'K71.5', 'K71.7', '^K73.*', '^K74.*',
         'K76.0', 'K76.2', 'K76.3', 'K76.4', 'K76.8', 'K76.9', 'Z94.4', ],
        'diabetes without complications':
        ['E10.0', 'E10.1', 'E10.6', 'E10.8', 'E10.9', 'E11.0', 'E11.1', 'E11.6', 'E11.8', 'E11.9', 'E12.0',
         'E12.1', 'E12.6', 'E12.8', 'E12.9', 'E13.0', 'E13.1', 'E13.6', 'E13.8', 'E13.9', 'E14.0', 'E14.1',
         'E14.6', 'E14.8', 'E14.9']
    }
    codes_dict_score_two = {
        'diabetes with complications':
        ['E10.2', 'E10.3', 'E10.4', 'E10.5', 'E10.7', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.7', 'E12.2',
         'E12.3', 'E12.4', 'E12.5', 'E12.7', 'E13.2', 'E13.3', 'E13.4', 'E13.5', 'E13.7', 'E14.2', 'E14.3',
         'E14.4', 'E14.5', 'E14.7', ],
        'hemiplegia/paraplegia':
        ['G04.1', 'G11.4', 'G80.1', 'G80.2', '^G81.*', '^G82.*', 'G83.0', 'G83.1', 'G83.2', 'G83.3', 'G83.4',
         'G83.9', ],
        'moderate to severe chronic kidney disease':
        ['I12.0', 'I13.1', 'N03.2', 'N03.3', 'N03.4', 'N03.5', 'N03.6', 'N03.7', 'N05.2', 'N05.3', 'N05.4',
         'N05.5', 'N05.6', 'N05.7', '^N18.*', '^N19.*', 'N25.0', 'Z49.0', 'Z49.1', 'Z49.2', 'Z94.0', 'Z99.2', ],
        'localized solid tumor':
        ['^C00.*', '^C01.*', '^C02.*', '^C03.*', '^C04.*', '^C05.*', '^C06.*', '^C07.*', '^C08.*', '^C09.*',
         '^C10.*', '^C11.*', '^C12.*', '^C13.*', '^C14.*', '^C15.*', '^C16.*', '^C17.*', '^C18.*', '^C19.*',
         '^C20.*', '^C21.*', '^C22.*', '^C23.*', '^C24.*', '^C25.*', '^C26.*', '^C30.*', '^C31.*', '^C32.*',
         '^C33.*', '^C34.*', '^C37.*', '^C38.*', '^C39.*', '^C40.*', '^C41.*', '^C43.*', '^C45.*', '^C46.*',
         '^C47.*', '^C45.*', '^C46.*', '^C47.*', '^C48.*', '^C49.*', '^C50.*', '^C51.*', '^C52.*', '^C53.*',
         '^C54.*', '^C55.*', '^C56.*', '^C57.*', '^C58.*', '^C60.*', '^C61.*', '^C62.*', '^C63.*', '^C64.*',
         '^C65.*', '^C66.*', '^C67.*', '^C68.*', '^C69.*', '^C70.*', '^C71.*', '^C72.*', '^C73.*', '^C74.*',
         '^C75.*', '^C76.*', '^C81.*', '^C82.*', '^C83.*', '^C84.*', '^C85.*', '^C88.*', '^C90.*', '^C91.*',
         '^C92.*', '^C93.*', '^C94.*', '^C95.*', '^C96.*', '^C97.*']
    }
    codes_dict_score_three = {
        'severe liver disease': ['I85.0', 'I85.9', 'I86.4', 'I98.2',
                                 'K70.4', 'K71.1', 'K72.1', 'K72.9', 'K76.5', 'K76.6', 'K76.7']
    }
    codes_dict_score_four = {
        'metastatic solid tumor': ['^C77.*', '^C78.*', '^C79.*',
                                   '^C80.*'],
        'HIV/AIDS': ['^B20.*', '^B21.*', '^B22.*', '^B24.*']
    }

    all_codes = {**codes_dict_score_four, **codes_dict_score_three, **codes_dict_score_two, **codes_dict_score_one}

    # replace codes with desciptions, so we can pivot later and get feature names
    named_icd_melt = icd_melt.copy()
    for dx_name, dx_code_list in all_codes.items():
        named_icd_melt.loc[:, 'icd10'] = named_icd_melt['icd10'].\
            replace(to_replace=dx_code_list, value=dx_name, regex=True)

    # codes that arent replaced should be null
    named_icd_melt.loc[~named_icd_melt['icd10'].isin(list(all_codes.keys())), 'icd10'] = np.nan
    wide_icd = pd.get_dummies(named_icd_melt[['patient_id','icd10']], columns=['icd10'])
    
    return wide_icd


def charlson_calc(icd_df):

    charlson_df = icd_df.set_index(['patient_id'])

    score_one_list = ['^I21.*', '^I22.*', 'I25.2',  # myocardial infarction
                      # congestive heart failure
                      'I09.9',  'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5', 'I42.6', 'I42.7', 'I42.8', 'I42.9',
                      '^I43.*', '^I50.*', 'P29.0',
                      # peripheral vascular disease
                      '^I70.*', '^I71.*', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9',
                      'Z95.8', 'Z95.9',
                      # cerebrovascular accident
                      '^G45.*', '^G46.*', 'H34.0', '^I60.*', '^I61.*', '^I62.*', '^I63.*', '^I64.*', '^I65.*', '^I66.*',
                      '^I67.*', '^I68.*', '^I69.*',
                      # dementia
                      '^F00.*', '^F01.*', '^F02.*', '^F03.*', 'F05.1', '^G30.*', 'G31.1',
                      # COPD
                      'I27.8', 'I27.9', '^J40.*', '^J41.*', '^J42.*', '^J43.*', '^J44.*', '^J45.*', '^J46.*', '^J47.*',
                      '^J60.*', '^J61.*', '^J62.*', '^J63.*', '^J64.*', '^J65.*', '^J66.*', '^J67.*', 'J68.4', 'J70.1',
                      'J70.3',
                      # connective tissue / rheumatic disease
                      '^M05.*', '^M06.*', 'M31.5', '^M32.*', '^M33.*', '^M34.*', 'M35.1', 'M35.3', 'M36.0',
                      # peptic ulcer disease
                      '^K25.*', '^K26.*', '^K27.*', '^K28.*',
                      # mild liver disease
                      '^B18.*', 'K70.0', 'K70.1', 'K70.2', 'K70.3', 'K70.9', 'K71.3', 'K71.4', 'K71.5', 'K71.7', '^K73.*', '^K74.*',
                      'K76.0', 'K76.2', 'K76.3', 'K76.4', 'K76.8', 'K76.9', 'Z94.4',
                      # diabetes without complications
                      'E10.0', 'E10.1', 'E10.6', 'E10.8', 'E10.9', 'E11.0', 'E11.1', 'E11.6', 'E11.8', 'E11.9', 'E12.0',
                      'E12.1', 'E12.6', 'E12.8', 'E12.9', 'E13.0', 'E13.1', 'E13.6', 'E13.8', 'E13.9', 'E14.0', 'E14.1',
                      'E14.6', 'E14.8', 'E14.9']

    score_two_list = ['E10.2', 'E10.3', 'E10.4', 'E10.5', 'E10.7', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.7', 'E12.2',
                      'E12.3', 'E12.4', 'E12.5', 'E12.7', 'E13.2', 'E13.3', 'E13.4', 'E13.5', 'E13.7', 'E14.2', 'E14.3',
                      'E14.4', 'E14.5', 'E14.7',  # diabetes with complications
                      # hemiplegia/paraplegia
                      'G04.1', 'G11.4', 'G80.1', 'G80.2', '^G81.*', '^G82.*', 'G83.0', 'G83.1', 'G83.2', 'G83.3', 'G83.4',
                      'G83.9',
                      # moderate to severe chronic kidney disease
                      'I12.0', 'I13.1', 'N03.2', 'N03.3', 'N03.4', 'N03.5', 'N03.6', 'N03.7', 'N05.2', 'N05.3', 'N05.4',
                      'N05.5', 'N05.6', 'N05.7', '^N18.*', '^N19.*', 'N25.0', 'Z49.0', 'Z49.1', 'Z49.2', 'Z94.0', 'Z99.2',
                      # localized solid tumor
                      '^C00.*', '^C01.*', '^C02.*', '^C03.*', '^C04.*', '^C05.*', '^C06.*', '^C07.*', '^C08.*', '^C09.*',
                      '^C10.*', '^C11.*', '^C12.*', '^C13.*', '^C14.*', '^C15.*', '^C16.*', '^C17.*', '^C18.*', '^C19.*',
                      '^C20.*', '^C21.*', '^C22.*', '^C23.*', '^C24.*', '^C25.*', '^C26.*', '^C30.*', '^C31.*', '^C32.*',
                      '^C33.*', '^C34.*', '^C37.*', '^C38.*', '^C39.*', '^C40.*', '^C41.*', '^C43.*', '^C45.*', '^C46.*',
                      '^C47.*', '^C45.*', '^C46.*', '^C47.*', '^C48.*', '^C49.*', '^C50.*', '^C51.*', '^C52.*', '^C53.*',
                      '^C54.*', '^C55.*', '^C56.*', '^C57.*', '^C58.*', '^C60.*', '^C61.*', '^C62.*', '^C63.*', '^C64.*',
                      '^C65.*', '^C66.*', '^C67.*', '^C68.*', '^C69.*', '^C70.*', '^C71.*', '^C72.*', '^C73.*', '^C74.*',
                      '^C75.*', '^C76.*', '^C81.*', '^C82.*', '^C83.*', '^C84.*', '^C85.*', '^C88.*', '^C90.*', '^C91.*',
                      '^C92.*', '^C93.*', '^C94.*', '^C95.*', '^C96.*', '^C97.*']

    # severe liver disease
    score_three_list = ['I85.0', 'I85.9', 'I86.4', 'I98.2',
                        'K70.4', 'K71.1', 'K72.1', 'K72.9', 'K76.5', 'K76.6', 'K76.7']

    # AIDS / metastatic solid tumor
    score_four_list = ['^C77.*', '^C78.*', '^C79.*',
                       '^C80.*', '^B20.*', '^B21.*', '^B22.*', '^B24.*']

    # combined list
    scores_list = [1, 2, 3, 4]

    # recode age ranges as per Charlson scoring metric
    age_bins = [0, 50, 60, 70, 80, ]
    age_scores = [0, 1, 2, 3, 4]

    d = dict(enumerate(age_scores, 1))

    charlson_df['age_score'] = np.vectorize(d.get)(
        np.digitize(charlson_df['age'], age_bins))
    # charlson_df = charlson_df.set_index(['PATIENT ID', 'EDAD/AGE', 'tratamiento','MOTIVO_ALTA/DESTINY_DISCHARGE_ING'])
    # charlson_df[~charlson_df.isin(combi_score_list)] = np.nan
    charlson_df = charlson_df.replace(
        to_replace=score_one_list, value=1, regex=True)
    charlson_df = charlson_df.replace(
        to_replace=score_two_list, value=2, regex=True)
    charlson_df = charlson_df.replace(
        to_replace=score_three_list, value=3, regex=True)
    charlson_df = charlson_df.replace(
        to_replace=score_four_list, value=4, regex=True)
    charlson_df[~charlson_df.isin(scores_list)] = np.nan
    charlson_df = charlson_df.replace(to_replace=np.nan, value=0, regex=True)
    charlson_df['charlson'] = charlson_df.sum(axis=1)

    # remove redundant ICD10 columns
    charlson_df = charlson_df[['charlson']]

    return charlson_df

