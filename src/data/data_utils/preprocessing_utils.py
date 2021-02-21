
import pandas as pd
import numpy as np

from .icd_parsing_functions import get_code_categories, charlson_calc



def append_more_covariates(dynamic_df, drugs_covariates):
    left = dynamic_df.reset_index()
    right = drugs_covariates.reset_index().rename(columns={'date':'datetime'})

    left.loc[:, 'datetime'] = pd.to_datetime(left.loc[:, 'datetime'])
    right.loc[:, 'datetime'] = pd.to_datetime(right.loc[:, 'datetime'])
    dynamic_df_merged = left.merge(right, how='outer',
        on=['patient_id', 'datetime'])
    dynamic_df_merged = dynamic_df_merged.set_index(['patient_id','datetime'])
    return dynamic_df_merged


def get_drugs_timeseries_df(inpatients_records, drugs, clinical_vars, labs, output_filepath,
max_num_drugs=100,logger=None):

    try:
        drugs_daily_wide = pd.read_csv(output_filepath + '/drugs_daily_wide.csv', index_col=0)

    except FileNotFoundError: 
        top_drug_labels = drugs['drug_name'].value_counts()[:max_num_drugs].index.tolist()
        drug_select = drugs[drugs['drug_name'].isin(top_drug_labels)]
        drugs_simple = drug_select[['patient_id','drug_name','average_daily_dose','start_date', 'end_date']]
        
        # we need an index column to use to group start and end times later on
        drugs_simple_indexcol = drugs_simple.reset_index()

        drugs_startend = drugs_simple_indexcol.melt(
            id_vars=['patient_id','drug_name','average_daily_dose','index'],
            value_name='date',
            var_name='start_or_end_date')

        # lots of drugs have the same start and end date, so drop duplicates there for resampling
        temp = drugs_startend.set_index(['date','index'])
        temp2 = temp[~temp.index.duplicated()].drop('start_or_end_date',axis=1)

        if logger is not None:
            logger.info('interpolate drugs df to daily (approx 2 mins)')
        drugs_daily = temp2.reset_index().set_index('date').groupby('index').resample('D').ffill()
        drugs_daily_clean = drugs_daily.drop('index',axis=1).\
            reset_index().\
            set_index('patient_id').drop('index', axis=1)

        # pivot to have one column per drug name
        drugs_daily_wide = drugs_daily_clean.pivot_table(index=['patient_id', 'date'], 
        columns='drug_name')
        drugs_daily_wide.columns = drugs_daily_wide.columns.droplevel(0)

        print('a')
        drugs_daily_wide = drugs_daily_wide.reset_index()

        drugs_daily_wide.loc[:, 'date'] = pd.to_datetime(
            drugs_daily_wide.loc[:, 'date'], dayfirst=True)

        drugs_daily_wide = drugs_daily_wide.set_index('patient_id')

        drugs_daily_wide.to_csv(output_filepath + '/drugs_daily_wide.csv')

    return drugs_daily_wide


def split_treat_covariates(drugs_timeseries_df, drug_name='HIBOR'):
    
    treat_cols = drugs_timeseries_df.columns.str.contains(drug_name)
    date_col = drugs_timeseries_df.columns == 'date'
    
    drugs_treatments = drugs_timeseries_df.loc[:, treat_cols | date_col]
    drugs_covariates = drugs_timeseries_df.loc[:, ~treat_cols | date_col]
    
    return drugs_treatments, drugs_covariates


def get_static_vars(inpatients_records, clinical_vars, labs, output_filepath,  to_print=True):
    # this function returns static_df, with age, sex... and dynamic, with HR... and labs (w/ times)
    # has everything except ICDs
    # TODO; split into smaller functions!
    if to_print:
        print('parsing with static data')
    # again, using lab times to indicate ICU in time
    labtimes = labs[labs['lab_request_id'].str.contains(
        'I-')][['patient_id', 'datetime']]
    clinicalvars_times = clinical_vars[['patient_id','datetime']]
    labtimes_min = labtimes.groupby('patient_id').min()['datetime'].reset_index()
    clinicalvars_time_max = clinicalvars_times.groupby('patient_id').max()['datetime'].reset_index()
    labtimes = labtimes_min.merge(clinicalvars_time_max, how='inner', on='patient_id', 
        suffixes=['_min','_max'])
    
    # joining the csvs
    static_df = labtimes.merge(
        inpatients_records[['patient_id', 'age', 'sex',
                            'inpatient_covid_dx', 'discharge_destination']],
        on='patient_id', how='inner')
    static_df = static_df[static_df['inpatient_covid_dx'] == 'COVID CONFIRMADO'].\
        drop(['inpatient_covid_dx'], axis=1)

    return static_df

def get_dynamic_vars(clinical_vars, labs, output_filepath, to_print=True, max_labs=20):
    try:
        wide_dynamic_df = pd.read_csv(output_filepath + '/labs_clinicalvars.csv', index_col=0)

    except FileNotFoundError: 
        # getting dynamic variables
        top_lab_labels = labs['label'].value_counts()[:max_labs].index.tolist()
        labs_select = labs[labs['label'].isin(top_lab_labels)]

        labs_simple = labs_select[['patient_id', 'label', 'value', 'datetime']].\
            rename({'datetime': 'datetime_lab'}, axis=1)

        clinical_vars_clean = clinical_vars.replace([0, '0'], np.nan).\
            rename({'datetime': 'datetime_clinical'}, axis=1)
        #clinical_vars_clean.loc[:, 'temperature'] = \
        #    clinical_vars_clean['temperature'].str.replace(',', '.')
        clinical_vars_melt = clinical_vars_clean.melt(
            id_vars=['patient_id', 'datetime_clinical'])
        clinical_vars_melt.dropna(subset=['value'], inplace=True)

        if to_print:
            print('merging labs and clinical vars')

        dynamic_df = labs_simple.merge(clinical_vars_melt, on='patient_id', how='inner',
                                       suffixes=['_lab', '_clinical'])

        if to_print:
            print('turning dynamic vars into long df (approx 20 secs)')
        # need to unstack into one long df
        melt_labels = dynamic_df.melt(id_vars=['patient_id'], value_vars=['label', 'variable'],
                                      value_name='label')[['patient_id', 'label']]
        melt_values = dynamic_df.melt(id_vars=['patient_id'],
                                      value_vars=['value_lab', 'value_clinical'])[['value']]
        melt_times = dynamic_df.melt(id_vars=['patient_id'],
                                     value_vars=['datetime_lab',
                                                 'datetime_clinical'], value_name='datetime')[['datetime']]
        dynamic_df_long = pd.concat(
            (melt_labels, melt_values, melt_times), axis=1)

        if to_print:
            print('dynamic df has ' + str(len(dynamic_df_long)) + ' rows')

        if to_print:
            print('casting to numeric values')

        dynamic_df = cast_to_numeric(dynamic_df_long)

        if to_print:
            print('dynamic df now has ' + str(len(dynamic_df_long)) + ' rows')
            print('widening df')

        wide_dynamic_df = get_widened_df(dynamic_df)

        wide_dynamic_df.to_csv(output_filepath + '/labs_clinicalvars.csv')#, index0=True)

    return wide_dynamic_df


def cast_to_numeric(long_df_pruned):
    # learn about messiness in data, will have to repeat if max_labs is changed from 100
    #temp2 = long_df_pruned[pd.to_numeric(long_df_pruned['value'], errors='coerce').isna()][[
    #    'label', 'value']]
    #temp3 = temp2.drop_duplicates()

    #print('starting messiness of data: ' +
    #      str(np.round(100 * len(temp2) / len(temp), 2)) + ' %')

    #print('TODO: for now just omitting these non-numeric data, include them cleaned up later')

    long_df_pruned.loc[:, 'value'] = pd.to_numeric(
        long_df_pruned['value'], errors='coerce')

    return long_df_pruned


def get_widened_df(dynamic_df):
    # dynamic df should only have patient_id, datetime, label, value

    wide_dynamic_df = dynamic_df.pivot_table(
        index=['patient_id', 'datetime'], columns='label')

    wide_dynamic_df.columns = wide_dynamic_df.columns.get_level_values(1)

    return wide_dynamic_df


def add_icds_to_static_vars(static_df, codes_emergency, codes_inpatient):
    # adds charlson index and code category columns to static df, keeping its len at num_patients
    # does this for emergency and inpatient codes
    # TODO: split into functions
    # for emergency codes: use directly
    static_codes_df = static_df.merge(
        codes_emergency, on='patient_id', how='left')

    dx_colnames = list(
        static_codes_df.columns[static_codes_df.columns.str.contains('DIA')])
    df_for_charlson_calc = static_codes_df[['patient_id', 'age'] + dx_colnames]

    charlson_df = charlson_calc(df_for_charlson_calc.copy())
    charlson_df = static_codes_df.merge(charlson_df, on='patient_id', how='left')[
        ['patient_id', 'charlson']]

    code_categories_df = get_code_categories(static_codes_df)
    combined_icd_df = charlson_df.merge(
        code_categories_df, on='patient_id', how='left')

    # for inpatient codes, just use if present on admission (sort bc not in same order)
    dx_colnames_inpatient = list(np.sort(
        codes_inpatient.columns[codes_inpatient.columns.str.contains('DIA')]))
    poa_colnames_inpatient = list(np.sort(
        codes_inpatient.columns[codes_inpatient.columns.str.contains('POA')]))

    codes_censor = codes_inpatient.copy()

    for i in range(len(dx_colnames_inpatient)):
        dx_col = dx_colnames_inpatient[i]
        poa_col = poa_colnames_inpatient[i]
        # make dx col nan where poa col is not S
        codes_censor[dx_col] = codes_censor[dx_col].where(
            codes_censor[poa_col].isin(['S']))

    static_inpatient_codes_df = static_df.merge(
        codes_censor, on='patient_id', how='left')
    df_for_charlson_calc_inpatient = static_inpatient_codes_df[[
        'patient_id', 'age'] + dx_colnames_inpatient]

    charlson_df_inpatient = charlson_calc(
        df_for_charlson_calc_inpatient.copy())
    code_categories_df_inpatient = get_code_categories(
        static_inpatient_codes_df)

    combined_icd_inpatient_df = charlson_df_inpatient.merge(code_categories_df_inpatient,
                                                            on='patient_id', how='inner')

    combined_all_icd_df = combined_icd_df.merge(combined_icd_inpatient_df,
                                                on='patient_id',
                                                how='inner', suffixes=['_emerg', '_inpat'])

    # need to only have one row per patient: need to merge 1 and 0 for categoricals, so take MAX
    # this way, if a patient EVER is recorded to have it, mark it as yes
    # note that using just charlson indices, there is just one value (for each of emerg, inpat) per patient
    combined_all_icd_df = combined_all_icd_df.drop_duplicates()
    combined_all_icd_df = combined_all_icd_df.groupby('patient_id').max()

    static_with_icd_df = static_df.merge(combined_all_icd_df, on='patient_id', how='left')
    return static_with_icd_df
