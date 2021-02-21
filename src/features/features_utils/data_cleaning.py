import pandas as pd
import numpy as np

def clean_static_df(static_df):
    static_df_clean = static_df
    static_df_clean = pd.get_dummies(data=static_df_clean, columns=[
                                     'sex']).drop('sex_MALE', axis=1)

    static_df_clean.drop('discharge_destination', axis=1, inplace=True)

    static_df_clean.loc[:, 'datetime_min'] = pd.to_datetime(static_df_clean['datetime_min'],
        dayfirst=True)

    first_lab_time_col = pd.to_datetime(
        static_df_clean['datetime_min'], dayfirst=True)
    static_df_clean.loc[:, 'icu_in_year'] = first_lab_time_col.dt.year
    static_df_clean.loc[:, 'icu_in_month'] = first_lab_time_col.dt.month
    static_df_clean.loc[:, 'icu_in_day'] = first_lab_time_col.dt.day
    static_df_clean.drop('datetime_min', axis=1, inplace=True)
    static_df_clean.drop('datetime_max', axis=1, inplace=True)
    # TODO: why is there no hourly info?

    static_df_clean.set_index('patient_id')

    return static_df_clean


def clean_dynamic_df(dynamic_df, static_df):
    dynamic_df_times = dynamic_df.merge(static_df[['patient_id','datetime_min']],
        how='inner', on='patient_id')

    dynamic_df_times.loc[:, 'datetime_min'] = pd.to_datetime(dynamic_df_times['datetime_min'],
        dayfirst=True)
    dynamic_df_times.loc[:, 'datetime'] = pd.to_datetime(dynamic_df_times['datetime'],
        dayfirst=True)

    dynamic_df_times.loc[:, 'hours_in'] = \
         (dynamic_df_times['datetime'] - dynamic_df_times['datetime_min'])

    dynamic_df_times = dynamic_df_times.drop(['datetime','datetime_min'], axis=1)

    # upsample to hourly
    dynamic_df_hourly = dynamic_df_times.set_index('hours_in').groupby('patient_id').resample('H').mean()
    dynamic_df_hourly = dynamic_df_hourly.drop('patient_id', axis=1).reset_index()
#    dynamic_df_hourly = dynamic_df_hourly.set_index('patient_id')

    dynamic_df_clean = dynamic_df_hourly.set_index(['patient_id','hours_in']).dropna(how='all')

    return dynamic_df_clean


def clean_treat_df(treat_df, static_df):
    treat_df_times = treat_df.merge(static_df[['patient_id','datetime_min']],
        how='inner', on='patient_id')

    treat_df_times.loc[:, 'datetime_min'] = pd.to_datetime(treat_df_times['datetime_min'],
        dayfirst=True)
    treat_df_times.loc[:, 'datetime'] = pd.to_datetime(treat_df_times['date'],
        dayfirst=True)

    treat_df_times.loc[:, 'hours_in'] = \
         (treat_df_times['datetime'] - treat_df_times['datetime_min'])

    treat_df_times = treat_df_times.drop(['datetime','datetime_min'], axis=1)

    treat_df_hourly = treat_df_times.set_index('hours_in').groupby('patient_id').resample('H').mean()
    treat_df_hourly = treat_df_hourly.drop('patient_id', axis=1).reset_index()

    treat_df_clean = treat_df_hourly.set_index(['patient_id','hours_in']).dropna(how='all')

    return treat_df_clean


def clean_outcome_df(static_df):
    outcome_df = static_df[['patient_id','discharge_destination','datetime_min','datetime_max']]
    outcome_df.loc[:,'datetime_min'] = pd.to_datetime(outcome_df['datetime_min'])
    outcome_df.loc[:,'datetime_max'] = pd.to_datetime(outcome_df.loc[:,'datetime_max'])

    outcome_df['hours_in'] = outcome_df['datetime_max'] - outcome_df['datetime_min']

    outcome_df.loc[:, 'death'] = (outcome_df['discharge_destination'] == 'Fallecimiento') * 1.0

    outcome_df = outcome_df.drop(['discharge_destination','datetime_min','datetime_max'], axis=1)
    outcome_df = outcome_df.set_index('patient_id')
    
    return outcome_df


def get_CXYT_for_modeling(static_df, dynamic_df, treat_df, outcome_df, output_filepath):
    static_df = static_df.reset_index()
    dynamic_df = dynamic_df.reset_index()
    treat_df = treat_df.reset_index()
    outcome_df = outcome_df.reset_index()

    # upsample treatemnt to hourly before merging
    treat_df.loc[:, 'hours_in'] = pd.to_timedelta(treat_df['hours_in'])
    dynamic_df.loc[:, 'hours_in'] = pd.to_timedelta(dynamic_df['hours_in'])
    outcome_df.loc[:, 'hours_in'] = pd.to_timedelta(outcome_df['hours_in'])
    treat_df.set_index('hours_in').groupby('patient_id').resample('H').ffill()
    treat_df_upsampled = treat_df.\
        set_index('hours_in').groupby('patient_id').resample('H').\
        ffill().drop('patient_id', axis=1)    

    dynamic_treat_outcome_df = dynamic_df.merge(
        treat_df_upsampled, how='outer',on=['patient_id', 'hours_in']).merge(
        outcome_df, how='outer',on=['patient_id', 'hours_in'])

    treat_cols = dynamic_treat_outcome_df.columns.str.contains('HIBOR')
    outcome_cols = dynamic_treat_outcome_df.columns == 'death'

    #fill in not treated timepoints and not dead timepoints with 0 before padding time
    dynamic_treat_outcome_df.loc[:,treat_cols | outcome_cols] = \
        dynamic_treat_outcome_df.loc[:,treat_cols | outcome_cols].fillna(0)


    date_first = dynamic_treat_outcome_df['hours_in'].min()  
    date_first_trim = pd.Timedelta('-2 days')
    date_last = dynamic_treat_outcome_df['hours_in'].max() 
    date_last_trim = pd.Timedelta('30 days 0 hours')
    print(date_first, date_last)

    mux = pd.MultiIndex.from_product([list(set(dynamic_treat_outcome_df['patient_id'])),
                                    pd.timedelta_range(date_first_trim, date_last_trim, freq='H')],
                                    names=['patient_id', 'hours_in_filled'])

    num_hours = len(pd.timedelta_range(date_first_trim, date_last_trim, freq='H'))
    #print(num_hours)
    print('size of new time-padded data ' + str(len(mux)))
    
    temp = dynamic_treat_outcome_df.set_index(['patient_id', 'hours_in'])
    result = temp.reindex(mux, fill_value=np.nan).reset_index()

    #fill in not treated timepoints and not dead timepoints with 0, after padding time too
    result.loc[:,treat_cols | outcome_cols] = result.loc[:,treat_cols | outcome_cols].fillna(0)

    # need to drop patients whose outcomes was after the max date cutoff
    death_times_dist = dynamic_treat_outcome_df.loc[dynamic_treat_outcome_df['death'] == 1,:][['patient_id','hours_in','death']]
    died_too_late = death_times_dist.loc[death_times_dist['hours_in'] > date_last_trim, :]['patient_id']
    print('omitting these patients' + str(died_too_late.values) + ' because their outcome times was after the date cutoff')

    XYT_df = result.loc[~result['patient_id'].isin(died_too_late),:]
    num_patients = len(set(XYT_df['patient_id']))

    CXYT_df = static_df.merge(XYT_df, on='patient_id', how='right')

    return CXYT_df
