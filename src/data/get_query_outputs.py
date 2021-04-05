# MIMIC IIIv14 on postgres 9.4
import argparse
import os
import pickle
import numpy as np
import pandas as pd

pickle.HIGHEST_PROTOCOL = 3

# Output filenames
static_filename = 'static_data.csv'
static_columns_filename = 'static_colnames.txt'

dynamic_filename = 'vitals_hourly_data.csv'
columns_filename = 'vitals_colnames.txt'
subjects_filename = 'subjects.npy'
times_filename = 'fenceposts.npy'
dynamic_hd5_filename = 'vitals_hourly_data.h5'
dynamic_hd5_filt_filename = 'all_hourly_data.h5'

codes_filename = 'C.npy'
codes_hd5_filename = 'C.h5'
idx_hd5_filename = 'C_idx.h5'

outcome_filename = 'outcomes_hourly_data.csv'
outcome_hd5_filename = 'outcomes_hourly_data.h5'
outcome_columns_filename = 'outcomes_colnames.txt'

# SQL command params
dbname = 'mimic'
schema_name = 'mimiciii'

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']


def get_values_by_name_from_df_column_or_index(data_df, colname):
    """ Easily get values for named field, whether a column or an index

    Returns
    -------
    values : 1D array
    """
    try:
        values = data_df[colname]
    except KeyError as e:
        if colname in data_df.index.names:
            values = data_df.index.get_level_values(colname)
        else:
            raise e
    return values


def add_outcome_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()

    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hours, 'on':on_vals}) #icustay_id': icustay_id})


def add_blank_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    #icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0]*len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hrs, 'on':vals})#'icustay_id': icustay_id,


def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = pd.read_csv(mimic_mapping_filename, index_col=None).fillna('')#.astype(str)
    var_map = var_map.loc[(var_map['LEVEL2'] != '') & (var_map.count(axis='columns')>0)]
    var_map = var_map.loc[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    return var_map


def get_variable_ranges(range_filename):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [ 'LEVEL2', 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH' ]
    to_rename = dict(zip(columns, [ c.replace(' ', '_') for c in columns ]))
    to_rename['LEVEL2'] = 'VARIABLE'
    var_ranges = pd.read_csv(range_filename, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges['VARIABLE'] = var_ranges['VARIABLE'].map(str.lower)
    var_ranges.set_index('VARIABLE', inplace=True)
    var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]

    return var_ranges


UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]


def standardize_units(X, name_col='itemid', unit_col='valueuom', value_col='value', inplace=False):
    if not inplace: X = X.copy()
    name_col_vals = get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except:
        print("Can't call *.str")
        print(name_col_vals)
        print(unit_col_vals)
        raise

    #name_filter, unit_filter = [
    #    (lambda n: col.contains(n, case=False, na=False)) for col in (name_col_vals, unit_col_vals)
    #]
    # TODO(mmd): Why does the above not work, but the below does?
    name_filter = lambda n: name_col_vals.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None: needs_conversion_filter_idx |= name_filter(unit) | unit_filter(unit)
        if rng_check_fn is not None: needs_conversion_filter_idx |= rng_check_fn(X[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        X.loc[idx, value_col] = convert_fn(X[value_col][idx])

    return X


def interpolate_variables(df):
    df_copy = df.copy()
    df_copy.loc[:, 'hours_in'] = (df_copy['charttime'] - df_copy['intime']).dt.floor('min')

    df_pruned = df_copy[ID_COLS + ['label', 'value', 'hours_in']]
    #df_pruned.loc[:, 'value'] = df_pruned['value'].astype(np.float32)
    df_pruned_indexed = df_pruned.set_index(ID_COLS)
    df_pivot = df_pruned_indexed.pivot_table(values='value',columns='label',index=ID_COLS + ['hours_in'], aggfunc='last')

    df_filled = df_pivot.reset_index().set_index(ID_COLS + ['hours_in'])
    #TODO DOESNT DO ANYTHING TODO
    #df_filled_grouped = df_filled.groupby([pd.Grouper(key=idx) for idx in ID_COLS]).resample('H').mean() #upsample to hourly


    return df_filled


def preprocess_dynamic_covariates(dynamic_covariates_df):
    # columns to start: ubject_id hadm_id icustay_id intime outtime charttime itemid value valueuom
    # var_map = var_map[['LEVEL2', 'ITEMID', 'LEVEL1']].set_index('ITEMID')

    # df = pd.merge(dynamic_covariates_df, var_map, how='left', left_on='label', right_on='LEVEL1')
    # df['value'] = pd.to_numeric(df['value'], 'coerce')

    # df = standardize_units(df, name_col='LEVEL1', inplace=False)

    # df = apply_variable_limits(df, var_ranges, 'LEVEL2')

    df = interpolate_variables(dynamic_covariates_df)

    return df
    
def interpolate_input_events(input_events_df):
    # in this function: there are two types of inputs, continuous and bolus
    ## for the continuous, add the rate over the entire time its given, and upsample to minutely
    ## note that some intervals of continuous drug overlap, so sum over those overlapping

    ## for the bolus, treat as separate columns and record the amount at a given time
    ## determine bolus vs continuous via wehtether rate is missing (-> bolus)

    #df = inputs_covariates_pruned.copy()
    df = input_events_df.copy()

    df_cont = df.loc[~pd.isna(df['rate']), :]
    df_bolus = df.loc[pd.isna(df['rate']), :]

    df_cont.loc[:, 'starttime_in'] = (df_cont['starttime'] - df_cont['intime']).dt.floor('min')
    df_cont.loc[:, 'endtime_in'] = (df_cont['endtime'] - df_cont['intime']).dt.floor('min')
    df_cont_pruned = df_cont[ID_COLS + ['rate', 'label', 'starttime_in', 'endtime_in']]
    df_pivot = df_cont_pruned.pivot_table(
        values='rate', columns='label', index=ID_COLS + ['starttime_in', 'endtime_in'],
        aggfunc='last')
    #sometimes, there is one start and several stopping times, so use the latest endtime only
    df_pivot_groupstarts = df_pivot.reset_index().\
        groupby(by=ID_COLS + ['starttime_in']).last().\
        set_index(['endtime_in'], append=True)

    #reset twice gives us a column to use called "index" later on when matching intervals
    indexed_df = df_pivot_groupstarts.reset_index().reset_index() 

    # this melts on indexes "index" (just 0 through num rows), ID_COLS, and each variable name
    melted_df = pd.melt(indexed_df,
        id_vars=list(indexed_df.drop(['starttime_in','endtime_in'], axis=1).columns),
        value_vars=['starttime_in','endtime_in'],
        var_name='startend',value_name='hours_in')

    # takes about a minute; fills only within each index; each index identifies a start-end pair
    filled_timegroups = melted_df.set_index('hours_in').groupby('index').resample('min').\
        ffill().drop('startend', axis=1)
    
    # sum overlapping intervals, only group by hours_in and id_cols
    summed_timegroups = filled_timegroups.drop('index',axis=1).reset_index().\
        groupby(['hours_in'] + ID_COLS).sum()
    df_summed_timegroups_clean = summed_timegroups.drop('index',axis=1).\
        reset_index().set_index(ID_COLS + ['hours_in']).sort_index()

    # TODO: merge with interpolate_variables fn
    # treat the bolus ins the same way as the chartevents, since they happen at a discrete time
    df_bolus.loc[:, 'hours_in'] = (df_bolus['starttime'] - df_bolus['intime']).dt.floor('min')

    df_bolus_pruned = df_bolus[ID_COLS + ['label', 'amount', 'hours_in']]

    df_bolus_pivot = df_bolus_pruned.pivot_table(
        values='amount',columns='label',index=ID_COLS + ['hours_in'], aggfunc='last')

    df_bolus_pivot_rename = df_bolus_pivot.copy()
    df_bolus_pivot_rename.columns = [colname + '_bolus' for colname in list(df_bolus_pivot.columns)]

    return df_summed_timegroups_clean, df_bolus_pivot_rename

def preprocess_input_events(input_events_df): #TODO: unify with other preprocessor
    input_events_processed = interpolate_input_events(input_events_df)

    return input_events_processed

def apply_variable_limits(df, var_ranges, var_names_index_col='LEVEL2'):
    idx_vals        = df[var_names_index_col]
    non_null_idx    = ~df['value'].isnull()
    var_names       = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    for var_name in var_names:
        if type(var_name) == float and np.isnan(var_name) or var_name.lower() not in var_range_names:
            print("No known ranges for %s" % var_name)
            continue
        else:
            var_name_lower = var_name.lower()

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x] for x in ('OUTLIER_LOW','OUTLIER_HIGH','VALID_LOW','VALID_HIGH')
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx  = (df.value < outlier_low_val)
        outlier_high_idx = (df.value > outlier_high_val)
        valid_low_idx    = ~outlier_low_idx & (df.value < valid_low_val)
        valid_high_idx   = ~outlier_high_idx & (df.value > valid_high_val)

        var_outlier_idx   = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        df.loc[var_outlier_idx, 'value'] = np.nan
        df.loc[var_valid_low_idx, 'value'] = valid_low_val
        df.loc[var_valid_high_idx, 'value'] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0:
            print(
                "%s had %d / %d rows cleaned:\n"
                "  %d rows were strict outliers, set to np.nan\n"
                "  %d rows were low valid outliers, set to %.2f\n"
                "  %d rows were high valid outliers, set to %.2f\n"
                "" % (
                    var_name,
                    n_outlier + n_valid_low + n_valid_high, sum(running_idx),
                    n_outlier, n_valid_low, valid_low_val, n_valid_high, valid_high_val
                )
            )

    return df

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_path', type=str, default= '../data/curated',
                    help='Enter the path you want the output')
    ap.add_argument('--resource_path',
        type=str,
        default=os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR/resources/"))
    ap.add_argument('--extract_pop', type=int, default=1,
                    help='Whether or not to extract population data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')

    ap.add_argument('--extract_numerics', type=int, default=1,
                    help='Whether or not to extract numerics data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--extract_outcomes', type=int, default=1,
                    help='Whether or not to extract outcome data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--extract_codes', type=int, default=1,
                    help='Whether or not to extract ICD9 codes: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--pop_size', type=int, default=-1,
                    help='Size of population to extract')
    ap.add_argument('--exit_after_loading', type=int, default=0)
    ap.add_argument('--var_limits', type=int, default=1,
                    help='Whether to create a version of the data with variable limits included. ' +
                    '1 - apply variable limits, 0 - do not apply variable limits')
    ap.add_argument('--plot_hist', type=int, default=1,
                    help='Whether to plot the histograms of the data')
    ap.add_argument('--psql_host', type=str, default=None,
                    help='Postgres host. Try "/var/run/postgresql/" for Unix domain socket errors.')
    ap.add_argument('--psql_password', type=str, default=None, help='Postgres password.')
    ap.add_argument('--group_by_level2', action='store_false', dest='group_by_level2', default=True,
                    help='Do group by level2.')
    
    ap.add_argument('--min_percent', type=float, default=0.0,
                    help='Minimum percentage of row numbers need to be observations for each numeric column.' +
                    'min_percent = 1 means columns with more than 99 percent of nan will be removed')
    ap.add_argument('--min_age', type=int, default=15,
                    help='Minimum age of patients to be included')
    ap.add_argument('--min_duration', type=int, default=12,
                    help='Minimum hours of stay to be included')
    ap.add_argument('--max_duration', type=int, default=240,
                    help='Maximum hours of stay to be included')
    
    #############
    # Parse args
    args = vars(ap.parse_args())
    printargs = False
    if printargs:
        for key in sorted(args.keys()):
            print(key, args[key])

#  TODO
    #if not isdir(args['resource_path']):
    #    raise ValueError("Invalid resource_path: %s" % args['resource_path'])


    return args

def get_eligibility_query_prefix(pop_size_string, min_age_string, min_dur_string, max_dur_string, min_day_string, 
    query_dir = './src/queries/'):
    with open(query_dir + '/eligibility.sql','r') as f:
        query_raw = f.read()
    query_parsed = query_raw.format(
        limit=pop_size_string, min_age=min_age_string, min_dur=min_dur_string, 
        max_dur=max_dur_string, min_day=min_day_string)
    return query_parsed

def get_eligibility_query(eligibility_query_prefix, query_dir = './src/queries'):
    query = eligibility_query_prefix + \
    """
    select * from eligible_patients
        ;
        """
    return query

def get_static_covariates_query(eligibility_query_prefix, query_dir = './src/queries/'):
    with open(query_dir + 'static_covariates.sql','r') as f:
        query_raw = f.read()
    query_parsed = eligibility_query_prefix + query_raw # no formatting needed
    return query_parsed

def get_outcomes_query(eligibility_query_prefix, query_dir = './src/queries/') -> str:    
    with open(query_dir +  'outcomes.sql','r') as f:
        query_raw = f.read()
    query_parsed = eligibility_query_prefix + query_raw # no formatting needed yet
    return query_parsed

def get_treatment_query(eligibility_query_prefix, query_dir = './src/queries/'):
    with open(query_dir + 'treatments.sql','r') as f:
        query_raw = f.read()
    query_parsed = eligibility_query_prefix + query_raw # no formatting needed yet
    return query_parsed

def get_inputs_query(eligibility_query_prefix, query_dir = './src/queries/'):
    with open(query_dir + 'inputs_dynamic.sql','r') as f:
        query_raw = f.read()
    query_parsed = eligibility_query_prefix + query_raw # no formatting needed yet
    return query_parsed

def get_dynamic_covariates_query(eligibility_query_prefix, query_dir = './src/queries/'):#, chartitems_to_keep, labitems_to_keep):
    with open(query_dir + 'dynamic_covariates_choose_all.sql','r') as f:
        query_raw = f.read()
    query_parsed = eligibility_query_prefix + query_raw#.format(\
        #chitem=chartitems_to_keep, lbitem=labitems_to_keep)
    return query_parsed

def get_useful_chart_lab_itemids(common_chartlabs_filename):
    common_ids = pd.read_csv(common_chartlabs_filename)

    common_chartevents = common_ids[common_ids['chartorlab'] == 'chart']['itemid']
    common_labevents = common_ids[common_ids['chartorlab'] == 'lab']['itemid']

    chartitems_to_keep = str(tuple(common_chartevents.values[:200]))
    labitems_to_keep = str(tuple(common_labevents.values[:100]))

if __name__ == '__main__':
    #TODO main: icd codes, "colloid_bolus", "crystalloid_bolus", "nivdurations" vasopressordurations', 'adenosinedurations', 'dobutaminedurations', 'dopaminedurations', 'epinephrinedurations', 'isupreldurations',                      'milrinonedurations', 'norepinephrinedurations', 'phenylephrinedurations', 'vasopressindurations']
    
    using_args = False
    if using_args:
        args = get_args()
        min_age_string = str(args['min_age'])
        min_dur_string = str(args['min_duration'])
        max_dur_string = str(args['max_duration'])
        min_day_string = str(float(args['min_duration'])/24)
        if args['pop_size'] == -1:
            pop_size_string = ''
        else:
            pop_size_string = str(args['pop_size'])
        #if args['psql_host'] is not None: query_args['host'] = args['psql_host']
        #if args['psql_password'] is not None: query_args['password'] = args['psql_password']
    else:
        min_age_string = '16'
        min_dur_string = '12'
        max_dur_string = '240'
        min_day_string = '0.5'
        pop_size_string = ''

    root_dir = '/home/joe/testbed/mimic_understander/'

    # used for joining rest of downstream queries
    eligibility_query_prefix = get_eligibility_query_prefix(
        pop_size_string, min_age_string, min_dur_string, max_dur_string, min_day_string)
    
    static_covariates_query = get_static_covariates_query(eligibility_query_prefix)
    dynamic_covariates_query = get_dynamic_covariates_query(eligibility_query_prefix)

    outcomes_query = get_outcomes_query(eligibility_query_prefix)


    inputs_query = get_inputs_query(eligibility_query_prefix)
    #print('getting dynamic covariates')
    #dynamic_covariates = pd.read_sql_query(dynamic_query, con)
    #print('got dynamic covariates')

#    common_vars = dynamic_covariates['label'].value_counts() > 20
#    common_vars_names = common_vars[common_vars].index
#    dynamic_covariates_pruned = dynamic_covariates[dynamic_covariates['label'].isin(common_vars_names)]
#    print('preprocessing dyn cov')
#    interpolated_dyn_cov = preprocess_dynamic_covariates(dynamic_covariates_pruned)
#    print('done preprocessing dyn cov')

#    common_inputs = inputs_covariates['label'].value_counts() > 400
#    common_inputs_names = common_inputs[common_inputs].index
#    inputs_covariates_pruned = inputs_covariates[inputs_covariates['label'].isin(common_inputs_names)]

#    print('preprocessing inputs')
#    interpolated_input_infusion, interpolated_input_bolus = \
#        preprocess_input_events(inputs_covariates_pruned)
#    print('done preprocessing inputs')
    
#    print('merging everything')
    #all_inputs = interpolated_input_bolus.merge(interpolated_input_infusion, how='outer',on=ID_COLS + ['hours_in'])
    #all_inputs_chartlab = all_inputs.merge(interpolated_dyn_cov, how='outer',on=ID_COLS + ['hours_in'])
    #print('total size:' + str(all_inputs_chartlab.shape))
    
    #print('saving')
    #interpolated_input_bolus.to_csv('./data/processed/interpolated_input_bolus.csv')
    #interpolated_input_infusion.to_csv('./data/processed//interpolated_input_infusion.csv')
    #interpolated_dyn_cov.to_csv('./data/processed/interpolated_dyn_cov.csv')
    #print('done!')

    #print(static_covariates_query)


    import pydata_google_auth
    credentials = pydata_google_auth.get_user_credentials(['https://www.googleapis.com/auth/cloud-platform'])
    from google.cloud import bigquery
    client = bigquery.Client(project='mimic-reader', credentials=credentials)


    rewrite_data = {}
    rewrite_data['static'] = False
    rewrite_data['dynamic'] = False
    rewrite_data['outcomes'] = False
    rewrite_data['inputs'] = False
    
    if ~os.path.exists(root_dir + './data/external/static_vars.csv') or rewrite_data['static']:
        query_job = client.query(static_covariates_query)
        results = query_job.result().to_dataframe()
        results.to_csv(root_dir + './data/external/static_vars.csv', index=False)

    if ~os.path.exists(root_dir + './data/external/dynamic_vars.csv') or rewrite_data['static']:
        query_job = client.query(dynamic_covariates_query)
        results = query_job.result().to_dataframe()
        results.to_csv(root_dir + './data/external/dynamic_vars.csv', index=False)

    if ~os.path.exists(root_dir + './data/external/outcome_vars.csv') or rewrite_data['outcomes']:
            print('writing outcomes')
            query_job = client.query(outcomes_query)
            results = query_job.result().to_dataframe()
            results.to_csv(root_dir + './data/external/outcome_vars.csv', index=False)


    if ~os.path.exists(root_dir + './data/external/input_vars.csv') or rewrite_data['inputs']:
                print('writing inputs (treatments)')
                query_job = client.query(inputs_query)
                results = query_job.result().to_dataframe()
                results.to_csv(root_dir + './data/external/input_vars.csv', index=False)



    
    #print(results.head())

    #os.path.exists(path)  

    #print()
