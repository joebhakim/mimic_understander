import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd

from features_utils.data_cleaning import (
    clean_static_df, clean_dynamic_df, clean_treat_df, clean_outcome_df,
    get_CXYT_for_modeling
)
from features_utils.data_preprocessing import (
    preprocess_CYXT, impute_dynamic_data, save_data_matrix
)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):

    print('cleaning dfs')
    dynamic_df = pd.read_csv(input_filepath + 'dynamic_vars.csv')
    # to skip annoying index line
    static_df = pd.read_csv(
        input_filepath + 'static_vars.csv').drop('Unnamed: 0', axis=1)

    treat_df = pd.read_csv(input_filepath + 'treatment_vars.csv')

    static_df_clean = clean_static_df(static_df)
    dynamic_df_clean = clean_dynamic_df(dynamic_df, static_df)
    treat_df_clean = clean_treat_df(treat_df, static_df)
    outcome_df_clean = clean_outcome_df(static_df)

    static_df_clean.to_csv(output_filepath + '/static_vars.csv')
    dynamic_df_clean.to_csv(output_filepath + '/dynamic_vars.csv')
    treat_df_clean.to_csv(output_filepath + '/treat_vars.csv')
    outcome_df_clean.to_csv(output_filepath + '/outcome_vars.csv')

    print('getting CYXT tensor for downstream modeling...')
    CXYT_df = get_CXYT_for_modeling(static_df_clean,
                           dynamic_df_clean,
                           treat_df_clean,
                           outcome_df_clean,
                           output_filepath)


    CXYT_df = preprocess_CYXT(CXYT_df)

    save_data_matrix(CXYT_df, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
