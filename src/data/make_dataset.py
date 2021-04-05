# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from data_utils.data_utils import read_local_data

#from data_utils.preprocessing_utils import (
#    get_static_vars, get_dynamic_vars, add_icds_to_static_vars, get_drugs_timeseries_df,
#    split_treat_covariates, append_more_covariates
#)
from data_utils.preprocessing_utils import (
    preprocess_all
    #get_regular_timeseries, impute_dynamic_data
)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info('reading raw data, already saved locally')
    static_vars, dynamic_vars, outcome_vars, input_vars =\
        read_local_data(input_filepath)


    static_vars, dynamic_vars, outcome_vars, input_vars = \
        preprocess_all(static_vars, dynamic_vars, outcome_vars, input_vars)

    static_vars.to_csv('/home/joe/testbed/mimic_understander/data/interim/static_vars.csv')
    dynamic_vars.to_csv('/home/joe/testbed/mimic_understander/data/interim/dynamic_vars.csv')
    outcome_vars.to_csv('/home/joe/testbed/mimic_understander/data/interim/outcome_vars.csv')
    #input_vars.to_csv('/home/joe/testbed/mimic_understander/data/interim/input_vars.csv')

    #logger.info('drugs_df')
    #drugs_timeseries_df = get_drugs_timeseries_df(inpatients_records, drugs, clinical_vars, labs,
    #                                              output_filepath,
    #                                              max_num_drugs=100, logger=logger)

    # in this dataset, the treatment variable is in the drugs_df, so remove it from the drugs_df
    # drugs_treatments in this case is all the HIBOR, drugs_covariates is the rest to merge with dynamic df
    #logger.info('splitting drugs df into treatment and other covariateds')
    #drugs_treatments, drugs_covariates = split_treat_covariates(
    #    drugs_timeseries_df, 'HIBOR')

    #static_df = get_static_vars(inpatients_records,clinical_vars, labs, output_filepath)
    
    #logger.info('processnig labs and clinical variables')
    #labs_clinicalvars_df = get_dynamic_vars(clinical_vars, labs, output_filepath)
    
    #dynamic_df = append_more_covariates(labs_clinicalvars_df, drugs_covariates)

    #logger.info('getting icd data')
    #static_df_icds = add_icds_to_static_vars(
    #    static_df, codes_emergency, codes_inpatient)

    # add column for binary treatment indicator

    #logger.info('saving')
    #dynamic_df.to_csv(output_filepath + 'dynamic_vars.csv')
    #static_df_icds.to_csv(output_filepath + 'static_vars.csv')
    #drugs_treatments.to_csv(output_filepath + 'treatment_vars.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()
