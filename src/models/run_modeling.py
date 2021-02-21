import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from get_clustering import print_unsupervised_clustering, print_supervised_clustering, print_cates_clustering

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    dynamic_df = pd.read_csv(input_filepath + 'dynamic_vars.csv')
    static_df = pd.read_csv(input_filepath + 'static_vars.csv')

    XYT = dynamic_df.merge(static_df,
                           on='patient_id', how='inner')

    # staic modeling
    CYT = XYT.groupby(by='patient_id').mean()

    np.random.seed(563)
    CYT_train, CYT_test, CYT_val = np.split(CYT.sample(
        frac=1), [int(.6*len(CYT)), int(.8*len(CYT))])
    
    
    CYT_splits = {'train':CYT_train, 'test':CYT_test, 'val':CYT_val}

    covariate_df_splits, yt_df_splits = {}, {}
    for split_name, split in CYT_splits.items():
        covariate_df_splits[split_name] = split.drop(['treated', 'death'], axis=1)
        yt_df_splits[split_name] = split[['treated', 'death']]

    print_unsupervised_clustering(covariate_df_splits, yt_df_splits, output_filepath)

    print_supervised_clustering(covariate_df_splits, yt_df_splits, output_filepath)

    print_cates_clustering(covariate_df_splits, yt_df_splits, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
