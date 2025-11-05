#!/usr/bin/python

__author__ = "Dipongkar Talukder"
# Pre-processing utils for HSI data

import ast
import numpy as np
import fig_utils
import configparser
import pandas as pd
import HSIBinReader
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

## Find the closest distance
def find_closest_index(df, col1, col2, target_x, target_y): # [REQUIRED]
    df['distance'] = ((df[col1] - target_x)**2 + (df[col2] - target_y)**2)**0.5
    closest_index = df['distance'].idxmin()
    closest_dist = df['distance'].min()
    df.drop(columns=['distance'], inplace=True)
    return closest_index, closest_dist, target_x, target_y


## Define zone based on the radius
def define_zones(row):  # [REQUIRED]
    if row['radius'] <= 50:
        return '0-50mm'
    elif row['radius'] > 50 and row['radius'] <=100:
        return '50-100mm'
    elif row['radius'] > 100 and row['radius'] <=130:
        return '100-130mm'
    elif row['radius'] > 130 and row['radius'] <=140:
        return '130-140mm'
    elif row['radius'] > 140 and row['radius'] <=150:
        return '140-150mm'
    elif row['radius'] > 150:
        return '>150nm'

## Load & parse ML config file
def load_config(config_file): #[REQUIRED]
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(config_file)
    config = {}
    for sec in cfg.sections():
        config_ = {k:ast.literal_eval(v) for k, v in cfg[sec].items()}
        config = {**config, **config_}
    return config


## Update configuration with optimized values from CV
def get_opt_setting(config, cv_bp):  # [REQUIRED]
    config['poly_wdim'] = cv_bp['feature_extract__poly_wdim']
    config['StartWvl'] = cv_bp['feature_extract__StartWvl']
    config['WvlRange'] = cv_bp['feature_extract__WvlRange']
    config['CWvlrange'] = cv_bp['feature_extract__CWvlrange']
    config['OverlapSize'] = cv_bp['feature_extract__OverlapSize']
    config['kernelridge_alpha'] = cv_bp['kernelridge__alpha']
    config['kernelridge_kernel'] = cv_bp['kernelridge__kernel']
    config['kernelridge_degree'] = cv_bp['kernelridge__degree']
    config['kernelridge_gamma'] = cv_bp['kernelridge__gamma']
    return config



## Get inference data - binary input
def get_bin_data(config, data):  # [REQUIRED?]
    bin_reader = HSIBinReader.BinReader(data)
    df = bin_reader.GetDataAsDataFrame()
    df.columns = df.columns.str.upper()
    df['radius'] = np.sqrt(df['X1']**2+df['Y1']**2)
    if config['exclude_edge']:
        df = df[df['radius']<=config['edge_exclusion_radius']]
    spectra_cols = [col for col in df.columns if 'BAND' in col]
    df_clean = df.dropna(subset=spectra_cols)
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


# Databricks spark output for selected applicaiton with reference or csv from ENVI
def table_reader(data, slot, config):  # [REQUIRED]
    # data: pandas dataframe from databricks sql query
    if config['hsi_format'].lower() == 'db':
        df = pd.read_csv(data)
        df = df[df['APPLICATION_NAME']==config['application_name']]  # Check if we want to do it differently
        df = df[df['WAFER']=='s'+str(slot)]
        df.rename(columns={'CENTER_X_mm': 'X1'}, inplace=True)
        df.rename(columns={'CENTER_Y_mm': 'Y1'}, inplace=True)
        df.columns = df.columns.str.replace('BAND_', 'Band')
    elif config['hsi_format'].lower() == 'csv':
        df = pd.read_csv(data)
        df.rename(columns={'WCC_X_Corrected_mm': 'X1'}, inplace=True)
        df.rename(columns={'WCC_Y_Corrected_mm': 'Y1'}, inplace=True)
        wavelength = np.linspace(config['start_wvl'], config['end_wvl'], config['num_band'])
        cols = [str(i) for i in wavelength]
        bands = ['Band'+str(i) for i in range(1,len(wavelength)+1)]
        col_rename = {i:j for i,j in zip(cols, bands)}
        df.rename(columns=col_rename, inplace=True)  ## Need to check if they are matching
    return df

# Support bin, csv (manual), databricks table format for HSI data
def data_reader(data, slot, config): # [REQUIRED]
    if config['hsi_format'].lower() == 'bin':
        bin_reader = HSIBinReader.BinReader(data)
        data = bin_reader.GetDataAsDataFrame()
    elif config['hsi_format'].lower() == 'db' or config['hsi_format'].lower() =='csv':
        data = table_reader(data, slot, config)
    return data

# Need to suport all three types of data - (1) Bin file (2) Databrickes table (3) manual csv file


def load_data(hsi_data, df_ref, config):   # [REQUIRED]
    matched_data = pd.DataFrame()
    for idx, row in hsi_data.iterrows():
        hsi_ = row['FileName']
        df_hsi = data_reader(hsi_, row['Slot'], config)  # But databricks will have all wafers in it (need to handle it differently)
        if config['target_param'] not in df_hsi.columns:  # Check if it is 'T1' or something else
                df_ref_slot = df_ref[df_ref['Slot']==row['Slot']]
                matched_data_ = match_hsi_ref(df_hsi, df_ref_slot, config) # No need if df_hsi contains reference
        else:
            matched_data_ = df_hsi
        # Need to filter nan-valued rows for reference & bands
        partition = row['Partition'].lower()
        if partition == 'mixed':
            train_, test_ = train_test_split(matched_data_,
                                            train_size=1-config['test_ratio'],
                                            test_size=config['test_ratio'],
                                            random_state=config['random_num'])
            train_['partition'] = 'train'
            test_['partition'] = 'test'
            matched_data_ = pd.concat([train_, test_], axis=0)
        else:
            matched_data_['partition'] = partition
        matched_data_['slot'] = row['Slot']
        matched_data = pd.concat([matched_data, matched_data_], axis=0)
    matched_data['radius'] = np.sqrt(matched_data['X1']**2+matched_data['Y1']**2)
    if config['exclude_edge']:
        matched_data = matched_data[matched_data['radius'] <=config['edge_exclusion_radius']]
    valid_train = False
    valid_test = False
    if matched_data['partition'].str.contains('train').any():
        valid_train = True
    else:
        print('Error: No train data')
    if matched_data['partition'].str.contains('test').any():
        valid_test = True
    else:
        print('Error: No test data')
    if valid_train and valid_test:
        valid_data = True
    else:
        valid_data = False
    matched_data.reset_index(drop=True, inplace=True)

    return matched_data, valid_data

def match_hsi_ref(hsi_data, ref_data, config): # [REQUIRED]
    x_list = []
    y_list = []
    m_list = []
    closest_idx_list = []
    closest_dist_list = []
    for index, row in ref_data.iterrows():
        closest_idx, closest_dist, x_, y_ = find_closest_index(hsi_data,
                                                               'X1', 'Y1',
                                                               row['X'],
                                                               row['Y'])
        closest_idx_list.append(closest_idx)
        closest_dist_list.append(closest_dist)
        x_list.append(x_)
        y_list.append(y_)
        m_list.append(row[config['target_param']])

    matched_data = hsi_data.iloc[closest_idx_list]
    matched_data.insert(0, config['target_param'], m_list)
    matched_data.insert(0, 'closest_dist', closest_dist_list)
    matched_data.insert(0, 'Y', y_list)
    matched_data.insert(0, 'X', x_list)
    matched_data.reset_index(inplace=True, drop=True)
    return matched_data
