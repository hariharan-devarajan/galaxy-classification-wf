#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import pandas as pd
import os
import shutil
import glob
import sys
import argparse
import random
#import threading

DATA_DIR      = ""
METADATA_FILE = 'training_solutions_rev1.csv'

def parse_args(args):
    
    parser = argparse.ArgumentParser(description="Enter description here")
    
    parser.add_argument(
        "-m","--max_img",type= int,default= 100,
        help="number of instances of each class"
        )
    parser.add_argument(
        "-s","--seed",type= int,default= 10,
        help="seed for random library"
        )
    return parser.parse_args(args)


def label_dataset(csv):        
    final_table_columns = ['GalaxyID', 'Class1.1', 'Class1.2', 'Class2.1', 'Class2.2', 'Class4.1', 'Class7.1', 'Class7.2', 'Class7.3']
    
    df = pd.read_csv(csv)
    df.drop(columns = df.columns.difference(final_table_columns), inplace=True)
    
    #create label column with default value
    df['Label'] = '$'
    
    #create conditions
    #conditions = (
    #    (df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.1'] >= 0.500),
    #    (df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.2'] >= 0.500),
    #    (df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.3'] >= 0.500),
    #    (df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.1'] >= 0.602),
    #    (df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.2'] >= 0.715) & (df['Class4.1'] >= 0.619)
    #)

    #create labels
    #labels = ['0', '1', '2', '3', '4']
    
    #apply the labels based on the conditions
    #df['Label'] = np.select(conditions, labels)
    
    df['Label'] = np.where((df['Class1.1'] >= 0.469) & (df['Class7.1'] >= 0.500), '0', df['Label'])
    df['Label'] = np.where((df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.2'] >= 0.500), '1', df['Label'])
    df['Label'] = np.where((df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.3'] >= 0.500), '2', df['Label'])
    df['Label'] = np.where((df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.1'] >= 0.602), '3', df['Label'])
    df['Label'] = np.where((df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.2'] >= 0.715) & (df['Class4.1'] >= 0.619), '4', df['Label'])

    #drop rows where Label is still '$'
    df = df[df['Label'] != '$']
    
    #return new_df
    return df[['GalaxyID', 'Label']]


def main():
    args = parse_args(sys.argv[1:])
    MAX_IMG    = args.max_img
    seed       = args.seed
    
    input_files = set(glob.glob("*.jpg"))
    df = label_dataset(METADATA_FILE)

    #create original filename with jpg extension
    df['orig_name'] = df['GalaxyID'].astype(str) + ".jpg"

    #keep the filenames that exist in the dir and are with labels 0..4
    #Checking labels is not really needed since the labeled dataset should only contain 0...4
    df = df[df['orig_name'].isin(input_files)]
    
    #drop >IMG_MAX and create new file name per label
    df_0 = df[df['Label'] == '0'].head(MAX_IMG)
    df_0['counter'] = range(len(df_0))
    
    df_1 = df[df['Label'] == '1'].head(MAX_IMG)
    df_1['counter'] = range(len(df_1))
    
    df_2 = df[df['Label'] == '2'].head(MAX_IMG)
    df_2['counter'] = range(len(df_2))
    
    df_3 = df[df['Label'] == '3'].head(MAX_IMG)
    df_3['counter'] = range(len(df_3))
    
    df_4 = df[df['Label'] == '4'].head(MAX_IMG)
    df_4['counter'] = range(len(df_4))
    
    #concat
    df = pd.concat([df_0, df_1, df_2, df_3, df_4])
    
    #split to train and validation
    train, val, test = np.split(df.sample(frac=1, random_state=seed), [int(len(df)*0.8), int(len(df)*0.9)])

    #create new name
    train['new_name'] = "train_class_" + train['Label'] + "_" + train['counter'].astype(str) + ".jpg"
    val['new_name'] = "val_class_" + val['Label'] + "_" + val['counter'].astype(str) + ".jpg"
    test['new_name'] = "test_class_" + test['Label'] + "_" + test['counter'].astype(str) + ".jpg"
    
    #concat
    df = pd.concat([train, val, test]).reset_index(drop=True)

    for ind in df.index:
        try:
            shutil.copy(df['orig_name'][ind], df['new_name'][ind])
        except Exception as e:
            print(e)
            break

if __name__ == '__main__':
    main()

