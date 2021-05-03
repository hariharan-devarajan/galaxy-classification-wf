#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import pandas as pd
import os       
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from pathlib import Path
import sys
import argparse
from os import path
from IPython import embed
import time


DATA_DIR      = ""
METADATA_FILE = 'training_solutions_rev1.csv'

MAX_IMG_0 = 84
MAX_IMG_1 = 80
MAX_IMG_2 = 8
MAX_IMG_3 = 39
MAX_IMG_4 = 78

def parse_args(args):
    
    parser = argparse.ArgumentParser(description="Enter description here")
    
    parser.add_argument(
        "-i","--input_dir", default=".",
        help="directory with images"
        )
    parser.add_argument(
        "-o","--output_dir",default=".",
        help="directory for outputs"
        )
    parser.add_argument(
        "-m","--max_img",type= int,default= 500,
        help="number of instances of each class"
        )

    return parser.parse_args(args)


def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

        
def label_dataset(csv):        
    df = pd.read_csv(csv)
    df.drop(['Class1.3','Class1.3','Class3.1','Class3.2','Class4.2','Class5.1','Class5.2','Class5.3','Class5.4', 
            'Class6.1','Class6.2','Class8.1','Class8.2','Class8.3','Class8.4','Class8.5','Class8.6','Class8.7',
            'Class9.1','Class9.2','Class9.3','Class10.1','Class10.2','Class10.3','Class11.1','Class11.2','Class11.3',
            'Class11.4','Class11.5','Class11.6'],axis=1, inplace=True)

    new_df = pd.DataFrame(columns = ['GalaxyID', 'Label'])

    for i in range(len(df)):
        label = '$'
        if df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.1'] >= 0.500:
            label = '0'
        elif df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.2'] >= 0.500:
            label = '1'
        elif df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.3'] >= 0.500:
            label = '2'
        elif df.at[i,'Class1.2'] >= 0.430 and df.at[i,'Class2.1'] >= 0.602:
            label = '3'
        # class 4 corresponds to class SPIRALE, there is an error in the table in the paper
        # instead use what is provided in Table 1's description
        elif df.at[i,'Class1.2'] >= 0.430 and df.at[i,'Class2.2'] >= 0.715 and df.at[i,'Class4.1'] >= 0.619:
            label = '4' 
        else:
            continue
        if label != '$':
            insert(new_df,[df.at[i,'GalaxyID'], label])
        else:
            continue
            
    return new_df



    
def main():
    start = time.time()
    
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir
    MAX_IMG    = args.max_img
    
    f = os.listdir(input_path)
    input_files = [i for i in f if ".jpg" in i]
    df = label_dataset(METADATA_FILE)
    output_path = args.output_dir

    if not path.exists(output_path):
        os.makedirs(output_path)
    f = open("galaxy_id_files.txt", "a")
    
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0

    for i in range(len(df)):
        try:
            if df['Label'].iloc[i] == '0' and count1 < MAX_IMG_0: 
                if (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
                    f.write(str(df['GalaxyID'].iloc[i])+'.jpg' + "\n")
                    img = plt.imread( DATA_DIR + str(df['GalaxyID'].iloc[i])+'.jpg')
                mpimg.imsave('class_0_' + str(count1) + '.jpg', img)
                count1+=1
            elif df['Label'].iloc[i] == '1' and count2 < MAX_IMG_1: 
                if (str(df['GalaxyID'].iloc[i])+'.jpg') in input_files:
                    f.write(str(df['GalaxyID'].iloc[i]) +'.jpg'+ "\n")
                    img = plt.imread( DATA_DIR + str(df['GalaxyID'].iloc[i])+'.jpg')
                mpimg.imsave('class_1_' + str(count2) + '.jpg', img)
                count2+=1
            elif df['Label'].iloc[i] == '2' and count3 < MAX_IMG_2: 
                if (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
                    f.write(str(df['GalaxyID'].iloc[i])+'.jpg' + "\n")
                    img = plt.imread( DATA_DIR + str(df['GalaxyID'].iloc[i])+'.jpg')
                mpimg.imsave('class_2_' + str(count3) + '.jpg', img)
                count3+=1
            elif df['Label'].iloc[i] == '3' and count4 < MAX_IMG_3: 
                if (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
                    f.write(str(df['GalaxyID'].iloc[i]) +'.jpg'+ "\n")
                    img = plt.imread( DATA_DIR + str(df['GalaxyID'].iloc[i])+'.jpg')
                mpimg.imsave('class_3_' + str(count4) + '.jpg', img)
                count4+=1
            elif df['Label'].iloc[i] == '4' and count5 < MAX_IMG_4:
                if (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
                    f.write(str(df['GalaxyID'].iloc[i]) +'.jpg'+ "\n")
                    img = plt.imread(DATA_DIR + str(df['GalaxyID'].iloc[i])+'.jpg')
                mpimg.imsave('class_4_' + str(count5) + '.jpg', img)
                count5+=1
        except Exception as e:
            print(e)
            break

    exec_time = time.time() - start

    print('Execution time in seconds: ' + str(exec_time))
    return 

if __name__ == '__main__':
    main()