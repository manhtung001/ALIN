import pandas as pd
import numpy as np
import os
import ast

for path_csv in os.listdir('report'):
    print(path_csv)
    if 'csv' not in path_csv:
        continue
    df = pd.read_csv('report/' + path_csv)
    print(df)
    
    for column in df.columns:
        for index, row in df.iterrows():
            if column == 'Unnamed: 0':
                continue
            row[column] = ast.literal_eval(row[column])
            row[column] = round(np.mean(row[column]), 1)
            
    print()
    print(df)
    
   
    path_csv = path_csv.replace('.csv', '_avg.csv')
    df.to_csv('report/avg/' + path_csv, index=False)
    