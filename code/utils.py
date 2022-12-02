import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path.cwd() / 'output'


def parse_sid(row):
    _sid = row['sid'].astype(str)
    return float(_sid[:7]), float(_sid[7:11]), float(_sid[11:])


def generate_result(input_file, output_file):
    # 季度数据处理
    predict_data = pd.read_csv(input_file).sort_values('sid')[[
        'sid', 'label', 'predict_result'
    ]]
    predict_data[['ID', 'year',
                'season']] = predict_data.apply(parse_sid, axis=1,
                                                result_type="expand")
    # predict_data.to_csv('output/result_sort.csv')
    print(predict_data['predict_result'].describe())


    # templ_df['result'] = predict_data['label']
    submit_df = predict_data.loc[(predict_data['season'].astype(float) == 4)
                                & (predict_data['year'].astype(float) == 2021)]
    # print(predict_data['season'].describe())
    print(submit_df.head())

    submit_df[['ID', 'predict_result']].to_csv(output_file,
                                               index=False,
                                               header=False)
    try:
        os.makedirs('/result', exist_ok=True)
        submit_df[['ID', 'predict_result']].to_csv('/result/result.csv',
                                                index=False,
                                                header=False)
    except:
        print('ERROR: save /result/result.csv error')

                        
