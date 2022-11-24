import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path.cwd() / 'output'


def parse_sid(row):
    _sid = row['sid'].astype(str)
    return float(_sid[:7]), float(_sid[7:11]), float(_sid[11:])


def generate_result(job_id):
    # 季度数据处理
    predict_data = pd.read_csv(
        base_dir / f'003_result_{job_id}.csv').sort_values('sid')[[
            'sid', 'label', 'predict_result'
        ]]
    predict_data[['ID', 'year',
                'season']] = predict_data.apply(parse_sid, axis=1,
                                                result_type="expand")
    predict_data.to_csv('output/result_sort.csv')
    print(predict_data['predict_result'].describe())


    # templ_df['result'] = predict_data['label']
    submit_df = predict_data.loc[(predict_data['season'].astype(float) == 4)
                                & (predict_data['year'].astype(float) == 2021)]
    # print(predict_data['season'].describe())
    print(submit_df.head())

    submit_df[['ID', 'predict_result'
            ]].to_csv(f'output/004_submit_{job_id}.csv',
                        index=False,
                        header=False)

    # predict_data_season = predict_data
    if 0:
        predict_data_season = predict_data.drop(index=submit_df.index)

        line = plt.plot(range(0, 256),
                predict_data_season[0:256][['predict_result', 'label']]) # 黄色是label，蓝色是predict
        plt.legend(iter(line), ['predict_result', 'label'])
        print(predict_data_season.shape)

if __name__ == '__main__':
    generate_result(job_id='202211241554279541190')