import pandas as pd
import numpy as np

from pathlib import Path

MONTH_SEASON_MAP = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4
}

base_dir = Path.cwd().parent / '初赛数据集'

suffix = '_simple_mean'

def power_data_jcxx(data_type):
    jcxx_file = base_dir / data_type / 'power_data' / 'dlsj_jcxx.csv'
    jcxx_df = pd.read_csv(jcxx_file)[[
        '公司ID', '负荷性质', '合同容量', '用户分类', '承压', '市场化属性分类', '生产班次', '用电类别',
        '运行容量', '所属行业'
    ]]

    jcxx_df = jcxx_df.rename(
        columns={
            '公司ID': 'ID',
            '负荷性质': 'fhxz',
            '合同容量': 'htrl',
            '用户分类': 'yhfl',
            '承压': 'cy',
            '市场化属性分类': 'schsxfl',
            '生产班次': 'scbc',
            '用电类别': 'ydlb',
            '运行容量': 'yxrl',
            '所属行业': 'sshy'
        })

    jcxx_df = jcxx_df.reset_index(drop=True).drop_duplicates(subset=['ID'],
                                                             keep='first',
                                                             inplace=False)

    jcxx_df.to_csv(f'output/power_data_jcxx_{data_type}.csv')

    return jcxx_df


def power_data(data_type,
               start_date='2019-1-1',
               end_date='2021-12-1'):

    jcxx_df = power_data_jcxx(data_type)

    dlsj_file = base_dir / data_type / 'power_data' / 'dlsj_df.csv'


    dlsj_df = pd.read_csv(dlsj_file, parse_dates=['应收年月'])[[
        '公司ID', '应收年月', '费用类别', '能源量', '应收金额', '实收金额', '用能类别'
    ]]
    dlsj_df = dlsj_df.rename(columns={
        '公司ID': 'ID',
        '应收年月': 'date',
        '费用类别': 'fylb',
        '能源量': 'nyl',
        '应收金额': 'ysje',
        '实收金额': 'ssje',
        '用能类别': 'ynlb',
    })
    ID_list = dlsj_df['ID'].unique()
    dlsj_df['date'] = pd.to_datetime(
        dlsj_df['date'],
        format="%Y%m").to_numpy().astype('datetime64[M]').astype(str)
    # dlsj_df = dlsj_df['date'].unique()
    # dlsj_df['date'] = dlsj_df['date'].to_numpy().astype('datetime64[M]').astype(str)
    print(len(dlsj_df['fylb'].value_counts()))

    dlsj_df = dlsj_df.groupby(['ID', 'date']).agg(
        nyl=('nyl', 'sum'),
        ysje=('ysje', 'sum'),
        ssje=('ssje', 'sum'),
        fylb=('fylb', lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else x.mean()),
        ynlb=('ynlb', lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else x.mean()))
    print(dlsj_df.head())

    df_templ = pd.DataFrame(ID_list, columns=['ID'])
    df_templ.loc[:, 'date'] = [[
        str(d.date())
        for d in pd.date_range(start=start_date, end=end_date, freq='MS')
    ]] * len(df_templ)
    df_templ = df_templ.explode('date')
    df_templ['date'] = df_templ['date'].to_numpy().astype(
        'datetime64[M]').astype(str)

    merge_dlsj = pd.merge(df_templ, dlsj_df, on=['ID', 'date'],
                        how='left').sort_values(
                            ['ID', 'date']).reset_index().drop(columns=['index'])
    merge_dlsj['nyl'] = merge_dlsj['nyl'].fillna(
        merge_dlsj.groupby('ID')['nyl'].transform('mean'))
    merge_dlsj['ysje'] = merge_dlsj['ysje'].fillna(
        merge_dlsj.groupby('ID')['ysje'].transform('mean'))

    # merge_dlsj['nyl2'] = np.square(merge_dlsj['nyl'])
    # merge_dlsj['ysje2'] = np.square(merge_dlsj['ysje'])

    merge_dlsj['year'] = merge_dlsj['date'].map(lambda x: x.split('-')[0]).astype('int')
    merge_dlsj['month'] = merge_dlsj['date'].map(
        lambda x: x.split('-')[1]).astype('int')
    merge_dlsj['season'] = merge_dlsj['month'].map(lambda x: MONTH_SEASON_MAP[x])

    power_data = merge_dlsj.groupby(
        ['ID', 'year', 'season']).sum().reset_index()

    power_data['ysje_avg'] = power_data.groupby(['ID', 'season'
                                                 ])['ysje'].transform('mean')
    power_data['nyl_avg'] = power_data.groupby(['ID', 'season'
                                                 ])['nyl'].transform('mean')


    power_data = pd.merge(power_data, jcxx_df, on=['ID'], how='left')

    power_data['sid'] = power_data.apply(
        lambda x: f"{int(x['ID'])}{int(x['year'])}{int(x['season']):02d}",
        axis=1)
    power_data.set_index('sid', inplace=True)
    power_data = power_data.drop(columns=['ID', 'month'])

    power_data['data_type'] = power_data.apply(
        lambda x: 'predict'
        if x['year'] == 2021 and x['season'] == 4 else 'valid',
        axis=1)

    # categorical columns
    cate_columns = [
        'year', 'season', 'fylb', 'ynlb', 'fhxz', 'yhfl', 'schsxfl', 'scbc',
        'ydlb', 'sshy'
    ]
    for col in cate_columns:
        _map = {key: i for i, key in enumerate(power_data[col].unique())}
        power_data[col] = power_data[col].map(_map).astype(int)

    if 'simple' in suffix:
        feature_list = ['nyl', 'ysje', 'ssje', 'season', 'ynlb', 'year'
                        ] + ['data_type'] + ['ysje_avg', 'nyl_avg']
        power_data = power_data[feature_list]

    if data_type == 'test':
        valid_data = power_data[power_data['data_type'] == 'valid'].drop(
            columns=['data_type'])
        print('valid_data shape:', valid_data.shape)
        valid_data.to_csv(f'output/002_power_data_valid{suffix}.csv')

    power_data = power_data.drop(columns=['data_type'])

    print(f'power_data {data_type}:', power_data.shape)
    power_data.to_csv(f'output/002_power_data_{data_type}{suffix}.csv')


if __name__ == '__main__':
    data_type = 'test' # test: 处理测试数据， train: 处理训练数据
    power_data(data_type='test')
    power_data(data_type='train')

