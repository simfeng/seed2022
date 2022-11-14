import pandas as pd
import numpy as np

from pathlib import Path

base_dir = Path.cwd() / '初赛数据集'


def jks(data_type: str = 'train'):
    jks_file = base_dir / data_type / 'government_data' / 'zs_jks.csv'
    jks_df = pd.read_csv(jks_file,
                         parse_dates=['税款所属期起', '税款所属期止'
                                      ])[['公司ID', '税款所属期起', '税款所属期止', '实缴金额']]

    jks_df = jks_df.rename(columns={
        '公司ID': 'ID',
        '税款所属期起': 'start_date',
        '税款所属期止': 'end_date',
        '实缴金额': 'sjje'
    })
    print(len(jks_df['ID'].unique()))

    jks_df['start_date'] = jks_df['start_date'].to_numpy().astype(
        'datetime64[M]')

    jks_df['end_date'] = jks_df['end_date'].to_numpy().astype('datetime64[M]')
    jks_df['date'] = jks_df.apply(lambda row: pd.date_range(
        start=row['start_date'], end=row['end_date'], freq='MS'),
                                  axis=1)
    jks_df['date'] = jks_df['date'].map(lambda x: [str(d.date()) for d in x])
    jks_df['date_len'] = jks_df['date'].map(lambda x: len(x))

    jks_df['sjje_per_month'] = jks_df['sjje'] / jks_df['date_len']

    jks_df_per_month = jks_df.explode('date')

    jks_df_per_month['date'] = jks_df_per_month['date'].to_numpy().astype(
        'datetime64[M]').astype(str)
    # print(jks_df_per_month.head())
    print(jks_df.shape, np.sum(jks_df['date_len'] * jks_df['sjje_per_month']),
        np.sum(jks_df['sjje']))
    np.sum(jks_df_per_month['sjje_per_month'])
    print(jks_df_per_month.shape, np.sum(jks_df['date_len']),
        np.sum(jks_df['sjje']), np.sum(jks_df_per_month['sjje_per_month']))
    # jks_df_per_month.to_csv('a.csv')
    # jks_df.to_csv('b.csv')
    jks_df_per_month_group = jks_df_per_month.groupby(['ID', 'date'
                                                    ])['sjje_per_month'].sum()
    # jks_df_per_month_group.groupby('ID').count().to_csv('ID_count.csv')

    jks_df_per_month_group.to_csv(f'output/per_month_{data_type}.csv')
    print('jks_df_per_month_group: ', jks_df_per_month_group.shape)
    return jks_df_per_month_group, jks_df['ID'].unique()

def xssr(data_type='train'):
    """销售收入表"""
    xssr_file = base_dir / data_type / 'government_data' / 'sb_xssr.csv'


    xssr_df = pd.read_csv(xssr_file, parse_dates=['申报日期'])[[
        '公司ID', '本月数、本期数(按季申报)', '本年累计', '申报日期'
    ]]
    xssr_df = xssr_df.rename(
        columns={
            '公司ID': 'ID',
            '本月数、本期数(按季申报)': 'month_amount',
            '本年累计': 'year_amount',
            '申报日期': 'date'
        })
    xssr_df['date'] = xssr_df['date'].to_numpy().astype(
        'datetime64[M]').astype(str)
    # print(len(xssr_df['ID'].unique()))
    xssr_df = xssr_df.groupby(['ID', 'date']).agg({
        'month_amount': 'mean',
        'year_amount': 'mean'
    })
    xssr_df.to_csv(f'output/xssr_{data_type}.csv')
    return xssr_df


def merge_gover(xssr_df: pd.DataFrame,
                jks_df: pd.DataFrame,
                ID_list,
                start_date='2019-1-1',
                end_date='2021-12-1',
                data_type='train',
                sum_n_rows=3):
    print('jks columns:', jks_df)
    df_templ = pd.DataFrame(ID_list, columns=['ID'])

    df_templ.loc[:, 'date'] = [[
        str(d.date())
        for d in pd.date_range(start=start_date, end=end_date, freq='MS')
    ]] * len(df_templ)
    df_templ = df_templ.explode('date')
    print(df_templ.shape)
    df_templ['date'] = df_templ['date'].to_numpy().astype('datetime64[M]').astype(
        str)
    merge_jks = pd.merge(df_templ, jks_df, on=['ID', 'date'], how='left')

    merge_jks = pd.merge(merge_jks, xssr_df, on=['ID', 'date'], how='left')
    print(merge_jks.shape)
    # merge_jks = merge_jks.merge(biz_df, how='left', on='ID', indicator=True)
    # print(merge_jks.shape)
    merge_jks = merge_jks.sort_values(['ID', 'date'
                                    ]).reset_index().drop(columns=['index'])
    merge_jks['month_amount'] = merge_jks['month_amount'].fillna(
        merge_jks.groupby('ID')['month_amount'].transform('mean'))
    merge_jks['year_amount'] = merge_jks['year_amount'].fillna(
        merge_jks.groupby('ID')['year_amount'].transform('mean'))
    merge_jks['sjje_per_month'] = merge_jks['sjje_per_month'].fillna(
        merge_jks.groupby('ID')['sjje_per_month'].transform('mean'))
    # merge_jks.to_csv('merge_jks.csv')

    merge_jks['month_amount2'] = np.square(merge_jks['month_amount'])
    merge_jks['year_amount2'] = np.square(merge_jks['year_amount'])

    # 日期取做年、月，删除店铺ID，其余字段归一化
    # merge_jks.loc[:, ['year', 'month']] = merge_jks.apply(
    #     lambda row: pd.Series({'year': row['date'].split('-')[0], 'month': row['date'].split('-')[1]}), axis=1
    #     )
    merge_jks['year'] = merge_jks['date'].map(lambda x: x.split('-')[0])
    merge_jks['month'] = merge_jks['date'].map(lambda x: x.split('-')[1])
    gover_data = merge_jks.drop(columns=['ID', 'date'])
    gover_data = gover_data.groupby(
        gover_data.index //
        sum_n_rows).sum().reset_index().drop(columns=['index'])
    # gover_data['month_amount'] = gover_data['month_amount'] / gover_data['month_amount'].mean()
    # gover_data['year_amount'] = gover_data['year_amount'] / gover_data['year_amount'].mean()
    gover_data.index.rename('idx', inplace=True)

    gover_data.to_csv(f'output/gover_data_{data_type}_{sum_n_rows}.csv')


def power_data(data_type, sum_n_rows=3):
    dlsj_file = base_dir / data_type / 'power_data' / 'dlsj_df.csv'


    dlsj_df = pd.read_csv(dlsj_file,
                        parse_dates=['应收年月'])[['公司ID', '应收年月', '能源量', '应收金额']]
    dlsj_df = dlsj_df.rename(columns={
        '公司ID': 'ID',
        '应收年月': 'date',
        '能源量': 'nyl',
        '应收金额': 'ysje'
    })
    ID_list = dlsj_df['ID'].unique()
    dlsj_df['date'] = pd.to_datetime(
        dlsj_df['date'],
        format="%Y%m").to_numpy().astype('datetime64[M]').astype(str)
    # dlsj_df = dlsj_df['date'].unique()
    # dlsj_df['date'] = dlsj_df['date'].to_numpy().astype('datetime64[M]').astype(str)
    dlsj_df = dlsj_df.groupby(['ID', 'date']).agg(
        nyl=('nyl', 'sum'),
        ysje=('ysje', 'sum'),
    )
    # dlsj_df

    df_templ = pd.DataFrame(ID_list, columns=['ID'])
    df_templ.loc[:, 'date'] = [[
        str(d.date())
        for d in pd.date_range(start='2019-1-1', end='2021-12-1', freq='MS')
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

    merge_dlsj['nyl2'] = np.square(merge_dlsj['nyl'])
    merge_dlsj['ysje2'] = np.square(merge_dlsj['ysje'])

    merge_dlsj['year'] = merge_dlsj['date'].map(lambda x: x.split('-')[0])
    merge_dlsj['month'] = merge_dlsj['date'].map(lambda x: x.split('-')[1])
    print(merge_dlsj.head())
    merge_dlsj[['ID', 'date'
                ]].to_csv(f'output/ID_date_templ_{data_type}_{sum_n_rows}.csv')
    power_data = merge_dlsj.drop(columns=['ID', 'date'])
    power_data = power_data.groupby(
        power_data.index //
        sum_n_rows).sum().reset_index().drop(columns=['index'])
    power_data.index.rename('idx', inplace=True)
    power_data.to_csv(f'output/power_data_{data_type}_{sum_n_rows}.csv')

if __name__ == '__main__':
    data_type = 'test'
    gover = 1
    sum_n_rows = 3
    if gover: # 处理政府数据
        xssr_df = xssr(data_type=data_type)
        jks_df, ID_list = jks(data_type=data_type)
        merge_gover(xssr_df=xssr_df,
                    jks_df=jks_df,
                    ID_list=ID_list,
                    data_type=data_type,
                    sum_n_rows=sum_n_rows)
    else:
        power_data(data_type, sum_n_rows=sum_n_rows)