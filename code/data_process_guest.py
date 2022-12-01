import pandas as pd
import numpy as np


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

def jks(self):
    """缴款书"""

    data_type = self.data_type
    dataset_dir = self.dataset_dir
    output_dir = self.output_dir

    jks_file = dataset_dir / data_type / 'government_data' / 'zs_jks.csv'
    jks_df = pd.read_csv(jks_file, header=[1],
                         parse_dates=['税款所属期起', '税款所属期止'
                                      ])[['公司ID', '税款所属期起', '税款所属期止', '实缴金额']]

    jks_df = jks_df.rename(columns={
        '公司ID': 'ID',
        '税款所属期起': 'start_date',
        '税款所属期止': 'end_date',
        '实缴金额': 'sjje'
    })
    # print(len(jks_df['ID'].unique()))

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

    jks_r_df = jks_df_per_month.groupby(
        ['ID', 'date'])['sjje_per_month'].sum().reset_index()

    return jks_r_df, jks_df['ID'].unique()

def xssr(self):
    """销售收入表"""
    data_type = self.data_type
    dataset_dir = self.dataset_dir
    output_dir = self.output_dir
    xssr_file = dataset_dir / data_type / 'government_data' / 'sb_xssr.csv'


    xssr_df = pd.read_csv(xssr_file, parse_dates=['申报日期'], header=[1])[[
        '公司ID', '本月数、本期数(按季申报)', '本年累计', '申报属性代码(11正常申报; 21自查补报)',
        '更正类型代码(1新产生申报表; 5更正后新产生的申报表)', '申报日期'
    ]]
    xssr_df = xssr_df.rename(
        columns={
            '公司ID': 'ID',
            '本月数、本期数(按季申报)': 'month_amount',
            '本年累计': 'year_amount',
            '申报属性代码(11正常申报; 21自查补报)': 'sbsx',
            '更正类型代码(1新产生申报表; 5更正后新产生的申报表)': 'gzlx',
            '申报日期': 'date'
        })

    xssr_df['date'] = xssr_df['date'].to_numpy().astype(
        'datetime64[M]').astype(str)
    # print(len(xssr_df['ID'].unique()))
    xssr_df = xssr_df.groupby(['ID', 'date']).agg({
        'month_amount': 'mean',
        'year_amount': 'mean',
        'sbsx': 'max',
        'gzlx': 'max'
    })

    # xssr_df.to_csv(f'{output_dir}/xssr_{data_type}.csv')
    return xssr_df


def base_info(self):
    """企业基础信息表"""
    data_type = self.data_type
    dataset_dir = self.dataset_dir
    output_dir = self.output_dir
    base_file = dataset_dir / data_type / 'government_data' / 'sz_tmp_baseinfo_ent.csv'

    base_df = pd.read_csv(base_file, parse_dates=['设立日期'], header=[1])[[
        '公司ID', '组织', '批准设立机关', '设立日期', '注册资本金（万元）', '行业类别',
        '状态1,在营（开业）企业、2,吊销企业、3,注销企业、4,迁出，从地方提取。不是标准代码，根据地方实际情况增加的。EX02', '所属管区'
    ]]

    base_df = base_df.rename(
        columns={
            '公司ID': 'ID',
            '组织': 'zuzhi',
            '批准设立机关': 'jiguan',
            '设立日期': 'slrq',
            '注册资本金（万元）': 'zijin',
            '行业类别': 'hylb',
            '状态1,在营（开业）企业、2,吊销企业、3,注销企业、4,迁出，从地方提取。不是标准代码，根据地方实际情况增加的。EX02': 'zt',
            '所属管区': 'ssxq'
        })

    def parse_zijin(x):
        x = x.replace('万', '')
        try:
            x = float(x)
        except:
            x = np.nan
        return x
    base_df['zijin'] = base_df['zijin'].map(parse_zijin)
    base_df['zijin'] = base_df['zijin'].fillna(base_df['zijin'].mean())
    base_df['slrq'] = pd.DatetimeIndex(base_df['slrq']).year

    base_df = base_df.reset_index(drop=True).drop_duplicates(subset=['ID'],
                                                             keep='first',
                                                             inplace=False)

    # print(len(base_df['ID'].unique()))
    print(base_df.head(), base_df.shape)
    # base_df.to_csv(f'{output_dir}/base_info_{data_type}.csv')
    return base_df



def gover_data(self):

    base_df = base_info(self)
    xssr_df = xssr(self)
    jks_df, ID_list = jks(self)

    data_type = self.data_type
    suffix = self.suffix
    output_dir = self.output_dir

    # print('jks columns:', jks_df)

    # 根据企业ID和日期生成模板
    print('ID_list:', ID_list)
    df_templ = pd.DataFrame(ID_list, columns=['ID'])
    # df_templ.loc['date'] = np.nan
    print('df_templ:\n', df_templ.head())
    start_date = jks_df['date'].min()
    end_date = jks_df['date'].max()
    data_range = [
        str(d.date())
        for d in pd.date_range(start=start_date, end=end_date, freq='MS')
    ]
    print('date range:', start_date, end_date, df_templ.shape)
    df_templ['date'] = [data_range] * len(df_templ)
    # for i in range(len(df_templ)):
    #     df_templ.at[i, 'date'] = data_range
    df_templ = df_templ.explode('date')
    df_templ['date'] = df_templ['date'].to_numpy().astype(
        'datetime64[M]').astype(str)

    # df_templ.merge(base_df.drop_duplicates(subset=['ID']), how='left')
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


    merge_jks['year'] = merge_jks['date'].map(
        lambda x: x.split('-')[0]).astype(int)
    merge_jks['month'] = merge_jks['date'].map(
        lambda x: x.split('-')[1]).astype(int)

    merge_jks['season'] = merge_jks['month'].map(lambda x: MONTH_SEASON_MAP[x])

    if data_type == 'test':
        # 测试数据的话，把21年四季度的值置为nan, 后面在用均值填充
        merge_jks.loc[(merge_jks['season'] == 4) &
                      (merge_jks['year'].isin([2021])),
                      'sjje_per_month'] = np.nan

    merge_jks['sjje_per_month'] = merge_jks['sjje_per_month'].fillna(
        merge_jks.groupby('ID')['sjje_per_month'].transform('mean'))


    gover_data = merge_jks.groupby(['ID', 'year',
                                    'season']).sum().reset_index()

    gover_data['y_avg'] = gover_data.groupby(
        ['ID', 'season'])['sjje_per_month'].transform('mean')
    gover_data['year_y_avg'] = gover_data.groupby(
        ['ID', 'year'])['sjje_per_month'].transform('mean')
    gover_data['month_amount_avg'] = gover_data.groupby(
        ['ID', 'season'])['month_amount'].transform('mean')
    gover_data['year_month_amount_avg'] = gover_data.groupby(
        ['ID', 'year'])['month_amount'].transform('mean')


    gover_data = pd.merge(gover_data, base_df, on=['ID'], how='left')
    gover_data[gover_data['zijin'] == 0] = gover_data['zijin'].mean()

    gover_data['sid'] = gover_data.apply(
        lambda x:
        f"{int(x['ID'])}{int(x['year'])}{int(x['season']):02d}",
        axis=1)
    gover_data.set_index('sid', inplace=True)
    gover_data = gover_data.drop(columns=['ID', 'month'])
    gover_data['data_type'] = gover_data.apply(
        lambda x: 'predict'
        if x['year'] == 2021 and x['season'] == 4 else 'valid',
        axis=1)

    # categorical columns
    cate_columns = [
        'year', 'season', 'sbsx', 'gzlx', 'zuzhi', 'jiguan', 'zt', 'hylb',
        'jiguan', 'ssxq', 'slrq'
    ]
    for col in cate_columns:
        _map = {key: i for i, key in enumerate(gover_data[col].unique())}
        gover_data[col] = gover_data[col].map(_map).astype(int)

    if 'simple' in suffix:

        feature_list = [
            'month_amount', 'year_amount', 'season', 'zijin', 'year'
        ] + [
            'month_amount_avg', 'y_avg', 'year_y_avg', 'sjje_per_month',
            'year_month_amount_avg'
        ] + ['data_type']
        gover_data = gover_data[feature_list]

    if data_type == 'test':
        valid_data = gover_data[gover_data['data_type'] == 'valid'].drop(
            columns=['data_type'])
        print('valid_data shape:', valid_data.shape)
        valid_data.to_csv(f'{output_dir}/001_gover_data_valid{suffix}.csv')

    gover_data = gover_data.drop(columns=['data_type'])

    print('gover_data shape', gover_data.shape)
    gover_data.to_csv(
        f'{output_dir}/001_gover_data_{data_type}{suffix}.csv')

if __name__ == '__main__':
    data_type = 'train' # test: 处理测试数据， train: 处理训练数据
