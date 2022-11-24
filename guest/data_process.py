import pandas as pd
import numpy as np

from pathlib import Path

month_season_map = {
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

base_dir = Path.cwd() / '初赛数据集'


def jks(data_type: str = 'train'):
    """缴款书"""
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
    # jks_df_per_month.to_csv('output/000.csv')

    jks_df_per_month['date'] = jks_df_per_month['date'].to_numpy().astype(
        'datetime64[M]').astype(str)
    # print(jks_df_per_month.head())
    print(jks_df.shape, np.sum(jks_df['date_len'] * jks_df['sjje_per_month']),
        np.sum(jks_df['sjje']))
    # np.sum(jks_df_per_month['sjje_per_month'])
    print(jks_df_per_month.shape, np.sum(jks_df['date_len']),
        np.sum(jks_df['sjje']), np.sum(jks_df_per_month['sjje_per_month']))
    # jks_df_per_month.to_csv('a.csv')
    # jks_df.to_csv('b.csv')
    jks_r_df = jks_df_per_month.groupby(['ID', 'date'])['sjje_per_month'].sum()

    # jks_r_df.groupby('ID').count().to_csv('ID_count.csv')
    jks_r_df.to_csv(f'output/per_month_{data_type}.csv')
    print('jks_r_df: ', jks_r_df.shape)
    return jks_r_df, jks_df['ID'].unique()

def xssr(data_type='train'):
    """销售收入表"""
    xssr_file = base_dir / data_type / 'government_data' / 'sb_xssr.csv'


    xssr_df = pd.read_csv(xssr_file, parse_dates=['申报日期'])[[
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
    # print(xssr_df)

    xssr_df.to_csv(f'output/xssr_{data_type}.csv')
    return xssr_df

def base_info(data_type='train'):
    """企业基础信息表"""
    base_file = base_dir / data_type / 'government_data' / 'sz_tmp_baseinfo_ent.csv'

    base_df = pd.read_csv(base_file, parse_dates=['设立日期'])[[
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
    base_df['zijin'] = base_df['zijin'].map(parse_zijin).fillna(0)
    base_df['slrq'] = pd.DatetimeIndex(base_df['slrq']).year

    zt_map = {key: i for i, key in enumerate(base_df['zt'].unique())}
    hylb_map = {key: i for i, key in enumerate(base_df['hylb'].unique())}
    jiguan_map = {key: i for i, key in enumerate(base_df['jiguan'].unique())}
    ssxq_map = {key: i for i, key in enumerate(base_df['ssxq'].unique())}
    # print('zt_map:', zt_map)
    base_df['zt'] = base_df['zt'].map(zt_map)
    base_df['hylb'] = base_df['hylb'].map(hylb_map)
    base_df['jiguan'] = base_df['jiguan'].map(jiguan_map)
    base_df['ssxq'] = base_df['ssxq'].map(ssxq_map)

    base_df = base_df.reset_index(drop=True).drop_duplicates(subset=['ID'],
                                                             keep='first',
                                                             inplace=False)

    # print(len(base_df['ID'].unique()))
    print(base_df.head(), base_df.shape)
    base_df.to_csv(f'output/base_info_{data_type}.csv')
    return base_df



def merge_gover(xssr_df: pd.DataFrame,
                jks_df: pd.DataFrame,
                base_df: pd.DataFrame,
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
    df_templ = pd.merge(df_templ, base_df, on=['ID'], how='left')
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
    if 0: # z-score
        merge_jks[['month_amount', 'year_amount', 'zijin']] = merge_jks[[
            'month_amount', 'year_amount', 'zijin'
        ]].apply(lambda x: (x - np.mean(x)) / (np.std(x)))


    merge_jks['year'] = merge_jks['date'].map(
        lambda x: x.split('-')[0]).astype(int)
    merge_jks['month'] = merge_jks['date'].map(
        lambda x: x.split('-')[1]).astype(int)

    merge_jks['season'] = merge_jks['month'].map(lambda x: month_season_map[x])

    if data_type == 'test':
        # 测试数据的话，把21年四季度的值置为nan, 后面在用均值填充
        merge_jks.loc[(merge_jks['season'] == 4) &
                     (merge_jks['year'].isin([2021])),
                     'sjje_per_month'] = np.nan

    merge_jks['sjje_per_month'] = merge_jks['sjje_per_month'].fillna(
        merge_jks.groupby('ID')['sjje_per_month'].transform('mean'))
    # merge_jks.to_csv('merge_jks.csv')

    # merge_jks['month_amount2'] = np.square(merge_jks['month_amount'])
    # merge_jks['year_amount2'] = np.square(merge_jks['year_amount'])



    # print('==============\n', merge_jks)
    dropped_col = ['ID', 'date']
    if sum_n_rows == 3:
        dropped_col.append('month')
    gover_data = merge_jks.drop(columns=dropped_col)
    gover_data = gover_data.groupby(
        gover_data.index //
        sum_n_rows).sum().reset_index().drop(columns=['index'])
    gover_data['year'] = gover_data['year'] / sum_n_rows
    gover_data['season'] = gover_data['season'] / sum_n_rows

    # gover_data['ID'] = gover_data['ID'] / sum_n_rows
    # gover_data['month_amount'] = gover_data['month_amount'] / gover_data['month_amount'].mean()
    # gover_data['year_amount'] = gover_data['year_amount'] / gover_data['year_amount'].mean()
    gover_data.index.rename('idx', inplace=True)
    if 0:
        gover_data = pd.get_dummies(gover_data,
                                    columns=[
                                        'zt', 'hylb', 'jiguan', 'ssxq', 'zuzhi',
                                        'slrq', 'sbsx', 'gzlx'
                                    ])

    feature_list = ['month_amount', 'zijin',
                    'jiguan']  + ['sjje_per_month'] # 53177423.01 !!!!!!!!!!!
    gover_data = gover_data[feature_list]
    print('gover_data shape', gover_data.shape)
    gover_data.to_csv(f'output/gover_data_{data_type}_{sum_n_rows}.csv')

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

    categorical_field_list = [
        'fhxz', 'yhfl', 'schsxfl', 'scbc', 'ydlb', 'sshy'
    ]

    for field_name in categorical_field_list:
        _tmp_map = {
            key: i
            for i, key in enumerate(jcxx_df[field_name].unique())
        }
        jcxx_df[field_name] = jcxx_df[field_name].map(_tmp_map)

    jcxx_df = jcxx_df.reset_index(drop=True).drop_duplicates(subset=['ID'],
                                                             keep='first',
                                                             inplace=False)

    jcxx_df.to_csv(f'output/power_data_jcxx_{data_type}.csv')

    return jcxx_df


def power_data(data_type,
               sum_n_rows=3,
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
    merge_dlsj['season'] = merge_dlsj['month'].map(lambda x: month_season_map[x])
    # print(merge_dlsj.head())
    merge_dlsj[['ID', 'date'
                ]].to_csv(f'output/ID_date_templ_{data_type}_{sum_n_rows}.csv')
    power_data = merge_dlsj.groupby(
        ['ID', 'year', 'season']).sum().reset_index()
    power_data.index.rename('idx', inplace=True)
    power_data = pd.merge(power_data, jcxx_df, on=['ID'], how='left')
    if 0: # z-score
        power_data[['nyl', 'ysje']] = power_data[[
            'nyl', 'ysje'
        ]].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    power_data = power_data.drop(columns=['ID', 'month']).reset_index(drop=True)

    feature_list = ['yxrl', 'htrl', 'ssje', 'nyl', 'season']
    power_data = power_data[feature_list]

    print(f'power_data {data_type}:', power_data.shape)
    power_data.to_csv(f'output/power_data_{data_type}_{sum_n_rows}.csv')

def one_hot():

    def fix_missing_cols(in_train, in_test):
        missing_cols = set(in_train.columns) - set(in_test.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            in_test[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        in_test = in_test[in_train.columns]
        return in_test

    one_hot_columns_gover = [
        'zt', 'hylb', 'jiguan', 'ssxq', 'zuzhi', 'slrq', 'sbsx', 'gzlx',
        'season', 'year'
    ]
    norm_cols_gover = ['month_amount', 'year_amount', 'zijin']
    df_train_gover = pd.read_csv('output/gover_data_train_3_all.csv')
    df_test_gover = pd.read_csv('output/gover_data_test_3_all.csv')

    df_train_gover = pd.get_dummies(df_train_gover, columns=one_hot_columns_gover)
    df_test_gover = pd.get_dummies(df_test_gover, columns=one_hot_columns_gover)

    df_test_gover = fix_missing_cols(df_train_gover, df_test_gover)

    df_train_gover[norm_cols_gover] = df_train_gover[norm_cols_gover].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))
    df_test_gover[norm_cols_gover] = df_test_gover[norm_cols_gover].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))
    # df_train_gover.index.rename('idx', inplace=True)
    # df_test_gover.index.rename('idx', inplace=True)

    df_train_gover.to_csv('output/gover_data_train_3_all_one_hot.csv',
                          index=False)
    df_test_gover.to_csv('output/gover_data_test_3_all_one_hot.csv',
                         index=False)
    print('gover onehot size:', df_test_gover.shape, df_train_gover.shape)

    # for power
    one_hot_columns_power = [
        'year', 'season', 'fylb', 'ynlb', 'fhxz', 'yhfl', 'schsxfl', 'scbc',
        'ydlb', 'sshy'
    ]
    norm_cols_power = ['nyl', 'ysje', 'ssje', 'htrl', 'cy', 'yxrl']
    df_train_power = pd.read_csv('output/power_data_train_3_all.csv')
    df_test_power = pd.read_csv('output/power_data_test_3_all.csv')

    df_train_power = pd.get_dummies(df_train_power,
                                    columns=one_hot_columns_power)
    df_test_power = pd.get_dummies(df_test_power,
                                   columns=one_hot_columns_power)

    df_test_power = fix_missing_cols(df_train_power, df_test_power)

    df_train_power[norm_cols_power] = df_train_power[norm_cols_power].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))
    df_test_power[norm_cols_power] = df_test_power[norm_cols_power].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))

    df_train_power = df_train_power.drop(columns=['Unnamed: 0'])
    df_train_power.index.rename('idx', inplace=True)
    df_test_power = df_test_power.drop(columns=['Unnamed: 0'])
    df_test_power.index.rename('idx', inplace=True)

    print('power onehot size:', df_test_power.shape, df_train_power.shape)
    df_train_power.to_csv('output/power_data_train_3_all_one_hot.csv')
    df_test_power.to_csv('output/power_data_test_3_all_one_hot.csv')


if __name__ == '__main__':
    data_type = 'test' # test: 处理测试数据， train: 处理训练数据
    gover = 1 # 1： 处理政府数据， 0：处理电力数据
    sum_n_rows = 3 # 3 代表按季度处理数据
    if 1:
        if gover: # 处理政府数据
            base_df = base_info(data_type=data_type)
            xssr_df = xssr(data_type=data_type)
            jks_df, ID_list = jks(data_type=data_type)
            merge_gover(xssr_df=xssr_df,
                        jks_df=jks_df,
                        base_df=base_df,
                        ID_list=ID_list,
                        data_type=data_type,
                        sum_n_rows=sum_n_rows)
        else:
            power_data(data_type, sum_n_rows=sum_n_rows)

    # one_hot()