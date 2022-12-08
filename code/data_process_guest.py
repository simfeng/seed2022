import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_transform = FunctionTransformer(np.log1p)


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

def gskz(self):
    """
    dj_nsrxx_kz.csv：工商快照表（复赛提供）
    headers: 公司ID,核算方式代码,从业人数,外籍从业人数,合伙人数,雇工人数,固定工人数,组织机构类型代码,会计制度（准则）代码,历史最大注册资本,投资总额,自然人投资比例,外资投资比例,国有投资比例,国有控股类型代码,总分机构类型代码
    """
    data_type = self.data_type
    dataset_dir = self.dataset_dir
    kz_file = dataset_dir / data_type / 'government_data' / 'dj_nsrxx_kz.csv'
    kz_df = pd.read_csv(kz_file, header=[1])[[
        '公司ID', '从业人数', '会计制度（准则）代码', '历史最大注册资本', '投资总额', '总分机构类型代码'
    ]]
    kz_df = kz_df.rename(
        columns={
            '公司ID': 'ID',
            '从业人数': 'cyrs',
            '会计制度（准则）代码': 'kjdm',
            '历史最大注册资本': 'lszdzb',
            '投资总额': 'tzze',
            '总分机构类型代码': 'zfjglxdm'
        })

    kz_df = kz_df.reset_index(drop=True).drop_duplicates(subset=['ID'],
                                                         keep='first',
                                                         inplace=False)
    kz_df['cyrs'] = kz_df['cyrs'].fillna(kz_df['cyrs'].mean())
    kz_df['lszdzb'] = kz_df['lszdzb'].fillna(kz_df['lszdzb'].mean())
    kz_df['tzze'] = kz_df['tzze'].fillna(kz_df['tzze'].mean())
    kz_df['kjdm'] = kz_df['kjdm'].fillna(kz_df['kjdm'].mode()[0])
    kz_df['zfjglxdm'] = kz_df['zfjglxdm'].fillna(kz_df['zfjglxdm'].mode()[0])

    return kz_df

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
    data_type = self.data_type
    kz_df = gskz(self)
    base_df = base_info(self)
    xssr_df = xssr(self)
    jks_df, ID_list = jks(self)
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
    gover_data = pd.merge(gover_data, kz_df, on=['ID'], how='left')

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

    if data_type == 'test':
        valid_data = gover_data[gover_data['data_type'] == 'valid'].drop(
            columns=['data_type'])
        print('valid_data shape:', valid_data.shape)
        valid_data.to_csv(f'{output_dir}/001_gover_data_valid{suffix}.csv')

    gover_data = gover_data.drop(columns=['data_type'])

    print('gover_data shape', gover_data.shape)
    output_file = f'{output_dir}/001_gover_data_{data_type}{suffix}.csv'
    gover_data.to_csv(output_file)
    return output_file

def data_process(self):
    self.data_type = 'train'
    train_output_file = gover_data(self)
    self.data_type = 'test'
    test_output_file = gover_data(self)
    vaild_output_file = test_output_file.replace('test', 'valid')
    train_df = pd.read_csv(train_output_file)
    test_df = pd.read_csv(test_output_file)
    valid_df = pd.read_csv(vaild_output_file)


    categorical_columns = [
        'year', 'season', 'sbsx', 'gzlx', 'zuzhi', 'jiguan', 'zt', 'hylb',
        'jiguan', 'ssxq', 'slrq', 'kjdm', 'zfjglxdm'
    ]

    log_scale_columns = [
        'sjje_per_month', 'month_amount', 'year_amount', 'y_avg', 'year_y_avg',
        'month_amount_avg', 'year_month_amount_avg', 'zijin', 'cyrs', 'lszdzb',
        'tzze'
    ]

    simple_feature_list = [
        'sid', 'month_amount', 'year_amount', 'season', 'zijin',
        'month_amount_avg', 'y_avg', 'year_y_avg', 'sjje_per_month',
        'year_month_amount_avg', 'cyrs', 'kjdm', 'lszdzb', 'tzze'
    ]

    if 'log_scale' in self.suffix:
        for col in log_scale_columns:
            train_df[col] = log_transform.fit_transform(train_df[col])
            test_df[col] = log_transform.fit_transform(test_df[col])
            valid_df[col] = log_transform.fit_transform(valid_df[col])

    if 'simple' in self.suffix:
        train_df = train_df[simple_feature_list]
        test_df = test_df[simple_feature_list]
        valid_df = valid_df[simple_feature_list]

    for col in train_df.columns:
        if col in categorical_columns:
            val_list = list(
                set(
                    list(train_df[col].unique()) +
                    list(test_df[col].unique())))
            print('\tcategorical ', col, val_list)
            _map = {key: i for i, key in enumerate(val_list)}
            train_df[col] = train_df[col].map(_map).astype(int)
            test_df[col] = test_df[col].map(_map).astype(int)
            valid_df[col] = valid_df[col].map(_map).astype(int)
        else:
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(test_df[col].mean())
            valid_df[col] = valid_df[col].fillna(valid_df[col].mean())

    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)
    valid_df.to_csv(vaild_output_file, index=False)



if __name__ == '__main__':
    data_type = 'train' # test: 处理测试数据， train: 处理训练数据
