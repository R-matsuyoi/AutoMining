import sys
import os
from pandas import DataFrame, Series
import numpy as np
import warnings
import itertools
import datetime
import dateutil.relativedelta
import calendar
import re
import copy
import pandas as pd
from Mining.selfmodule.toolmodule.dataprep import na_inf_examine

pd.set_option('display.width', 100000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

try:
    from sklearn.externals import joblib
except:
    import joblib

from Mining.selfmodule.tablemodule.basestr import *
from Mining.selfmodule.toolmodule.dataprep import *
from Mining.selfmodule.toolmodule.privy_outredirect import privy_Autonomy, privy_fileattr_fun
from Mining.selfmodule.toolmodule.ignorewarn import filter_warnings
from Mining.selfmodule.toolmodule.datatrans import *
from Mining.selfmodule.toolmodule.strtotable import string_to_table, strtotable_valuefun, infos_to_table
flat = lambda x: set(sum([(i if isinstance(i, list) else [i]) for i in x], []))

seconds = 3


def month_mark(Info, method):
    """
    获取账期标识
    :param Info: 命名元组，单个模型信息
    :param method: 获取不同账期标识
    :return: 账期标识
    """
    mark_tabexp = '' if str(Info.month_tabexp) == str(Info.month_train) else f'{Info.month_tabexp}exp~'  # 与Info.table_train的结尾一致
    if method == 'train':      # 获取 ~[宽表探索账期~]训练账期
        mark = f"~{mark_tabexp}{Info.month_train}"
    if method == 'test':       # 获取 ~[宽表探索账期~]测试账期
        mark = f"~{mark_tabexp}{Info.month_test}"
    if method == 'predict':    # 获取 ~[宽表探索账期~]~预测账期
        mark = f"~{mark_tabexp}{Info.month_predict}"
    if method == 'traintest':  # 获取 ~[宽表探索账期~]训练测试[~测试账期]
        mark = f"~{mark_tabexp}{Info.month_train}~{Info.month_test}"
    return mark


def field_base_fun(s_field_base):
    """
    转换、检查并纠正基础数据字典
    :param s_base: 从excel粘贴的数据字典长字符串
    :return: DataFrame
    """
    s_base = eval(s_field_base)
    error = []
    field_base = string_to_table(s_base)  # 基础数据字典
    field_base = field_base.apply(lambda c: c.str.strip(' ').str.lower(), axis=1)  # 每列：剔除两端空格，转小写
    print(f'field_base: {len(field_base)}行')

    table_info = string_to_table(s_table_info)
    table_info = infos_to_table(s_table_info, col_eval=['tableXday_desc', 'tableXscore_desc'], col_index=None)  # 数据库基础表信息

    # 检查取值合规性
    colvalue_exam(field_base, 'dtype_classify', ['数值型', '类别型', '日期型'])
    colvalue_exam(field_base, 'field_src', ['原始', '手动衍生_py', '手动衍生_sql', '自动衍生_py', '其他'])
    colvalue_exam(field_base, 'into_model', ['删除', np.nan])
    colvalue_exam(field_base, 'must_remain', ['是', np.nan])
    colvalue_exam(field_base, 'is_cause', ['否', np.nan])
    colvalue_exam(field_base, 'remark', ['不参与自动衍生', '不参与当月自动衍生', '不参与近n月自动衍生', np.nan])
    for a in field_base.columns[field_base.columns.fillna('').str.contains('available')]:  # fillna可能存在全空列，列名也为nan
        colvalue_exam(field_base, a, ['不可用', np.nan])

    if_dup = field_base.duplicated()
    if if_dup.sum():
        print('剔除field_base中的完全重复的行')
        print(f"    剔除前 {len(field_base)}行")
        field_base = field_base[~if_dup]  # 删除重复记录
        print(f"    剔除后 {len(field_base)}行")

    # 判断列中是否存在不应有的重复（防止复制粘贴后忘记修改）
    for c in ['field_name', 'comment', 'formula']:
        if_col_dup = field_base[c][field_base[c].notnull() & (field_base[c] != "{'notago_tovalue': 1}")].value_counts()
        if_col_dup = if_col_dup[if_col_dup > 1]
        if len(if_col_dup):
            s = f"error field_base中的{c}列存在重复值:{if_col_dup.index}"
            error.append(s)

    # 从field_base中剔除“不可用”字段及基于“不可用”字段加工的字段
    field_base = field_base_delinvalid(field_base, 'available')

    mustcol_na = field_base[['field_name', 'comment', 'dtype_classify', 'field_src']].isnull().sum()
    mustcol_na = mustcol_na[mustcol_na > 0]
    if len(mustcol_na):
        s = f'error field_base下列字段存在缺失值：\n{mustcol_na}'
        error.append(s)

    if error:
        s = '\n\n' + '\n\n'.join(error)
        warnings.warn(s)

    # 检查时长类字段是否限制了不参与近n月自动衍生
    field_months = field_base.loc[field_base.field_name.str.contains('^monthsaready|^monthsremain') |
                                  (field_base.comment.str.contains('入网时长|在网时长|网龄') & (field_base.field_src.isin(['原始', '手动衍生_sql']))),
                                  ['field_name', 'comment', 'remark']]
    field_months = field_months.loc[field_months.remark.isnull()]
    if len(field_months):
        s = f"确定下列时长类字段的renmark是否限制 ‘不参与近n月自动衍生’：\n{field_months}"
        warnings.warn(s)

    # 检查日期型字段是否进入模型
    field_date = field_base.loc[((field_base.dtype_classify == '日期型') | (field_base.comment.str.contains('时间|日期'))) &
                                (field_base.field_src == '原始'), ['field_name', 'comment', 'into_model']]
    field_date = field_date.loc[field_date.into_model.isnull()]
    if len(field_date):
        s = f"确定下列日期类字段的into_model是否限制 ‘删除’：\n{field_date}"
        warnings.warn(s)

    # 检查分数原始字段是否进入模型、检查目标字段与分数字段（做入模型输入）的对应关系
    tablescore = table_info.loc[(table_info.s_field_base == s_field_base) & (table_info.tabletype == 'tableXscore'), 'tablename']
    score_into = DataFrame()
    col_score = []
    for i in tablescore:
        score_i = field_base.loc[(field_base.table == i) & (field_base.into_model != '删除'), ['field_name', 'comment', 'into_model']]
        score_into = pd.concat([score_into, score_i])
        col_score.extend(field_base[field_base.table == i].field_name)
    if len(score_into) > 0:
        raise Exception(f"下列分数字段可能入模，请确认打分时这些分数已经具备？\n{score_into}")
    flag_lack = set([re.sub('^score_', '', i) for i in col_score]) - set(field_base.field_name)
    if flag_lack:
        raise Exception(f"{s_field_base}：分数字段包括：{[f'score_{i}' for i in flag_lack]}，但是目标字段缺少：{flag_lack}, 请检查！")

    # 检查日表的账期字段是否在field_base中
    col_day = []
    tableXday_desc = table_info.loc[(table_info.s_field_base == s_field_base) & (table_info.tabletype == 'tableXday'), 'tableXday_desc']
    for i in tableXday_desc:
        # print(i)
        if type(i['col_day'])==list:
            col_day.append(i['col_day'][1])
        else:
            col_day.append(i['col_day'])
    col_day_lack = set(col_day) - set(field_base.field_name)
    if col_day_lack:
        raise Exception(f"在field_base中补充日表账期字段：{col_day_lack}")

    # 检查手动衍生_py字段是否缺少公式
    formula_null = field_base.loc[(field_base.field_src == '手动衍生_py') & field_base.formula.isnull(), ['field_name', 'comment', 'field_src', 'formula']]
    if len(formula_null) > 0:
        raise Exception(f"下列手动衍生_py字段未填充公式，请补充\n{formula_null}")

    # 检查dtype_db
    dtype_db_null = field_base.loc[field_base.dtype_db.isnull() & field_base.field_src.isin(['原始', '手动衍生_sql']), ['field_name', 'comment', 'field_src', 'dtype_db']]
    if len(dtype_db_null) > 0:
        raise Exception(f"下列字段未填充dtype_db，请补充\n{dtype_db_null}")

    # 检查table
    table_null = field_base.loc[field_base.table.isnull() & field_base.field_src.isin(['原始', '手动衍生_sql']), ['field_name', 'comment', 'field_src', 'table']]
    if len(table_null) > 0:
        raise Exception(f"下列字段未填充table，请补充\n{table_null}")
    return field_base


def field_base_delinvalid(field_base, col_avail):
    """
    从field_base中剔除“不可用”字段及基于“不可用”字段加工的字段
    :param field_base: DataFrame，数据字典
    :return: 剔除后的数据字典
    """
    field_manual = field_base.loc[(field_base.field_src == '手动衍生_py')].copy()
    field_auto = field_base.loc[(field_base.field_src == '自动衍生_py')].copy()

    col_invalid = list(field_base.loc[field_base[col_avail] == '不可用', 'field_name'])
    if len(col_invalid):
        print(f'删除{len(col_invalid)}个{col_avail}“不可用”字段：{col_invalid}')
        field_base = field_base[~field_base.field_name.isin(col_invalid)].copy()
        print(f'field_base: {len(field_base)}行\n')

    manual_need = manual_fun(DataFrame(), None, field_manual, if_compute=False)[1]
    col_invalid2 = [k for k, v in manual_need.items() if set(v) & set(col_invalid)]
    invalid_manual = field_base.loc[field_base.field_name.isin(col_invalid2), ['field_name', 'formula']]
    if len(invalid_manual):
        s = f'删除{len(invalid_manual)}个基于“不可用”字段加工的手动衍生_py字段：\n{invalid_manual}'
        print(s)
        field_base = field_base.loc[~field_base.field_name.isin(invalid_manual.field_name)]
        print(f'field_base: {len(field_base)}行')

    con3 = field_auto.field_name.str.replace('^.*?__|~.*?$', '').isin(col_invalid+col_invalid2)
    col_invalid3 = list(field_auto.loc[con3, 'field_name'])
    if len(col_invalid3):
        s = f'删除{len(col_invalid3)}个基于“不可用”字段加工的自动衍生_py字段：{col_invalid3}'
        print(s)
        field_base = field_base.loc[~field_base.field_name.isin(col_invalid3)]
        print(f'field_base: {len(field_base)}行')

    return field_base


def get_condition_col(condition):
    """
    获取筛选条件中涉及的字段名称列表(适用于sql的筛选条件，若需python-DataFrame版的筛选条件，可扩充)
    :param condition: 表述筛选条件的字符串
    :return: 筛选条件中的字段列表
    """
    def is_value(x):
        try:
            eval(x)
            return True
        except:
            return False

    def del_typechange(x):
        # 去除类型转换，只保留字段
        # 用::符号转换
        p1 = '|'.join('::.*' + Series(['$', '\)', ' ']))
        x = re.sub(p1, '', x)
        # 用cast(col as type) 转换
        if re.findall('cast(.* as .*)', x):
            p2 = '|'.join(Series([' ', '^']) + 'cast\(') + '|' + '|'.join('as .*\)' + Series([' ', '\)', '$']))
            x = re.sub(p2, '', x)
        return x.replace(' ', '')

    if condition is None:
        return []
    elif str(condition) == 'nan':
        return []

    # 拆分各条件分支
    con_connect = ' and | and\n|\nand |\nand\n| or | or\n|\nor |\nor\n'
    con_split = re.split(con_connect, condition)

    # (字段 操作符1 字段) 操作符2√ 取值：第2种操作符
    opera2 = ['>=', '<=', '=', '<>', '!=', '>', '<', ' not in ', ' in ', ' not like ', ' like ', ' is not null', ' is null']
    opera2_get = '|'.join([f".*?{i}" for i in opera2])
    opera2_rep = '|'.join([f"{i} *$" for i in opera2])
    field2 = [re.sub(opera2_rep, '', get_onlyvalue(re.findall(opera2_get, i))) for i in con_split]

    # (字段 操作符1√ 字段) 操作符2 取值：第1种操作符
    opera1 = ['\+', '-', '\*', '/', '\|\|', ]
    opera1_get = '|'.join(opera1)
    field1 = sum([re.split(opera1_get, i) for i in field2], [])

    # 去掉类型转换语句 re.sub:去掉形如right(user_id ::text, 1)的函数壳子
    field = [Series(del_typechange(re.sub('^.*\(|,.*\)$|as .*$', '', i))).str.strip('(|)| ').iloc[0] for i in field1]
    field = [i.lower() for i in field if not is_value(i)]
    return field


def get_col(Info, comment_valid):
    """
    获取加工宽表所需读取的 原始/手动衍生_sql 字段;
    及其中的数值型、类别型、日期型字段；
    需要加工的‘手动衍生_py’字段的数据字典
    :param Info: 单个模型信息，命名元组
    :param comment_valid: 数据字典，字段范围：结果宽表字段
    :return: col_need: 加工宽表所需要读取的 原始/手动衍生_sql 字段
             col_num: col_need中的数值型字段
             col_char: col_need中的类别型字段
             col_date: col_need中的日期型字段
             field_manual: 需要‘手动衍生_py’的字段的数据字典（结果宽表中的‘手动衍生_py’字段，宽表中字段基于的‘手动衍生_py’字段）
    """
    col_need0 = flat(comment_valid.base_init)  # 宽表字段基于的字段(原始/手动衍生_sql字段 + 可能的手动衍生_py字段)

    # 获取需要的手动衍生_py字段（保持数据字典中顺序）：宽表中的手动衍生_py字段 + 计算宽表中字段需要的手动衍生_py字段
    field_manual_exp = Info.comment_all.loc[Info.comment_all.field_src == '手动衍生_py']
    field_manual1 = comment_valid.loc[comment_valid.field_src == '手动衍生_py', 'field_name'].to_list()
    field_manual2 = [i for i in col_need0 if i in Info.comment_all.loc[Info.comment_all.field_src == '手动衍生_py', 'field_name'].values]
    field_manual = field_manual_exp.loc[field_manual_exp.field_name.isin(field_manual1 + field_manual2)]

    col_need0 = col_need0 | flat(Info.comment_all.loc[Info.comment_all.field_name.isin(
        field_manual.field_name), 'base_init'])  # col_need0中可能的手动衍生_py字段基于的字段
    col_need = [i for i in Info.comment_all.loc[Info.comment_all.field_src.isin(['原始', '手动衍生_sql']), 'field_name'].values if
                i in col_need0]  # 宽表字段基于的 原始/手动衍生_sql 字段

    col_num = [i for i in Info.comment_all.loc[Info.comment_all.dtype_classify == '数值型', 'field_name'] if i in col_need]
    col_char = [i for i in Info.comment_all.loc[Info.comment_all.dtype_classify == '类别型', 'field_name'] if i in col_need]
    col_date = [i for i in Info.comment_all.loc[Info.comment_all.dtype_classify == '日期型', 'field_name'] if i in col_need]
    return col_need, field_manual, col_num, col_char, col_date


def month_add(x, m_add):
    """
    yyyymm加减n个月
    :param x: 日期， 形如 yyyymm 或 yyyymmdd
    :param m_add: 增加的月份数，可以为负数
    :return:
    """
    d = str(x)[6:8]
    d_ad = int(d) if d else 1
    y, m = int(str(x)[:4]), int(str(x)[4:6])
    res_m = datetime.date(y, m, d_ad) + dateutil.relativedelta.relativedelta(months=m_add)
    res_m = str(res_m).replace('-', '')[:6] + str(d)
    return type(x)(res_m)


def month_list_fun(start, end=None, periods=None):
    """
    生成账期列表
    :param start、end: 起始账期，形如 yyyymm 或 yyyymmdd
    :param periods: 列表长度，正数则向后顺延，负数则向前追溯
    :return: 账期列表
    """
    if end is not None:
       if len(str(start)) != len(str(end)):
            s = '参数start、end的长度不一致！'
            raise Exception(s)
    ds = str(start)[6:8]
    ds_ad = '' if ds else '01'
    de = str(end)[6:8]
    de_ad = '' if de else '01'
    if end is not None:
        if periods is not None:
            print('忽略periods参数')
        start01 = str(start) + ds_ad
        end01 = str(month_add(end, 1)) + de_ad
        m_list = pd.date_range(start01,  end01, freq='M').map(lambda x: x.strftime('%Y%m')) + ds
        m_list = [i for i in m_list if i <= str(end)]
    elif periods is not None:
        start01 = str(start if periods > 0 else month_add(start, periods + 1)) + ds_ad
        m_list = pd.date_range(start01,  freq='M', periods=abs(periods)).map(lambda x: x.strftime('%Y%m')) + ds
    else:
        raise Exception('end、periods参数不可同时为None')
    return [type(start)(i) for i in m_list]


def get_month_min(mlist):
    """
    获得最小账期，若中间有缺失账期，则只取最大账期向前连续账期的最小账期
    :param mlist: 账期列表
    :return: 最小账期
    example：
    get_month_min(['202001', '202002', '202003', '202004'])  # 无缺失账期，返回 '202001'
    get_month_min(['202001', '202003', '202004'])  # 有缺失账期，返回'202003'
    """
    start = min(mlist)
    end = max(mlist)
    mlist_full = Series(month_list_fun(start, end)).sort_values(ascending=False)
    month_min = mlist_full.iloc[(~mlist_full.isin(mlist)).argmax() - 1]
    return month_min


def wherefun(x):
    """
    生成sql中的where语句
    :param x: 可以是None、np.nan、字符串、列表，分情况返回不同where语句
    :return: sql中where语句
    """
    if x is None:
        w = ''
    elif str(x) == 'nan':
        w = ''
    elif isinstance(x, str):
        w = f'where {x}' if x else ''
    elif x == []:
        w = ''
    elif isinstance(x, list):
        x_ad = Series(x)
        w = 'where ' + (' and '.join(x_ad[x_ad.notnull() & (x_ad != '')]))
        if re.sub(' *', '', w) == 'where':
            w = ''
    return w


def table_XY_fun(sqltype, tabletype, infos_all, s_field_base, month=None, nmg_yaxin=('', ''), sqlcomment='----'):
    """
    根据field_base的字段信息将各表的字段汇总到一起，包括所有所需特征、所有模型的目标字段
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param tabletype: 表的操作类型，create：创建分区表   insert：插入数据
    :param infos_all: 所有模型的模型，DataFrame
    :param s_field_base: field_base字符串变量字符
    :param month: 账期 yyyymm
    :param nmg_yaxin: 为适应内蒙古电信的亚信数据挖掘平台，其他环境无需理会
    :param sqlcomment: sql的注释字符
    :return: 打印或执行sql
    """
    # 用于sql中转换字符型 cast(字段 as ?)
    type_sql_str = type_py_sql[str]

    Info = to_namedtuple(infos_all.loc[infos_all.s_field_base == s_field_base].iloc[0])
    field_base = Info.field_base

    type_col_month = get_onlyvalue(field_base.loc[field_base.field_name == Info.col_month, 'dtype_classify'])
    if (type_col_month == '类别型') & (not isinstance(month, str)):
        print(f"field_base中{Info.col_month}字段为类别型，month参数取值为{type(month)}, 将month纠正为字符")
        month = str(month)
    elif (type_col_month == '数值型') & (isinstance(month, str)):
        print(f"field_base中{Info.col_month}字段为数值型，month参数取值为{type(month)}，将month纠正为数值型")
        month = int(month)
    elif type_col_month not in ('类别型', '数值型'):
        raise Exception(f"field_base中{Info.col_month}字段类型为{type_col_month}，应在('类别型', '数值型')中, 或扩充此处代码")

    table_info = infos_to_table(s_table_info, col_eval=['tableXday_desc', 'tableXscore_desc'], col_index=None)
    table_info['subquery'] = np.nan
    table_info = table_info[table_info.s_field_base == s_field_base]

    tables_actual = set(table_info.tablename)
    tables_expect = set(field_base.table[field_base.table.notnull()].unique()) - {'python', 'sql'}
    if tables_actual != tables_expect:
        raise Exception(
            f"表名核对有误：\n{' ' * 14}field_base.table：{tables_expect}; \ntableXs tableXday tableY中实际：{tables_actual}")

    tableXscore = table_info.loc[table_info.tabletype == 'tableXscore', ['tablename', 'alias', 'on', 'tableXscore_desc']]
    for i in range(len(tableXscore)):
        if tableXscore.tableXscore_desc.iloc[i]['if_unpivot']:
            # 分数表为竖排，需要将列转行的子查询语句放在left join后
            tablename = tableXscore.tablename.iloc[i]
            col_score = field_base[field_base.table == tablename].field_name
            flag_model = infos_all.set_index('col_target').model_name

            casewhen = [f"max(case when model_name='{flag_model.loc[re.sub('^score_', '', c)]}' then score else null end) {c}" for c in col_score]
            casewhen = ',\n    '.join(casewhen)

            # 获取分数表id
            s_alias = f"{tableXscore.alias.iloc[i]}."
            s_id = re.sub('^.*\.', '', [i for i in tableXscore.on.iloc[i].split('=') if s_alias in i][0])

            sql = sql_format(f"""
            (
                select 
                {Info.col_month}, {s_id},
                {casewhen}
                from {tablename}
                where {Info.col_month}={sql_value(month)}
                group by {Info.col_month}, {s_id}
            ) s_ad{i}
            """)
            table_info.loc[table_info.tablename == tablename, 'subquery'] = sql

    tableXs = table_info.loc[table_info.tabletype.isin(['tableXmain', 'tableXadd', 'tableXscore']), ['tablename', 'subquery', 'alias', 'on']]
    tableXday = table_info.loc[table_info.tabletype == 'tableXday', ['tablename', 'alias', 'on', 'tableXday_desc']]
    tableY = table_info.loc[table_info.tabletype == 'tableY', ['tablename', 'alias', 'on']]
    table_XY = Info.table_XY

    # 分区所需语句
    col_month_type = field_base.loc[field_base.field_name == Info.col_month, "dtype_db"].iloc[0]
    field_base_db = field_base.loc[field_base.field_src.isin(['原始', '手动衍生_sql'])]
    part, clear, insert = part_sql(table_XY, [(Info.col_month, month, col_month_type)])

    if tabletype == "create":
        # 创建分区表
        fieldcre = ',\n'.join(field_base_db.field_name + ' ' + field_base_db.dtype_db)
        create_sql = sql_format(f"""
        drop table if exists {table_XY};
        {my_sleep_sql()}
        create {nmg_yaxin[0]} table {table_XY} (
        {fieldcre})
        {nmg_yaxin[1].replace('%s', table_XY)}
        {part}
        """)
        sql_show(f"\n{sqlcomment} 创建分区表", create_sql)
        if sqltype == 'execute':
            my_sql_fun(create_sql, method='execute')
        elif sqltype == 'print':
            pass
        else:
            raise Exception(f'sqltype取值有误（{sqltype}），应该为execute、print!')

    elif tabletype == "insert":
        # 特征表的字段、关联语句
        fieldXs = []
        leftjoinXs = []
        tableXs = pd.concat([tableXs[tableXs.on == '主表'], tableXs[tableXs.on != '主表']])  # 确保主表在第一位
        for i in range(len(tableXs)):
            table1 = tableXs.tablename.iloc[i]  # 表名
            table2 = tableXs.subquery.where(tableXs.subquery.notnull(), tableXs.tablename).iloc[i]  # 表名或子查询语句
            alias = tableXs.alias.iloc[i]
            on = tableXs.on.iloc[i]
            fieldXs.append(', '.join(alias + '.' + field_base.loc[field_base.table == table1, 'field_name']))
            if i == 0:
                leftjoinXs.append(f"from (select * from {table2} where {Info.col_month}={sql_value(month)}) {alias}")
            elif i > 0:
                leftjoinXs.append(f"left join (select*from {table2} where {Info.col_month}={sql_value(month)}) {alias} on {on}")
        fieldXs = '\n,'.join(fieldXs)
        leftjoinXs = '\n'.join(leftjoinXs)

        # 日表的字段、关联语句
        fieldXday = []
        leftjoinXday = []
        for i in range(len(tableXday)):
            table = tableXday.tablename.iloc[i]
            alias = tableXday.alias.iloc[i]
            on = tableXday.on.iloc[i]
            tableXday_desc = tableXday.tableXday_desc.iloc[i]
            col_day = tableXday_desc['col_day']
            if type(col_day)==list:
                col_day=col_day[1]
            type_col_day = get_onlyvalue(field_base.loc[field_base.field_name == col_day, "dtype_classify"])

            day_next = str(month_add(month, int(tableXday_desc['monthadd']))) + str(tableXday_desc['dd'])  # 观察期最后账期次月dd日
            if type_col_day == '数值型':
                day_next = int(day_next)
            fieldXday.append(', '.join([f"{alias}.{c.replace('dayvalue_', '')} {c}" for c in field_base.loc[field_base.table == table, 'field_name']]))
            if tableXday_desc['day_datetye']==2:
                leftjoinXday.append(f"left join (select*from {table} where {col_day}={sql_value(day_next)}) {alias} on {on}")
            elif tableXday_desc['day_datetye']==1:
                leftjoinXday.append(f"left join (select*from {table} where {Info.col_month}={sql_value(day_next[:-2])} and {col_day}={sql_value(day_next[-2:])}) {alias} on {on}")
        fieldXday = ',\n'.join(fieldXday)
        fieldXday = f',{fieldXday}' if fieldXday else fieldXday
        leftjoinXday = '\n'.join(leftjoinXday)

        # 目标表的字段、关联语句 (按照一个目标字段表的设计)
        yname = get_onlyvalue(tableY.tablename)
        fieldY = ',' + ', '.join('y.' + field_base.loc[field_base.table == yname, 'field_name'])
        leftjoinY = f"left join (select * from {yname} where {Info.col_month}={sql_value(month)}) y on x0.{Info.col_id} = y.{Info.col_id}"

        # 关联所有field_base.field_name字段
        # 随机排序字段：为兼容解决内蒙古电信亚信平台 order by 随机函数 limit n 时，sql执行超时报错
        insert_sql = sql_format(f"""
        {clear}
        {insert}
        select {', '.join(field_base_db.field_name)} from 
        (
        select 
        {fieldXs}
        {fieldXday}
        {fieldY}
        ,row_number() over(order by {sqlrandfun}) rn
        {leftjoinXs}
        {leftjoinXday}
        {leftjoinY} 
        ) t 
        """)
        sql_show(f"\n{sqlcomment} 插入数据(请及时清空无用的历史账期数据):", insert_sql)
        if sqltype == 'execute':
            my_sql_fun(insert_sql, method='execute')
            cou_sql = f"""select {Info.col_month}, count(1) from {table_XY} group by {Info.col_month} order by {Info.col_month}"""
            sql_show("\n统计:", cou_sql)
            cou = my_sql_fun(cou_sql, method='read')
            print(cou)
        elif sqltype == 'print':
            pass
        else:
            raise Exception(f'sqltype取值有误（{sqltype}），应该为execute、print!')


def privy_basedatafun(step, Info, drop_midtable=True, nmg_yaxin=('', ''), sqltype='execute', sqlcomment='----'):
    """
    汇总近n月基础数据（训练、测试、或预测账期的近n月基础数据）
    :param step: 取值为'train'：生成训练账期的近n月基础数据，用于探索宽表字段，以及用于按照探索结果加工训练账期的宽表
                 取值为'test'：生成测试账期的的近n月基础数据，用于按照探索结果加工训测试账期的宽表
                 取值为'test'：生成预测账期的的近n月基础数据，用于按照探索结果加工训预测账期的宽表
    :param Info: 单个模型信息（命名元组）
    :param drop_midtable: 是否删除中间表
    :param nmg_yaxin: 为适应内蒙古电信的亚信数据挖掘平台，其他环境无需理会
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param sqlcomment: sql的注释字符

    :return: 结果表表名
    备注：根据本地项目的实际环境调整数据输入、输出形式
      Info.n_recent: 加工近n月基础数据的月份数
      Info.traintable_ratio: 仅step='train'时发生作用，训练账期基础数据中：负例样本数 = 负例样本数 * traintable_ratio
      Info.Pcase_limit: 仅step='train'时发生作用，表示正例样本上限（限定训练账期计数），实际正例量超出则随机抽样
      Info.timein_count: 仅step='train'时发生作用，表示时间内验证集的样本量，若为None，则不设置验证集
      Info.timeout_limit: 仅step='test'时发生作用，表示时间外测试账期样本数上限，实际样本量超出则随机抽样
      Info.Pcumsum_limit: 训练账期正例样本不足时，向前追溯若干账期d正例
    """
    start_time = datetime.datetime.now()
    if sqltype == 'execute':
        print(f"开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 用于sql中转换字符型 cast(字段 as ?)
    type_sql_str = type_py_sql[str]

    # --------------------------------- <editor-fold desc="打印参数"> --------------------------------------------------
    month = Info._asdict()[f'month_{step}']
    print(f'{sqlcomment} month: {month}')
    if str(month) == 'nan':
        s = f'Info.month_{step}取值为{month}!'
        raise Exception(s)

    type_col_month = get_onlyvalue(Info.field_base.loc[Info.field_base.field_name == Info.col_month, 'dtype_classify'])
    if (type_col_month == '类别型') & (not isinstance(month, str)):
        print(f"{sqlcomment} field_base中{Info.col_month}字段为类别型，month参数取值为{type(month)}, 将month纠正为字符")
        month = str(month)
    elif (type_col_month == '数值型') & (isinstance(month, str)):
        print(f"{sqlcomment} field_base中{Info.col_month}字段为数值型，month参数取值为{type(month)}，将month纠正为数值型")
        month = int(month)
    elif type_col_month not in ('类别型', '数值型'):
        raise Exception(f"{sqlcomment} field_base中{Info.col_month}字段类型为{type_col_month}，应在('类别型', '数值型')中, 或扩充此处代码")

    paras = {f'{sqlcomment} step': step,
             f'{sqlcomment} Info.model_name': Info.model_name,
             f'{sqlcomment} Info.n_recent': Info.n_recent}
    if step == 'train':
        paras[f'{sqlcomment} Info.Pcase_limit'] = Info.Pcase_limit
        paras[f'{sqlcomment} Info.traintable_ratio'] = Info.traintable_ratio
        paras[f'{sqlcomment} Info.Pcumsum_limit'] = Info.Pcumsum_limit
        paras[f'{sqlcomment} Info.timein_count'] = Info.timein_count
    elif step == 'test':
        paras[f'{sqlcomment} Info.timeout_limit'] = Info.timeout_limit

    paras_p = '\n    '.join([str(k) + ': ' + str(v) for k, v in paras.items()])
    print(f"{sqlcomment} 参数设置：\n    {paras_p}")

    type_col_target = get_onlyvalue(Info.field_base.loc[Info.field_base.field_name == Info.col_target, 'dtype_classify'])
    if (type_col_target == '类别型') & (not isinstance(Info.Pcase, str)):
        print(f"{sqlcomment} field_base中{Info.col_month}字段为类别型，Info.Pcase参数取值为{type(month)}, 将Info.Pcase纠正为字符")
        Info = choradd_namedtuple(Info, {'Pcase': str(Info.Pcase), 'Ncase': str(Info.Ncase)})
    elif (type_col_target == '数值型') & (isinstance(Info.Pcase, str)):
        print(f"{sqlcomment} field_base中{Info.col_month}字段为数值型，Info.Pcase参数取值为{type(month)}，将Info.Pcase纠正为数值型")
        Info = choradd_namedtuple(Info, {'Pcase': int(Info.Pcase), 'Ncase': int(Info.Ncase)})
    elif type_col_target not in ('类别型', '数值型'):
        raise Exception(f"{sqlcomment} field_base中{Info.col_month}字段类型为{type_col_month}，应在('类别型', '数值型')中, 或扩充此处代码")
    # </editor-fold> ---------------------------------------------------------------------------------------------------

    # -------------------------------- <editor-fold desc="内部函数"> ---------------------------------------------------
    def my_casecount_stat(tablename, method='Allcase', where=''):
        """
        统计表的数据分布
        :param tablename: 表名
        :param method: 取值为Allcase， 统计表的数据量，取值PNcase，统计表的正例分布情况
        :param where: 筛选条件
        :return: 取值为Allcase，返回{'count': ?}，
                 取值PNcase，返回 {'count': ?,'Pcount': ?, 'prop': ?}
        """
        if method == 'Allcase':
            sql_c = f'select  count(1) from {tablename} {where}'
            sql_show('数据量：', sql_c)
            count_res = my_sql_fun(sql_c, method='read')['count']
            dis = {'count': count_res.iloc[0]}
            if dis['count'] == 0:
                raise Exception(f'统计{tablename}总样本数，结果为0！')
        elif method == 'PNcase':  # cast: 防止将数据库中的1读取为1.0，与Pcase的取值对不上
            sql_c = f"select cast({Info.col_target} as {type_sql_str}) {Info.col_target}, count(1) from {tablename} {where} group by {Info.col_target}"
            sql_show('正负例分布：', sql_c)
            count_res = my_sql_fun(sql_c, method='read', index_col=Info.col_target)['count']
            pcount = dict(count_res).get(str(Info.Pcase), 0)
            ncount = dict(count_res).get(str(Info.Ncase), 0)
            if (pcount == 0) | (ncount == 0):
                s = f"\n统计{tablename}中{Info.col_target}字段正负例样本数有误，正例：{pcount}个，负例：{ncount}个" + \
                    f"\n数据库中字段取值: {[sql_value(i) for i in count_res.index]}" + \
                    f"\nInfo.Pcase: {sql_value(Info.Pcase)}, Info.Ncase: {sql_value(Info.Ncase)}"
                raise Exception(s)
            dis = {'count': count_res.sum(),
                   'Pcount': pcount,
                   'prop': round(pcount / count_res.sum(), 3)}  # 总量、正例量，正例比率
        return dis

    def recent_fun(mainsql, month_final):
        """
        获取近n月数据的sql
        :param mainsql: from后的主表或语句
        :param month_final: 观察期最后一个月的账期
        :return: sql语句字符串
        """
        m_s, m_e = month_list_fun(month_final, periods=-Info.n_recent)[0], month_final
        sql = sql_format(f"""
        select a.user_acct_month, a.data_use, b.*
        from {mainsql} a
        inner join (select * from {Info.table_XY} where {Info.col_month}>={sql_value(m_s)} and {Info.col_month}<={sql_value(m_e)}) b on a.{Info.col_id} = b.{Info.col_id}
        """)
        return sql

    def my_prop_exam(d1, d2, thred=5 / 100):
        """
        两份数据正例占比的比较，如果变动幅度超过阈值将发出警告
        :param d1: 第一份数据的正例占比
        :param d2: 第二份数据的正例占比
        :param thred: 正例占比变动幅度阈值
        :return: None
        """
        p1, p2 = Info._asdict()[d1], Info._asdict()[d2]
        diff = (p2['prop'] - p1['prop']) / p1['prop']
        s = f"    {d2} 较 {d1} 正例占比 变动幅度{format(diff, '.5%')} ({p2['prop']} 较 {p1['prop']})\n"
        print(s)
        if abs(diff) > thred:
            warnings.warn(s); time.sleep(seconds)
        # return diff
    # </editor-fold> ---------------------------------------------------------------------------------------------------

    # -------------------------------- <editor-fold desc="前置检查"> ---------------------------------------------------
    if sqltype == 'execute':
        print(f"\n------------------------------------- 检查各前置表 --------------------------------------------------- ")
        print(f'检查 {Info.table_XY}')    # 有个小bug待解决  只insert  202012时测试
        field_all = get_field(Info.table_XY).index
        print(f'    {len(field_all)} 列')
        mlist_all = get_month_count(Info.table_XY, Info.col_month, end=month)
        for month_i in reversed(month_list_fun(month, periods=-Info.n_recent)):
            print(f"    {month_i}账期： {mlist_all['count'].loc[str(month_i)]}行")
        print('\n')

    if step == 'predict':  # 只加工预测时会用到的字段
        # 当读取时无法使用condition筛选时，可能在读取后在python筛选，将使用条件字段
        # 预测打分时将用到dict_sortscore、col_out来排序和输出
        col_other = list(set(get_condition_col(Info.condition)) |
                         set(Info.dict_sortscore.keys() if Info.dict_sortscore else []) |
                         set(Info.col_out if Info.col_out else []))

        # 入模字段所需要的近三月基础字段
        field_comment = Info.comment_all
        comment_valid = field_comment.loc[
            ((field_comment.是否入模 == '是') & (field_comment.field_name != Info.col_target)) |
            (field_comment.field_name.isin(Info.col_mark + col_other))]
        col_need = get_col(Info, comment_valid)[0]  # 只加工入模字段所需的字段，及标识、条件、分数排序字段、分数输出字段
        # 过滤掉后续加工过程中添加的字段：'data_use', 'user_acct_month'
        field_base_new = Info.field_base.loc[Info.field_base.field_name.isin(col_need)]
        Info = choradd_namedtuple(Info, {'field_base': field_base_new})
    # </editor-fold> ---------------------------------------------------------------------------------------------------

    if sqltype == 'execute':
        if step == 'predict':
            cou_all = get_month_count(Info.table_XY, Info.col_month, start=month, end=month)
            Info = choradd_namedtuple(Info, {f'dis_predict_total': {'count': cou_all}})
            print(f"        dis_predict_total {Info.dis_predict_total}")
        else:
            dis_all = my_casecount_stat(Info.table_XY, method='PNcase')
            Info = choradd_namedtuple(Info, {f'dis_{step}_total': dis_all})
            print(f"    {f'dis_{step}_total'}: {dis_all}\n")
    # </editor-fold> ---------------------------------------------------------------------------------------------------

    field_base_db = Info.field_base.loc[Info.field_base.field_src.isin(['原始', '手动衍生_sql'])]

    table_info = string_to_table(s_table_info)
    tableY = table_info.loc[(table_info.s_field_base == Info.s_field_base) & (table_info.tabletype == 'tableY'), "tablename"].iloc[0]
    col_recent = list(field_base_db[((field_base_db.table != tableY) | (field_base_db.field_name == Info.col_target)) & (field_base_db.field_name != 'rn')].field_name)
    col_recent = ['user_acct_month', 'data_use'] + col_recent
    # 分区（账期）字段如果不是表中原有字段（如hive中），在中间表中加入分区（账期）字段
    col_month_add = "" if Info.col_month in col_recent else Info.col_month
    if col_month_add:
        col_recent = col_recent + [col_month_add]
    col_recent = ', '.join(col_recent)

    con_month = f"{Info.col_month}={sql_value(month)}"
    table_user = f'{prefix}mid_{Info.short_name}_user_{step}_{month}'      # 用户范围表表名
    table_recent = f'{prefix}mid_{Info.short_name}_recent_{step}_{month}'  # 近n月数据表表名
    table_model = ''
    if step == 'train':
        # ---------------------- <editor-fold desc="限定模型目标用户范围"> ---------------------------------------------
        # 添加/删除随机序号字段rn，以便划分训练集(data_train)、验证集（data_timein）
        rnself = f',row_number() over(order by {Info.col_month} desc, {sqlrandfun}) rnself' # if (step == 'train') and not np.isnan(Info.timein_count) else ''
        if str(Info.Pcumsum_limit) == 'nan':
            month_cumsum = [month]
        else:
            month_cumsum = month_list_fun(month, periods=-(Info.Pcumsum_limit+1))
            # 如果累计历史账期，修改con_month
            con_month = f"{Info.col_month}>={sql_value(month_cumsum[0])} and {Info.col_month}<={sql_value(month_cumsum[-1])}"
        table_model = f'{prefix}mid_{Info.short_name}_model_{step}_{month}'
        col_model = f"{Info.col_month}, {Info.col_id}, {Info.col_target}"
        sql_model = sql_format(f"""
                drop table if exists {table_model};
                {my_sleep_sql()}
                create {nmg_yaxin[0]} table {table_model} {nmg_yaxin[1].replace('%s', table_model)}as 
                select {col_model} {rnself} 
                from {Info.table_XY}
                {wherefun([con_month, Info.condition, f"{Info.col_target} is not null"])}""")
        print('\n')
        sql_show(f'{sqlcomment} 建表语句（限定{month}账期当月目标用户）：', add_printindent(sql_model) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_model, method='execute', indent='    ')

            print(f'统计{table_model}行列数')
            c_model = get_month_count(table_model, Info.col_month)
            # c_model = get_onlyvalue(c_model['count']) if len(c_model) > 0 else 0
            f_model = get_field(table_model)
            print(f"    行数：{dict(c_model['count'])}\n    列数：{len(f_model)}")
        # </editor-fold>

        # ---------------------- <editor-fold desc="划分训练/验证用户"> ------------------------------------------------
        timein_thred = Info.timein_count if str(Info.timein_count) != 'nan' else 0
        sql_user = sql_format(f"""
        drop table if exists {table_user};
        create {nmg_yaxin[0]} table {table_user} {nmg_yaxin[1].replace('%s', table_user)}as 
        (select acct_month user_acct_month, 'data_timein' data_use, * from {table_model} where rnself <= {timein_thred})
        union all
        (select acct_month user_acct_month, 'data_train'  data_use, * from {table_model} where rnself > {timein_thred} and {Info.col_target}={sql_value(Info.Pcase)} order by rnself limit {Info.Pcase_limit})
        union all
        (select acct_month user_acct_month, 'data_train'  data_use, * from {table_model} where rnself > {timein_thred} and {Info.col_target}={sql_value(Info.Ncase)} order by rnself limit {int(
            Info.Pcase_limit * Info.traintable_ratio)}) 
        """)
        print('\n')
        sql_show(f'{sqlcomment} 建表语句（划分训练/验证数据集）：', add_printindent(sql_user) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_user, method='execute', indent='    ')

            Info = choradd_namedtuple(Info, {f'dis_train_sample': my_casecount_stat(table_user, method='PNcase')})
            print(f"    dis_train_sample {Info.dis_train_sample}\n")
        # </editor-fold>

        # ---------------------- <editor-fold desc="关联近n月数据"> ----------------------------------------------------
        timein_recent = recent_fun(f"(select * from {table_user} where data_use='data_timein')", month)
        train_recent = [recent_fun(f"(select * from {table_user} where data_use='data_train' and {Info.col_month}={sql_value(i)})", i) for i in month_cumsum]
        train_recent = '\n\nunion all\n'.join(train_recent)
        sql_recent = sql_format(f"""
        drop table if exists {table_recent};
        create {nmg_yaxin[0]} table {table_recent} {nmg_yaxin[1].replace('%s', table_recent)}as
        select {col_recent} 
        from (
        {timein_recent}
        union all 
        {train_recent}
        ) t
        """)
        print('\n')
        sql_show(f'{sqlcomment} 关联近n月数据：', add_printindent(sql_recent) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_recent, method='execute', indent='    ')
        # </editor-fold>

    elif step == 'test':
        # ---------------------- <editor-fold desc="限定目标用户范围"> -------------------------------------------------
        where_con_test = wherefun([con_month, Info.condition, f"{Info.col_target} is not null"])
        if sqltype == 'execute':
            dis_test_model = my_casecount_stat(Info.table_XY, method='PNcase', where=where_con_test)
            Info = choradd_namedtuple(Info, {f'dis_{step}_model': dis_test_model})
            print(f"    {f'dis_{step}_model'} {dis_test_model}\n")
            if 'dis_train_model' in Info._asdict():
                my_prop_exam('dis_train_model', 'dis_test_model')
            else:
                print('    Info无dis_train_model，无法对比dis_train_model与dis_test_model的正例占比变动幅度\n')

            if Info.timeout_limit >= dis_test_model['count']:
                    print(f"Info.timeout_limit（{Info.timeout_limit}）>= 总数据量（{dis_test_model['count']}），更正Info.timeout_limit，无需抽样")
                    Info = choradd_namedtuple(Info, {'timeout_limit': np.nan})

        limit = '' if np.isnan(Info.timeout_limit) else f'\norder by rn limit {Info.timeout_limit}'
        where_model = where_con_test + limit

        sql_user = sql_format(f"""
        drop table if exists {table_user};
        create {nmg_yaxin[0]} table {table_user} {nmg_yaxin[1].replace('%s', table_user)}as 
        select acct_month user_acct_month, 'data_timeout' data_use, * from {Info.table_XY} 
        {where_model}
        """)
        print('\n')
        sql_show(f'{sqlcomment} 建表语句：', add_printindent(sql_user) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_user, method='execute', indent='    ')

            if ~np.isnan(Info.timeout_limit):
                Info = choradd_namedtuple(Info, {f'dis_test_sample': my_casecount_stat(table_user, method='PNcase')})
                print(f"    dis_test_sample {Info.dis_test_sample}\n")
                my_prop_exam('dis_test_model', 'dis_test_sample')
        # </editor-fold>

        # ---------------------- <editor-fold desc="关联近n月数据"> ----------------------------------------------------
        test_recent = recent_fun(f"(select * from {table_user} where data_use='data_timeout')", month)
        sql_recent = sql_format(f"""
        drop table if exists {table_recent};
        create {nmg_yaxin[0]} table {table_recent} {nmg_yaxin[1].replace('%s', table_recent)}as
        select {col_recent} 
        from (
        {test_recent}
        ) t
        """)
        print('\n')
        sql_show(f'{sqlcomment} 关联近n月数据：', add_printindent(sql_recent) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_recent, method='execute', indent='    ')
        # </editor-fold>
    elif step == 'predict':
        # ---------------------- <editor-fold desc="关联近n月数据"> ----------------------------------------------------
        where_con_predict = wherefun([con_month, Info.condition])

        if sqltype == 'execute':
            dis_pred_model = my_casecount_stat(Info.table_XY, method='Allcase', where=where_con_predict)
            Info = choradd_namedtuple(Info, {'dis_predict_model': dis_pred_model})
            print(f"    dis_predict_model {Info.dis_predict_model}\n")

        print(f'\n------------------- 关联近n月数据：{table_recent} ------------------------------- ')
        pre_recent = recent_fun(f"(select acct_month user_acct_month, 'data_predict' data_use, * from {Info.table_XY} {where_con_predict})", month)
        sql_recent = sql_format(f"""
        drop table if exists {table_recent};
        create {nmg_yaxin[0]} table {table_recent} {nmg_yaxin[1].replace('%s', table_recent)}as
        select {col_recent} 
        from (
        {pre_recent}
        ) t
        """)
        print('\n')
        sql_show(f'{sqlcomment} 关联近n月数据：', add_printindent(sql_recent) + ';')
        if sqltype == 'execute':
            my_sql_fun(sql_recent, method='execute', indent='    ')
        # </editor-fold>

    if sqltype == 'execute':
        print(f'统计{table_recent}行列数')
        c_recent = get_month_count(table_recent, Info.col_month)['count'].sum()
        f_recent = get_field(table_recent)
        print(f"    {c_recent}行，{len(f_recent)}列")

    # --------------------------------- <editor-fold desc="核验各账期数据量"> ------------------------------------------
    if sqltype == 'execute':
        print('\n\n--------------------------------- 核验各账期数据量  ---------------------------------------------- ')
        cols_group = f'user_acct_month, data_use, {Info.col_month}' + ('' if step == 'predict' else ', ' + Info.col_target)
        sql_monly_count = f'\nselect S, count(1) \nfrom {table_recent} \ngroup by S \norder by S'.replace('S', cols_group)
        sql_show('sql语句：', add_printindent(sql_monly_count))
        monly_count = my_sql_fun(sql_monly_count, method='read', indent='    ')
        print('\n', end='')
        print(f"结果：\n{monly_count}")

        # 检查目标字段取值
        if step != 'predict':
            con_now = monly_count.acct_month == monly_count.acct_month.max()
            yyy = monly_count.loc[con_now].groupby(Info.col_target)[Info.col_target].count().to_list()
            if (len(yyy) != 2) or (len(set(yyy)) != 1):
                raise Exception(f'{Info.col_target}字段取值分布有误，请检查！')

        c = monly_count.groupby(['user_acct_month', 'data_use', Info.col_month]).sum().reset_index(level=Info.col_month)
        c_unique = c.groupby(c.index).apply(lambda x: len(x['count'].unique()))
        if any(c_unique != 1):
            w = f'下列账期用户量不同，请检查！\n{c.loc[c_unique[c_unique != 1].index]}'
            warnings.warn(w); time.sleep(seconds)

        c_len = c.groupby(c.index).apply(lambda x: len(x))
        if any(c_len != Info.n_recent):
            w = f'下列账期不全（不等于{Info.n_recent}），请检查！\n{c.loc[c_len[c_len != Info.n_recent].index]}'
            warnings.warn(w); time.sleep(seconds)
    # </editor-fold> ---------------------------------------------------------------------------------------------------

    if drop_midtable:
        print(f'\n\n{sqlcomment} 删除中间表')
        sql_drop = ([f'drop table if exists {table_model}'] if table_model else []) + \
                   [f'drop table if exists {table_user}']
        for i in sql_drop:
            sql_show('', i + ';')
            if sqltype == 'execute':
                my_sql_fun(i, method='execute', indent='    ')

    print('\n', end='')
    print(f'{sqlcomment} 返回结果表名：{table_recent}')  # 根据本地环境的便利性决定返回方式

    file_info = f"{Info.model_wd_traintest}/Info~base_{step}.pkl"
    print(f'{sqlcomment} 将Info保存至{file_info}\n')
    privy_fileattr_fun(file_info, 'unhide')
    joblib.dump(Info._asdict(), file_info)
    privy_fileattr_fun(file_info)

    if sqltype == 'execute':
        end_time = datetime.datetime.now()
        print(f"结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time_cost = (end_time - start_time).seconds
        print(f"耗时：{time_cost} s")
    return (table_recent, Info)


def privy_get_trans(pipe, trans, indent=''):
    """
    获取流失线中的某个转换器
    :param pipe: 流水线对象
    :param trans: 转换器名称
    :param indent: print时的缩进
    :return:
    """
    gl_name = None
    def xj (pipe):
        nonlocal gl_name
        if 'Pipeline' in type(pipe).__name__:
            for i in pipe.named_steps.keys():
                xj(pipe.named_steps[i])
        elif 'FeatureUnion' in type(pipe).__name__:
            for i in pipe.transformer_list:
                xj(i[1])
        elif trans.lower() in type(pipe).__name__.lower():
            gl_name = pipe
            return pipe
    if trans in str(pipe):
        print(f'{indent}从流水线中获取{trans}')
        xj(pipe)
    else:
        print(f'{indent}未能从流水线中获取{trans}')
    return gl_name


# ----------------------------------------------------------------------------------------------------------------------

def field_pair_fun(field_list, n):
    """
    生成字段组合
    :param field_list: 字段列表,可以是list、Series等
    :param n: 每组字段包括的字段数量
    :return: 字段对tuple构成的list
    """
    return list(itertools.combinations(field_list, n))


def opera_pair_fun(data, field_pair, method_desc, col_add=None, comment=None, paste_sign='_'):
    """
    对字段组合进行加减乘除操作，其中加法、乘法不限制字段组合大小，减法和除法限制2个字段的组合
    :param data: DataFrame
    :param field_pair: 序列，序列中的元素为字段对
    :param method: 对字段对的运算方法，相加：add、相减：sub、相乘：mult、相除：div;   paste:交叉(按字符的形式拼接列）
    :param col_add: 输出DataFrame中需要额外添加的字段列表，如果为None，则为不添加
    :return: DataFrame
    """
    data = data.copy()
    method, desc = method_desc
    comment_ad = comment if comment else {}
    col_add = col_add if col_add else []
    col_out = [[c, comment_ad.get(c, c)] for c in col_add]
    if len(field_pair) == 0:
        print('参数field_pair长度为0，返回空的DataFrame')
        return DataFrame(index=data.index)
    for i in field_pair:
        col0, col1 = i[0:2]
        col_new = '__'.join((method,) + tuple(i))
        col_new_comment = '、'.join([comment_ad.get(c, c) for c in i]) + '：' + desc
        col_out.append([col_new, col_new_comment])
        if method == 'add':
            # data[col_new] = data[i].apply(sum, axis=1)
            data[col_new] = data[col0]
            for j in i[1:]:
                data[col_new] = data[col_new] + data[j]
        if method == 'sub':
            if len(i) > 2:
                raise Exception('字段对长度大于2！')
            data[col_new] = data[col0] - data[col1]
        if method == 'mult':
            data[col_new] = data[col0]
            for j in i[1:]:
                data[col_new] = data[col_new] * data[j]
        if method == 'div':
            if len(i) > 2:
                raise Exception('字段对长度大于2！')
            min_without_0 = data.loc[data[col1] > 0, col1].min()
            data.loc[data[col1] == 0, col1] = min_without_0
            data[col_new] = data[col0] / data[col1]
        if method == 'paste':
            i_type = data[list(i)].dtypes.astype(str)
            i_nonchar = list(i_type[i_type != 'object'].index)
            if i_nonchar:
                s = f'opera_pair_fun paste时存在非类别型字段，请确认: {i_nonchar}'
                warnings.warn(s); time.sleep(seconds)
            data[col_new] = data[col0].astype(str)
            for j in i[1:]:
                data[col_new] = data[col_new] + paste_sign + data[j].astype(str)
    comment_out = DataFrame(col_out, columns=['field_name', 'comment']).set_index('field_name')
    data_out = data[comment_out.index].copy()
    comment_out['dtype_classify'] = data_out.dtypes.astype(str).apply(lambda x: '类别型' if x == 'object' else '数值型')
    na_count = data_out.isnull().sum()
    na_count = na_count[na_count > 0]
    if len(na_count) > 0:
        warnings.warn(f'{len(na_count)}个字段出现缺失值：\n {na_count}'); time.sleep(seconds)
    return data_out, comment_out if comment else None


def recent_num_fun(data_recent, num_recent, Info, method_desc, col_add=None, comment=None):
    """
    基于近n月数据计算 均值、标准差、波动性、成长率等
    :param data_recent:近n月基础数据
    :param num_recent:待计算字段列表
    :param Info: 单个模型信息（命名元组）
    :param method: 方法，'均值': 'avg', '离散系数': 'sep', '波动性': 'wave', '成长率': 'grow',
                   '最大值': 'max', '最小值': 'min', '标准差': 'std'
    :param col_add: 输出DataFrame中需要额外添加的字段列表，如果为None，则为不添加
    :return: DataFrame
    """
    def base_fun(method):
        if method == 'avg':
            result = data_recent[num_recent].groupby(data_recent[Info.col_id]).mean()
        elif method == 'std':
            result = data_recent[num_recent].groupby(data_recent[Info.col_id]).std()
        elif method == 'min':
            result = data_recent[num_recent].groupby(data_recent[Info.col_id]).min()
        elif method == 'max':
            result = data_recent[num_recent].groupby(data_recent[Info.col_id]).max()
        elif method == 'avgago':
            result = data_recent.loc[data_recent[Info.col_month] != m_max, :]
            result = result[num_recent].groupby(result[Info.col_id]).mean()
        result.columns = method + '__' + result.columns
        return result

    method, desc = method_desc
    comment_ad = comment if comment else {}
    col_add = col_add if col_add else []
    input_output_com = [(i, method + '__' + i, comment_ad.get(i, i) + '：' + desc) for i in num_recent]
    data_recent = data_recent.copy()
    data_recent.index = data_recent[Info.col_id].values
    m_max = data_recent[Info.col_month].max()
    data_now = data_recent.loc[data_recent[Info.col_month] == m_max, :].copy()

    if method in ['avg', 'std', 'min', 'max']:
        result = base_fun(method)
    elif method == 'sep':
        result = pd.concat([base_fun('avg'), base_fun('std')], join='inner', axis=1)
        for col, col_new, _ in input_output_com:
            result[col_new] = result['std__' + col] / result['avg__' + col]
            result.loc[result['avg__' + col] == 0, col_new] = 0
    elif method == 'wave':
        result = pd.concat([base_fun('avg'), base_fun('min'), base_fun('max')], join='inner', axis=1)
        for col, col_new, _ in input_output_com:
            result[col_new] = (result['max__' + col] - result['min__' + col]) / result['avg__' + col]
            result.loc[result['avg__' + col] == 0, col_new] = 1
    elif method == 'grow':
        result = pd.concat([data_now[num_recent], base_fun('avgago')], axis=1)
        for col, col_new, _ in input_output_com:
            col_avgago = 'avgago__' + col
            result[col_new] = my_div(result[col], result[col_avgago])

    comment_out = DataFrame(input_output_com, columns=['', 'field_name', 'comment']).iloc[:, 1:].set_index('field_name')
    data_out = pd.concat([result[comment_out.index], data_now[col_add]], join='inner', axis=1)
    comment_out['dtype_classify'] = data_out.dtypes.astype(str).apply(lambda x: '类别型' if x == 'object' else '数值型')
    na_inf_examine(data_out)
    return data_out, comment_out if comment else None


def recent_morecnt_fun(data_recent, Info, thre_dict, col_add=None, comment=None):
    """
    计算近n月字段大于某阈值的月份数
    :param data_recent: 近n月基础数据
    :param Info: 单个模型信息（命名元组）
    :param thre_dict: 记录各字段各种阈值的字典，
                      键：记录阈值的名称，对应列名 more键cnt__原始字段名
                      值：记录每个字段该阈值的Series
    :param col_add: 输出DataFrame中需要额外添加的字段列表，如果为None，则为不添加
    :return: DataFrame
    """
    comment_ad = comment if comment else {}
    col_add = col_add if col_add else []
    col_out = [[c, comment_ad.get(c, c)] for c in col_add]
    data_recent = data_recent.copy()
    data_recent.index = data_recent[Info.col_id].values
    m_max = data_recent[Info.col_month].max()
    data_now = data_recent.loc[data_recent[Info.col_month] == m_max, :].copy()
    for k in thre_dict.keys():
        print(k)
        thre = thre_dict[k]
        for col in thre.index:
            data_recent['i_bool'] = data_recent[col] > thre[col]
            col_new = k[0] + '__' + col
            col_new_comment = f"{comment_ad.get(col, col)}：{k[1]}"
            data_now[col_new] = data_recent['i_bool'].groupby(data_recent[Info.col_id]).sum()
            col_out.append([col_new, col_new_comment])
    comment_out = DataFrame(col_out, columns=['field_name', 'comment']).set_index('field_name')
    data_out = data_now[comment_out.index].copy()
    comment_out['dtype_classify'] = data_out.dtypes.astype(str).apply(lambda x: '类别型' if x == 'object' else '数值型')
    na_inf_examine(data_out)
    return data_out, comment_out if comment else None


def recent_valuecnt_fun(data_recent, char_recent, Info, col_add=None, comment=None):
    """
    计算近n月字段取某值的月份数
    :param data_recent: 近n月基础数据
    :param char_recent: 字段名~取值名构成的列表
    :param Info: 单个模型信息（命名元组）
    :param col_add: 输出DataFrame中需要额外添加的字段列表，如果为None，则为不添加
    :return: DataFrame
    """
    data_recent = data_recent.copy()
    recent_com = f'近{len(data_recent[Info.col_month].value_counts())}月取值为%s的月份数'
    data_recent.index = data_recent[Info.col_id].values
    m_max = data_recent[Info.col_month].max()
    data_now = data_recent.loc[data_recent[Info.col_month] == m_max, :].copy()
    suf_valuecnt = 'valuecnt__'
    comment_ad = comment if comment else {}
    col_add = col_add if col_add else []
    col_out = [[c, comment_ad.get(c, c)] for c in col_add]
    for i in char_recent:
        col, value = i.split('~')
        col_new = suf_valuecnt + col + '~' + value
        col_new_comment = comment_ad.get(col, col) + '：' + recent_com % value
        data_now[col_new] = (data_recent[col] == value).groupby(data_recent[Info.col_id]).sum()
        col_out.append([col_new, col_new_comment])
    comment_out = DataFrame(col_out, columns=['field_name', 'comment']).set_index('field_name')
    data_out = data_now[comment_out.index].copy()
    comment_out['dtype_classify'] = data_out.dtypes.astype(str).apply(lambda x: '类别型' if x == 'object' else '数值型')
    na_inf_examine(data_out)
    return data_out, comment_out if comment else None


def now_greatest_fun(data, greatest_dict, Info, col_add=None):
    """
    基于当期数据，计算几个字段中取值最大的字段
    :param data: 输入数据
    :param greatest_dict: 字典
                           键：结果字段名
                           值：字典，记录参与比较的每个字段获得最大值时对应的结果取值
                           如：{'gprs_flow_mode': {'gprs_flow_23g': '23G', 'gprs_flow_4g': '4G', 'gprs_flow_5g': '5G'}}
    :param Info: 单个模型信息（命名元组）
    :param col_add:  输出DataFrame中需要额外添加的字段列表，如果为None，则为不添加
    :return: DataFrame
    """
    def greatest_fun(x):
        x = x.fillna(0)
        x.index = k_v.values()
        return x.idxmax()
    data = data.copy()
    col_out = col_add.copy() if col_add else []
    for c, k_v in greatest_dict.items():
        data[c] = data[k_v.keys()].apply(greatest_fun, axis=1)
    col_out.extend(greatest_dict.keys())
    data_greatest = data[col_out].copy()
    na_inf_examine(data_greatest)
    return data_greatest, None
# ----------------------------------------------------------------------------------------------------------------------


def manual_fun(data_recent, Info, field_manual, if_compute=True):
    """
    提前手动加工字段，将作为输入字段进入下一步'自动衍生_py'环节
    :param data_recent: DataFrame，近n月基础数据
    :param Info: 单个模型信息（命名元组）
    :param field_manual: 手动衍生_py字段信息
    :param if_compute: True:加工手动衍生字段；False: 不加工，仅仅为了获取来源字段
    :return: 在data_recent中添加完手动衍生_py字段的结果DataFrame
    """
    if len(field_manual) == 0:
        return data_recent, {}

    dels = {'days_month', 'current_date'}
    pa = '（|）|\(|\)| |\'|"|’|”'

    field_manual = field_manual.copy()
    data_recent = data_recent.copy()
    col_need_ad = dict()

    if if_compute:
        print(f"手动衍生_py{len(field_manual)}个字段: {dict(field_manual.set_index('field_name').comment)}")  # 作为宽表探索的输入数据
        field_manual.formula = field_manual.formula.fillna('')
        if 'days_month' in str(field_manual.formula):  # 当月的总天数
            data_recent['days_month'] = data_recent[Info.col_month].apply(lambda x:  calendar.monthrange(int(x[:4]), int(x[4:]))[1])
        if 'current_date' in str(field_manual.formula):  # 当月的日期 yyyymm01
            data_recent['current_date'] = pd.to_datetime(data_recent[Info.col_month].astype(str) + '01')

    for i in range(len(field_manual)):
        field_name_i, formula_i = tuple(field_manual[['field_name', 'formula']].iloc[i])
        if if_compute:
            print(f"{field_name_i}: {formula_i}")
        if 'notago_tovalue' in formula_i:
            f = re.sub('^ago_', '', field_name_i)
            if if_compute:
                mmax = data_recent[Info.col_month].max()
                data_recent[field_name_i] = data_recent[f]
                data_recent.loc[data_recent[Info.col_month] == mmax, field_name_i] = eval(formula_i)['notago_tovalue']
            col_need_ad[field_name_i] = [f]
        elif '/' in formula_i:
            d1, d2 = formula_i.replace(' ', '').split('/')
            if if_compute:
                data_recent[field_name_i] = my_div(data_recent[d1].fillna(0), data_recent[d2].fillna(0))
            col_need_ad[field_name_i] = [i for i in [d1, d2] if i not in dels]
        elif '+' in formula_i:
            adds = formula_i.replace(' ', '').split('+')
            if if_compute:
                data_recent[field_name_i] = 0
                for a in adds:
                    data_recent[field_name_i] = data_recent[field_name_i] + data_recent[a].fillna(0)
            col_need_ad[field_name_i] = [i for i in adds if i not in dels]
        elif '-' in formula_i:
            d1, d2 = formula_i.replace(' ', '').split('-')
            if if_compute:
                if 'current_date' in formula_i:
                    data_recent[field_name_i] = np.round((data_recent[d1] - data_recent[d2]).apply(lambda x: x.days) / 30)
                    con_count = data_recent[field_name_i].isnull()
                    if con_count.sum():
                        na_to = -99999
                        notna_min = data_recent.loc[~con_count, field_name_i].min()
                        print(f'    将 {field_name_i} 字段的 {con_count.sum()} 个缺失值赋值为: {na_to} （非缺失的最小值{notna_min}）')
                        data_recent.loc[con_count, field_name_i] = na_to
                else:
                    data_recent[field_name_i] = data_recent[d1].fillna(0) - data_recent[d2].fillna(0)
            col_need_ad[field_name_i] = [i for i in [d1, d2] if i not in dels]
        elif field_name_i.startswith('paste_'):
            col_pair = [re.split('，|,', re.sub(pa, '', formula_i))]
            if if_compute:
                data_recent[field_name_i] = opera_pair_fun(data_recent, col_pair, ('paste', '交叉'), col_add=None, comment=None)[0]
            col_need_ad[field_name_i] = [i for i in sum(col_pair, []) if i not in dels]
        elif field_name_i.startswith('greatest_'):
            greatest_dict = {field_name_i: eval(formula_i)}
            if if_compute:
                data_recent[field_name_i] = now_greatest_fun(data_recent, greatest_dict, Info, col_add=None)[0]
            col_need_ad[field_name_i] = [i for i in eval(formula_i).keys() if i not in dels]
        elif field_name_i.startswith('casewhen_'):
            modify = lambda x: '(data_recent.' + x.replace('&', ') & (data_recent.').replace('|', ') | (data_recent.') + ')'
            rule = re.sub(pa, '', formula_i).strip('{|}').replace('，', ',').replace('：', ':').split(',')
            rule = [i.split(':') for i in rule]
            rule = [[i[0], strtotable_valuefun(Series(i[1])).iloc[0]] for i in rule]  # 正确转换取值为字符型、浮点型、整型
            rule_1 = [[modify(i[0]), i[1]] for i in rule if i[0] != 'else']
            if if_compute:
                for r, v in rule_1:
                    print(f"    {r} 赋值为: {v}")
                    data_recent.loc[eval(r), field_name_i] = v
                v_else = [i[1] for i in rule if i[0] == 'else']
                if v_else:
                    data_recent.loc[data_recent[field_name_i].isnull(), field_name_i] = v_else[0]
            need = set(sum([re.split('&|==|>=|<=|!=', re.sub('\(|\)| ', '',i[0])) for i in rule_1], []))
            col_need_ad[field_name_i] = [i.replace('data_recent.', '') for i in need if 'data_recent.' in i]
        else:
            s = f'手动衍生_py加工: {field_name_i}字段, 加工方式（{formula_i}）未实现，请扩充代码!'
            raise Exception(s)

    if if_compute:
        data_recent = data_recent.drop(columns=dels & set(data_recent.columns))
        na_inf_examine(data_recent[field_manual.field_name])

    # 公式中的字段可能非原始/手动衍生_sql字段，而是手动衍生_py字段，递归的获取该手动衍生_py字段使用的中间字段，直至所有中间字段、原始/手动衍生_sql字段被加入
    manuals = field_manual.field_name.values
    col_need_ad2 = copy.deepcopy(col_need_ad)
    for k, v in col_need_ad2.items():
        goon = [i for i in v if i in manuals]
        k_goon_list = []
        while goon:
            k_goon = goon[0]
            k_goon_list.append(k_goon)
            v_add = [i for i in col_need_ad[k_goon] if i not in col_need_ad2[k]]
            col_need_ad2[k].extend(v_add)
            goon = [i for i in col_need_ad2[k] if i in manuals and i not in k_goon_list]
    return data_recent, col_need_ad2
# ----------------------------------------------------------------------------------------------------------------------


def recent_stat_deal(data_recent, Info, equate='del', diff_limit=0.1):
    """
    统计近n月基础数据的各账期数据，提出异常警告，并根据需要决定是否进行处理
    :param data_recent: 近n月基础数据
    :param Info: 单个模型信息（命名元组）
    :param equate: 取值'del'：删除账期缺失的记录， 取值'fill':以nan填充缺失账期的记录; 其他：不做处理
    :param diff_limit: 某字段两个账期之间取值占比差值>=diff_limit,发出警告
                       若取值为None，则不进行比较
    :return: 处理后的data_recent，记录异常情况的字典
    """
    def idx_fun(idx):  # 兼容版本
        idx = idx.copy()
        idx = Series(idx)
        isna = idx.isnull()
        idx = Series([interval_to_str(i) for i in idx])
        idx[isna] = 'NP.NAN'
        return idx

    def col_compare(col, m1, m2):  # 比较字段col，在m1, m2两个账期的取值
        dt = data_recent.dtypes.astype(str)
        col1 = data_recent.loc[data_recent[Info.col_month] == m1, col]
        col2 = data_recent.loc[data_recent[Info.col_month] == m2, col]
        col1.name, col2.name = m1, m2
        if (dt[col] in ('float64', 'int64')) & (len(data_recent[col].unique()) > 20):
            cp = col1.quantile(q).drop_duplicates()
            cp.iloc[0], cp.iloc[-1] = cp.iloc[0] * 0.8, cp.iloc[-1] * 1.2
            col1 = pd.cut(col1, cp, include_lowest=True)
            col2 = pd.cut(col2, cp, include_lowest=True)
        c1 = col1.value_counts(dropna=False)
        c2 = col2.value_counts(dropna=False)
        p1, p2 = c1 / c1.sum(), c2 / c2.sum()
        p1.index, p2.index = idx_fun(p1.index), idx_fun(p2.index)  # 兼容版本
        p12 = pd.concat([p1, p2], join="outer", axis=1)
        p12 = p12.fillna(0)
        p12['differ'] = p12[m1] - p12[m2]
        return p12

    data_recent = data_recent.copy()
    col_month_na = data_recent[Info.col_month].isnull()
    m_max = data_recent[Info.col_month].max()
    if col_month_na.sum():
        s = f'账期字段{Info.col_month}存在缺失值{col_month_na.sum()}个, 此部分数据将被忽略，请及时修正！'
        warnings.warn(s); time.sleep(seconds)
        data_recent = data_recent.loc[~col_month_na, :]

    # 针对训练集、测试集， 对预测集无效
    if (~data_recent[Info.col_target].isnull()).sum() if (Info.col_target in data_recent.columns) else False:
        col_tagret_na = (data_recent[Info.col_month] == m_max) & (data_recent[Info.col_target].isnull())
        if col_tagret_na.sum():
            s = f'目标字段{Info.col_target}存在缺失值{col_tagret_na.sum()}个, 此部分数据将被忽略，请及时修正！'
            warnings.warn(s); time.sleep(seconds)
            data_recent = data_recent.loc[~col_tagret_na, :]

    id_more = data_recent[Info.col_id].groupby(data_recent[Info.col_month]).apply(lambda x: len(x.unique()) < len(x))
    id_more = id_more[id_more]
    if len(id_more) > 0:
        s = f'下列账期数据{Info.col_id}字段存在重复值，请检查：{id_more}'
        raise Exception(s)

    s = "data_recent.groupby([Info.col_month, 'data_use'])[Info.col_id].count()"
    month_vc = eval(s)
    month_amount = len(data_recent[Info.col_month].value_counts())
    print(f'各账期数据量分布：{add_printindent(month_vc)}')

    col_id_amount = data_recent.groupby(Info.col_id).apply(len)
    if col_id_amount.max() > month_amount:
        s = f'存在{(col_id_amount > month_amount).sum()}个{Info.col_id}的个数大于{month_amount}, 请检查数据'
        raise Exception(s)
    if len(month_vc.unique()) != 1:
        print(f'各账期数据量不一致')
        if equate == 'del':
            print(f'\n删除账期不足的记录，修改后：')
            col_id_valid = col_id_amount[col_id_amount == month_amount].index
            data_recent = data_recent.loc[data_recent[Info.col_id].isin(col_id_valid)]
            print(eval(s))
        elif equate == 'fill':
            print(f'以nan填充缺少账期的记录')
            data_now = data_recent.loc[data_recent[Info.col_month] == m_max].copy()
            data_now.index = data_now[Info.col_id].values
            month_vc_df = month_vc.reset_index()
            for d in month_vc_df.data_use:
                id_all = data_now.loc[(data_now.data_use == d) & (data_now[Info.col_month] == m_max), Info.col_id].copy()
                for m in set(month_vc_df[Info.col_month]) - {m_max}:
                    con_lack = ~(id_all.isin(data_recent.loc[data_recent[Info.col_month] == m, Info.col_id].unique()))
                    id_lack = id_all[con_lack]
                    fill_lack = DataFrame(index=id_lack, columns=data_recent.columns, dtype=float)
                    fill_lack[Info.col_id] = fill_lack.index
                    fill_lack[Info.col_month] = m
                    fill_lack['data_use'] = d
                    data_recent = pd.concat([data_recent, fill_lack], axis=0)
            print(f"    修改后：{add_printindent(eval(s))})")
        else:
            print(f'    不做修改')

    dis_exam = None
    if diff_limit is not None:
        print(f'\n考察近n月字段各账期取值分布')
        col_unique = data_recent.apply(lambda x: len(x.unique()))
        unique_one = list(col_unique[col_unique == 1].index)

        q = [0, 0.25, 0.5, 0.75, 1]
        months = sorted(data_recent[Info.col_month].unique(), reverse=True)
        dis_same = {}
        dis_diff = {}
        de = set(Info.col_mark + [Info.col_target] + unique_one)
        for col in data_recent.drop(columns=de & set(data_recent.columns)).columns:
            for m1, m2 in field_pair_fun(months, 2):
                compare = col_compare(col, m1, m2)
                diff_high = compare[compare.differ.abs() > diff_limit]
                if len(compare.differ.unique()) == 1:
                    dis_same[col] = compare
                    break
                elif len(diff_high) > 0:
                    dis_diff[col] = compare
                    break
        if unique_one:
            s = f'recent_stat_deal 所有账期取值唯一的字段：{unique_one}'
            warnings.warn(s); time.sleep(seconds)
        if dis_same:
            s = f'recent_stat_deal 不同账期分布相同的字段：{list(dis_same.keys())}'
            warnings.warn(s); time.sleep(seconds)
        if dis_diff:
            s = f'recent_stat_deal 不同账期分布差异大的字段：{list(dis_diff.keys())}'
            warnings.warn(s); time.sleep(seconds)
        dis_exam = {'unique_one': unique_one, 'dis_same': dis_same, 'dis_diff': dis_diff}
    else:
        print(f'\ndiff_limitweiNone,不考察近n月字段各账期取值分布, dis_exam返回None！')
    return data_recent, dis_exam
# ----------------------------------------------------------------------------------------------------------------------


def table_add_fun(pipeline, newdata, Info, table_already, iv_already, woe_already):
    """
    向已有宽表中添加字段的过程
    一、对新的数据集进行woe转换，剔除iv低的字段，剔除相关性系数强的字段
    二、新数据转换后，与已有数据对比，相关性强则剔除，否则加入（首份数据无从比较，直接全部保留）
    :param pipeline: 数据处理流水线
    :param newdata: DataFrame，待新增的数据
    :param Info: 单个模型信息（命名元组）
                 r_limit: 相关性系数阈值，剔除超过阈值的字段
                 iv_limit: IV阈值，剔除小于阈值的字段
    :param table_already: 已有的宽表, 首次使用为空数据DataFrame()
    :param iv_already: 已有宽表字段的iv，首次使用为空向量Series(dtype=np.float64)
    :param woe_already: 已有宽表字段的woe，首次使用为{}
    :return: 保存了必要结果的字典
    """
    if newdata is None:
        print('新加入数据newdata为None, 中止！')
        return None
    elif len(newdata) == 0:
        print('新加入数据newdata行数为0， 中止！')
        return None

    original = sys.stdout
    col_del_r = list()
    col_iv = None
    r_matrix_newdata = None
    r_matrix_newdata = None
    message_pipe = None
    pipeline = copy.deepcopy(pipeline)

    na_inf_examine(newdata)
    s = set(newdata.columns) & set(table_already.columns) - set(Info.col_mark + [Info.col_target])
    if s:
        raise Exception(f'参数newdata 与 参数table_already 的字段名称有重复，请检查：{s}')

    if False if table_already.shape[0] == 0 else table_already.shape[0] != newdata.shape[0]:
        raise Exception('参数table_already的长度与newdata的行数不相等，请检查!')

    if False if table_already.shape[0] == 0 else set(table_already.index) != set(newdata.index):
        raise Exception('参数table_already与newdata的index不同，请检查!')

    s = set(table_already.columns) - set(iv_already.index) - set(Info.col_mark + [Info.col_target])
    if s:
        raise Exception(f'参数iv_already的index应该包括参数table_already的字段名,缺少：{s}')

    newdata = newdata.copy()
    table_already = table_already.copy()
    iv_already = iv_already.copy()
    woe_already = woe_already.copy()
    table_already.index.name = None
    newdata.index.name = None

    print(f'新数据newdata的shape：{newdata.shape}')

    de = set(newdata.columns) & set(Info.col_mark + [Info.col_target])
    X = newdata.drop(columns=de)
    if X.shape[1] == 0:
        print(f"newdata.columns:{list(newdata.columns)}，删除{de}后剩余0列，中止！")
        return None
    else:
        print('数据转换')  # 完成woe编码，计算字段IV值，剔除了iv值过小的字段，剔除了相关性系数过大的字段
        message_pipe = privy_Autonomy()
        sys.stdout = message_pipe
        try:
            newdata_pipe = pipeline.fit_transform(X, newdata[Info.col_target])
        except Exception as er:
            raise Exception(er)
        finally:
            sys.stdout = original

    print(f'转换后shape：{newdata_pipe.shape}')

    if newdata_pipe.shape[1] == 0:
        print('中止！')
        return None

    #从流水线中获取WoeTransformer_DF
    woe = privy_get_trans(pipeline, 'WoeTransformer_DF')
    if len(woe.fit_in_colnames_) > 0:
        col_iv = woe.col_iv_
        r_matrix_newdata = woe.r_matrix_
        col_woe = woe.col_tabl_

    if table_already.shape == (0, 0):
        print('首个数据集，直接返回woe编码结果')
        # newdata_pipe = pd.concat([newdata[[Info.col_id]], newdata_pipe], axis=1)
        return {'col_in': newdata.columns,
                'table': newdata_pipe,
                'iv_accum': col_iv,
                 'woe_accum': col_woe,
                # 'newdata_pipe': newdata_pipe,
                # 'iv_newdata': col_iv,
                'r_matrix_newdata': r_matrix_newdata,
                'r_matrix_oldnew': None,
                'col_del_r': col_del_r,
                'col_del_table': [],
                'col_add_table': [],
                'pipeline': pipeline,
                'message_pipe': message_pipe
                }

    r_matrix = None
    col_add = []
    col_del = []
    r_matrix = newdata_pipe.apply(lambda x: table_already.corrwith(x))
    na_count = r_matrix.isnull().sum().sum()
    if na_count:
        raise Exception(f'相关性系数矩阵存在缺失值{na_count}个!')
    r_matrix = r_matrix.stack()
    r_matrix = r_matrix.reset_index()
    r_matrix = r_matrix.rename(columns={0: "r"})
    r_matrix['iv_0'] = r_matrix.level_0.map(dict(iv_already))
    r_matrix['iv_1'] = r_matrix.level_1.map(dict(col_iv))

    r_matrix_high = r_matrix.loc[abs(r_matrix.r) >= Info.r_limit, :].copy()
    r_matrix_high = r_matrix_high.loc[abs(r_matrix_high.r).sort_values(ascending=False).index]
    r_matrix_high['need_deal'] = r_matrix_high.level_0.notnull() & r_matrix_high.level_1.notnull()
    print(f'待新增字段与原宽表字段 相关性系数大于Info.r_limit（{Info.r_limit}）的字段对有{len(r_matrix_high)}对')

    if len(r_matrix_high) > 0:
        deal = r_matrix_high.loc[r_matrix_high.need_deal, 'need_deal']
        while len(deal) > 0:
            i = deal.idxmax()
            if r_matrix_high.loc[i, 'iv_0'] < r_matrix_high.loc[i, 'iv_1']:
                to_na = r_matrix_high.loc[i, 'level_0']
            else:
                to_na = r_matrix_high.loc[i, 'level_1']
            col_del_r.append(to_na)

            r_matrix_high.loc[r_matrix_high.level_0 == to_na, 'level_0'] = np.nan
            r_matrix_high.loc[r_matrix_high.level_1 == to_na, 'level_1'] = np.nan

            r_matrix_high['need_deal'] = r_matrix_high.level_0.notnull() & r_matrix_high.level_1.notnull()  # 更新
            deal = r_matrix_high.loc[r_matrix_high.need_deal, 'need_deal']
        print(f'    从中二者中剔除字段{len(col_del_r)}个字段:{col_del_r}')
        print(f'    其中待新增中剔除{len(set(newdata_pipe.columns) & set(col_del_r))}个')
        print(f'    其中原宽表中剔除{len(set(table_already.columns) & set(col_del_r))}个')
        print()
    else:
        print('    无需剔除')

    print(f'已有宽表shape：{table_already.shape}')
    col_add = list(set(newdata_pipe.columns) - set(col_del_r))
    col_del = list(set(table_already.columns) & set(col_del_r) - set(Info.col_mark))

    print(f'向宽表中加入字段（{len(col_add)}个）：{col_add}')
    table_already_new = pd.concat([table_already, newdata_pipe[col_add]], axis=1)

    if len(col_add) == 0:  # 不加入则不删除，有时新加入字段与原字段相关，并且原字段全被替换掉
        col_del = []
    print(f'从宽表中删除字段（{len(col_del)}个）：{col_del}')
    table_already_new = table_already_new.drop(columns=col_del)

    print(f'目前宽表shape：{table_already_new.shape}')

    if set(table_already_new.index) != set(table_already.index):
        raise Exception('新宽表与旧宽表的index不同， 请检查！')

    na_count = table_already_new.isnull().sum()
    na_count = na_count[na_count > 0]
    if len(na_count) > 0:
        raise Exception(f'出现缺失值：\n{na_count}')

    iv_accum = pd.concat([iv_already, col_iv])
    woe_accum = dict(woe_already, **col_woe)
    return {'col_in': newdata.columns,
            'table': table_already_new,      # 最新宽表：合并已有宽表与新增宽表
            'iv_accum': iv_accum,            # 记录了每次数据所有字段的iv
            'woe_accum': woe_accum,          # 记录了每次数据所有字段的woe
            # 'newdata_pipe': newdata_pipe,          # 新数据经过流水线处理后的结果
            # 'iv_newdata': col_iv,                  # 新数据的字段IV
            'r_matrix_newdata': r_matrix_newdata,    # 新数据字段之间的相关性
            'r_matrix_oldnew': r_matrix,     # 新数据筛选出的字段 与 已有宽表字段 之间的相关性
            'col_del_r': col_del_r,          # 新数据 与 已有宽表 之间剔除相关性高的字段
            'col_del_table': col_del,        # 从已有宽表中删除的字段（col_del_r 与 已有字段字段 的交集）
            'col_add_table': col_add,        # 新加入宽表中的字段
            'pipeline': pipeline,            # 数据处理流水线
            'message_pipe': message_pipe     # 过程输出
            }
# ----------------------------------------------------------------------------------------------------------------------


def tab_explore_create(table_in, Info, stage, src, if_condition=False):
    """
    探索建模宽表 或 按探索的结果加工宽表
    :param table_in: 基础字段（union all近n月基础数据）
    :param Info: 单个模型信息（命名元组）
                 auto_pair2: 是否自动进行字段两两衍生（任意两个字段之间的和差积商、字符拼接等）
                 diff_limit: recent_stat_deal函数参数，某字段两个账期之间取值占比差值>=diff_limit,发出警告（若取值为None，则不进行比较）
                 tab_psi: 取值True时，计算结果宽表字段稳定度，False时不计算
    :param stage: 取值为explore时，探索宽表，并保存相关结果
                  取值为create时，加载探索结果，以此为依据加工宽表
    :param src: read_data函数的src，规定从数据库还是文件中读取输入数据
    :param if_condition: 是否按照Info.condition筛选用户，因为加工近n月基础数据时，已经限定了用户，此函数中可以忽略nfo.condition，即if_condition=False
    :return: 宽表、添加本过程信息的Info，并会保存必要的中间结果
    备注：1.近n月基础数据已经筛选出了预期用户群，故在宽表探索和宽表加工环节，不用再用condition限制用户群了
            且condition只限定观察期最后一个月的用户范围，若在本函数中再次限定，则会将观察期前几个月的用户错误剔除
            如果一定要在本函数中读取数据时限定用户范围，需要将条件修改为只限定观察期最后一个月用户范围
          2.将剔除目标表的所有字段，并添加本模型的目标字段，所有无需担心其他模型的目标字段入模

    """
    with warnings.catch_warnings(record=True) as w:
        start_time = datetime.datetime.now()
        print(f"开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # --------------------------------- <editor-fold desc="打印参数"> ----------------------------------------------
        if str(table_in) == 'nan':
            s = 'table_in为nan，结束！'
            raise Exception(s)

        # 获取步骤并打印
        step = re.findall('train|test|predict', re.sub('^.*/|^.*\\\\', '', table_in))
        if len(step) == 1:
            step = step[0]
        else:
            raise Exception('未在table_in中识别出train、test或predict')

        mark = month_mark(Info, step)
        wd = Info.model_wd_predict if step == 'predict' else Info.model_wd_traintest

        iv_limit = Info.iv_limit
        paras = {'table_in': table_in,
                 'stage': stage,
                 'Info.model_name': Info.model_name,
                 'Info.r_limit': Info.r_limit,
                 'Info.iv_limit': Info.iv_limit,
                 'Info.auto_pair2': Info.auto_pair2,
                 'stage': stage,
                 'step': step
                 }
        paras_p = '\n    '.join({str(k) + ': ' + str(v) for k, v in paras.items()})
        print(f"参数设置：\n    {paras_p}\n")
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # ---------------------------------- <editor-fold desc="内部函数"> ---------------------------------------------
        def pipe_fun():
            """
            创建宽表探索的数据处理流水线，可更具个人需要灵活需改此处
            :return:
            """
            num_pipeline_df = Pipeline_DF([  # 数值型字段处理
                ('select_num', NumStrSpliter_DF(select='num', trans_na_error=False)),
                ('imputer_num', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value=0)),
                ('prefilter_num', FeaturePrefilter_DF(freq_limit=1)),
                ('Mdlp_dt_DF', Mdlp_dt_DF())])

            notnum_pipeline_df = Pipeline_DF([  # 类别型字段处理
                ('select_notnum', NumStrSpliter_DF(select='notnum', trans_na_error=False)),
                ('imputer_notnum', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value='unknown')),
                ('prefilter_notnum', FeaturePrefilter_DF(freq_limit=1, unique_limit=5000, valuecount_limit=50)),
                ('encoder', CategoricalEncoder_DF(encoding='onehot-dense', valueiv_limit=Info.iv_limit, Pcase=Info.Pcase, Ncase=Info.Ncase, toobject_xj=True))
            ])

            union = FeatureUnion_DF([  # 合并数值、类别型字段
                ('num_pipe', num_pipeline_df),
                ('notnum_pipe', notnum_pipeline_df)])

            pipeline = Pipeline_DF([
                ('union', union),
                ('woe', WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=True, iv_limit=Info.iv_limit, r_limit=Info.r_limit, warn_mark='tab_explore'))])
            return pipeline

        def stage_fun(Info, res, comment_auto, result, stage, pipe=None):
            """
            探索/加工宽表的一步步叠加
            :param res: 探索宽表时为table_add_fun函数的部分输入参数字典，加工宽表时为已有宽表
            :param comment_auto: 仅针对探索环节，记录宽表数据字典
            :param result: （待新增数据，数据字典），加工宽表时为数据字典为None即可
            :param stage: 探索（explore） 或加工（create）
            :param pipe: 数据处理流水线，仅针对探索环节
            :return: 迭代后的新的res, comment_auto
            """
            pipe = copy.deepcopy(pipe)
            comment_new = comment_auto
            if stage == 'explore':
                if pipe is None:
                    raise Exception('stage取值explor时，pipe必须设置')
                res_new = table_add_fun(pipe, result[0], Info, res['table'], res['iv_accum'], res['woe_accum'])
                if res_new is None:  # table_add_fun的newdata为空或newdata数据转换后为空，无从添加宽表字段，将返回None
                    res_new = res  # 没有添加字段，保持原res
                if result[1] is not None:
                    comment_new = pd.concat([comment_auto, result[1]], axis=0)
            elif stage == 'create':
                res_new = pd.concat([res, result[0]], axis=1)
                print(res_new.shape)
            return res_new, comment_new

        def table_r_fun(table):
            """
            计算字段之间的相关性系数
            :param table: Dataframe
            :return: Series, 字段对由层次化索引表示
            """
            r_m = table.corr()
            lenght = len(r_m)
            c = DataFrame([range(lenght)] * lenght)
            r = c.T
            r_m[(r >= c).values] = np.nan
            r_m = r_m.stack()
            return r_m
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # ----------------------------- <editor-fold desc="探索/创建宽表相关参数"> -------------------------------------
        Info = choradd_namedtuple(Info, {'Pcase': str(Info.Pcase), 'Ncase': str(Info.Ncase)})
        comment_auto = DataFrame()
        # 分stage操作
        r_iv_limit = Series({'r_limit': Info.r_limit, 'iv_limit': Info.iv_limit})

        # 当读取时无法使用condition筛选时，可能在读取后在python筛选，将使用条件字段
        # 预测打分时将用到dict_sortscore、col_out来排序和输出
        col_other = (set(get_condition_col(Info.condition)) |
                     set(Info.dict_sortscore.keys() if Info.dict_sortscore else []) |
                     set(Info.col_out if Info.col_out else []))

        table_info = string_to_table(s_table_info)
        tableY = table_info.loc[(table_info.s_field_base == Info.s_field_base) & (table_info.tabletype == 'tableY'), "tablename"].iloc[0]
        tableXscore = table_info.loc[(table_info.s_field_base == Info.s_field_base) & (table_info.tabletype == 'tableXscore'), "tablename"].iloc[0]
        if stage == 'explore':
            limit_err = r_iv_limit[r_iv_limit.isnull()]
            if len(limit_err):
                raise Exception(f"stage取值为explore时，{', '.join(limit_err.index)}必须设置!")

            field_base = copy.deepcopy(Info.field_base)
            # field_base = field_base.loc[field_base.field_src != "其他"].copy()  # "其他"字段只是为了记录字段相关信息，不会出现再宽表字段中
            print(f'field_base: {len(field_base)}行\n')

            col_notavail = {}
            if str(Info.available_add) != 'nan':
                # field_base_fun()已经删除了所有模型“不可用”字段（available列）
                # 此处分模型删除各自不需要用到的字段，删除的“不可用”字段将不会出现在宽表探索范围内
                colvalue_exam(field_base, Info.available_add, ['不可用', np.nan])

                field1 = field_base.field_name
                # 从field_base中剔除“不可用”字段及基于“不可用”字段加工的字段
                field_base = field_base_delinvalid(field_base, Info.available_add)
                col_notavail = set(field1) - set(field_base.field_name)

            field_init = field_base.loc[
                field_base.field_src.isin(['原始', '手动衍生_sql']) &
                ((field_base.table != tableY) | (field_base.field_name == Info.col_target)) &
                (field_base.field_name != 'rn')].copy()
            field_init.loc[field_init.field_name.isin([Info.col_month, Info.col_target]), 'dtype_classify'] = '类别型'
            Info = choradd_namedtuple(Info, {'Pcase': str(Info.Pcase), 'Ncase': str(Info.Ncase)})

            col_char = dropdup(list(field_init.loc[field_init.dtype_classify == '类别型', 'field_name']) + [Info.col_target])
            col_num = list(field_init.loc[field_init.dtype_classify == '数值型', 'field_name'])
            col_date = list(field_init.loc[field_init.dtype_classify == '日期型', 'field_name'])
            condition = Info.condition if if_condition else None
            if if_condition:
                raise Exception('需要将条件修改为只限定观察期最后一个月用户范围')
            else:
                condition = None
            condition = (f"({condition}) and " if condition else '') + "data_use='data_train'"  # 探索宽表时应删除时间内验证集
            col_need = dropdup(Info.col_mark + col_num + col_date + col_char)

            # 需要提前加工的字段
            field_manual = field_base.loc[(field_base.field_src == '手动衍生_py')].copy()

            # 创建数据处理流水线
            pipeline = pipe_fun()

            res = {'col_in': [], 'table': DataFrame(), 'iv_accum': Series(dtype=np.float64), 'woe_accum': {}}
            comment = dict(field_base[~field_base.comment.isnull()].set_index('field_name').comment)
            col_add = [Info.col_target]

        elif stage == 'create':
            limit_warn = r_iv_limit[~r_iv_limit.isnull()]
            if len(limit_warn):
                s = f"stage取值为create时,{', '.join(limit_warn.index)}参数无效，将被忽略"
                warnings.warn(s); time.sleep(seconds)

            sub_target = {Info.col_target} if step == 'predict' else set()   # 用于在预测时删除目标字段

            if step != 'predict':  # 训练测试时
                comment_valid = Info.comment_all.loc[(Info.comment_all.是否宽表字段 == '是') &
                                                     (~Info.comment_all.field_name.isin(sub_target))]
                col_all = comment_valid.field_name  # 宽表所有字段名
            else:  # 预测时只关注入模字段，减少预测数据大小
                field_comment = Info.comment_all.loc[Info.comment_all.是否宽表字段 == '是']
                comment_valid = field_comment.loc[
                    ((field_comment.是否入模 == '是') & (field_comment.field_name != Info.col_target)) |
                    field_comment.field_name.isin(Info.col_mark + list(col_other))]
                col_all = comment_valid.field_name

            col_need, field_manual, col_num, col_char, col_date = get_col(Info, comment_valid)

            # 自动衍生_py字段的字段前缀、计算字段
            col_df = DataFrame(col_all[col_all.str.contains('__')].str.split('__').map(lambda x: [x[0], x[1:] if len(x) > 2 else x[-1]]).to_list(), columns=['suf', 'col'])

            field_init = None
            condition = Info.condition if if_condition else None
            res = DataFrame()
            col_add = None
            comment = None
            pipeline = None
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # ------------------------------ <editor-fold desc="近n月基础数据准备"> ----------------------------------------
        # 读取数据
        data_recent = read_data(table_in, src, condition, col_need, col_char=col_char, col_num=col_num, col_date=col_date)

        # 分数字段的缺失值赋值为0，否则在下文加工 手动衍生_py 字段时，会返回缺失值将报错
        col_score = Info.field_base[Info.field_base.table == tableXscore].field_name
        col_score = [i for i in col_score if i in data_recent.columns]
        data_recent[col_score] = data_recent[col_score].fillna(0)

        # 检验数据账期是否完备
        m_actual = sorted(data_recent[Info.col_month].astype(str).unique())
        if 'user_acct_month' in data_recent.columns:
            m_expect = sorted(set(sum([month_list_fun(str(i), periods=-Info.n_recent) for i in data_recent.user_acct_month.unique()],[])))
        else:
            m_expect = month_list_fun(str(Info._asdict()[f"month_{step}"]), periods=-Info.n_recent)
        if m_actual != m_expect:
            s = f"data_recent应有下列账期数据：{m_expect}， 但实际：{m_actual}"
            raise Exception(s)

        m_max = data_recent[Info.col_month].max()
        if 'user_acct_month' in data_recent.columns:
            data_recent.user_acct_month = data_recent.user_acct_month.astype(str)
            user_acct_month_u = data_recent.user_acct_month.unique()
            # 累计多个账期的正例样本时：统一不同user_acct_month取值的数据，逻辑详见文末的统一示例
            if len(user_acct_month_u) > 1:
                print('统一user_acct_month --------------------------------------')
                print(f'user_acct_month: {user_acct_month_u}')
                print(f'修改 {Info.col_id} (user_acct_month _ {Info.col_id})')
                data_recent[Info.col_id] = data_recent.user_acct_month.astype(str) + '_' + data_recent[Info.col_id].astype(str)
                print(f'修改 {Info.col_month}')
                for i in sorted(set(user_acct_month_u) - {m_max}):
                    print(f'    修改 user_acct_month - {i} 的{Info.col_month}')
                    m_diff = round((datetime.datetime.strptime(m_max, '%Y%m') - datetime.datetime.strptime(i, '%Y%m')).days/30)
                    con = data_recent.user_acct_month == i
                    print(f'    修改量：{con.sum()}')
                    data_recent.loc[con, Info.col_month] = data_recent.loc[con, Info.col_month].apply(lambda x: month_add(x, m_diff))
                print('统一user_acct_month 完毕 ---------------------------------\n')

        # 提前加工：手动衍生_py 字段
        data_recent, col_need_ad = manual_fun(data_recent, Info, field_manual)
        print(f'数据量: {data_recent.shape}\n')

        print('检查数据')
        data_recent, dis_exam = recent_stat_deal(data_recent, Info, equate='fill', diff_limit=Info.diff_limit)

        # 更新
        col_type = data_recent.dtypes.astype(str).str.replace('64.*?$|32.*?$|16.*?$|8.*?$', '')
        col_num = list(col_type[col_type.isin(['float', 'int', 'uint'])].index)
        col_char = list(col_type[col_type.isin(['object'])].index)
        col_date = list(col_type[col_type.isin(['datetime'])].index)

        if stage == 'explore':
            print('\n', end='')
            print(f'字段类型分布: {add_printindent(data_recent.dtypes.value_counts())}')
            col_lack = set(field_init.field_name) - set(col_type.index)
            col_more = set(col_type.index) - set(field_base.field_name) - {Info.col_target}
            type_lack = set(col_type.index) - set(field_base.field_name[field_base.dtype_classify.notnull()]) - {Info.col_target}
            if col_lack:
                s = f'field_base中包括下列字段，但实际数据 缺少：{col_lack}'
                warnings.warn(s); time.sleep(seconds)
            if col_more:
                s = f'field_base中未包括下列字段，但实际数据 多出：{col_more}'
                warnings.warn(s); time.sleep(seconds)
            if type_lack:
                s = f'field_base中未规定下列字段数据类型，请确认默认类型是否正确：{dict(col_type[type_lack])}'
                warnings.warn(s); time.sleep(seconds)

            col_rest = col_type[~col_type.isin(['float', 'int', 'uint', 'object', 'datetime'])].index
            if len(col_rest) > 0:
                warnings.warn(f'下列字段未归类，将被忽略，请及时更新程序：{dict(col_type[col_rest])}'); time.sleep(seconds)

            col_now = [i for i in col_char + col_num if i not in Info.col_mark]

        print('\n')
        print('填充数值型&类别型字段缺失值')  # 为后续加工衍生字段，故不依靠流水线中的处理缺失值环节
        data_recent[col_num] = data_recent[col_num].fillna(0)
        data_recent[col_char] = data_recent[col_char].fillna('unknown')
        # 后续可能基于日期计算时长，所以不填充日期型的缺失值
        print(f'日期型字段缺失值情况：{add_printindent(data_recent[col_date].isnull().sum())}')

        # 筛选当月原始数据
        m_expect = Info._asdict()[f'month_{step}']
        m_expect_value = type(m_max)(m_expect)  # 兼顾 账期字段：数据库中数值型，python中规定类别型
        if m_max != m_expect_value:
            raise Exception(f"{m_expect}={m_expect_value},但数据的最大账期={m_max}")
        data_now = data_recent.loc[data_recent[Info.col_month] == m_max].copy()
        data_now.index = data_now[Info.col_id].values
        if step != 'predict':
            print('\n', end='')
            dis_target = data_now.groupby(['data_use', Info.col_month, Info.col_target])[Info.col_id].count()
            print(f'正负例分布：{add_printindent(dis_target)}')
            yyy = dis_target.groupby(Info.col_target).count().to_list()
            if (len(yyy) != 2) or (len(set(yyy)) != 1):
                raise Exception(f'{Info.col_target}字段取值分布有误，请检查！')
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # 当月数据 两个字段之间的衍生
        now_methods_desc = {'add': '之和', 'sub': '之差', 'mult': '之积', 'div': '之比'}

        # 近n月数据 字段的近n月均值等
        rencent_n = f'近{len(data_recent[Info.col_month].unique())}月'
        recent_methods_desc = {'avg': '均值', 'sep': '离散系数', 'wave': '波动性', 'grow': '成长率',
                               'max': '最大值', 'min': '最小值', 'std': '标准差'}
        recent_methods_desc = {k: rencent_n + v for k, v in recent_methods_desc.items()}

        if stage == 'create':
            print('\n----------------------------------------- 当月 类别型 onthot ')
            col_onthot = col_all[~col_all.str.contains('valuecnt__') & col_all.str.contains('~')]
            if len(col_onthot):
                print(list(col_onthot))
                for col_value in col_onthot:
                    col, value = col_value.split('~')
                    data_now[col_value] = (data_now[col] == value).astype(int)
            col_now = set(data_now.columns) & set(col_all) - sub_target

        print('\n----------------------------------------- 当月 数值型&类别型 原始数据 ')
        res, comment_auto = stage_fun(Info, res, comment_auto, (data_now[col_now], None), stage, pipeline)
        if stage == 'explore':
            iv_init = res['iv_accum']

        # --------------------------------- <editor-fold desc="筛选衍生字段的原始/手动衍生_sql字段"> -------------------
        print('筛选衍生字段的 原始/手动衍生_sql 字段')
        if stage == 'explore':
            # 不参与衍生的字段
            invalid = "field_base.remark.fillna('').str.lower().fillna('').str.contains('不参与自动衍生|%s')"
            now_invalid = field_base.loc[eval(invalid % '不参与当月自动衍生'), 'field_name'].values
            recent_invalid = field_base.loc[eval(invalid % '不参与近n月自动衍生'), 'field_name'].values

            num_valid = set(res['table'].columns) & set(col_num) - set([Info.col_id, Info.col_target])
            num_valid = Series(list(num_valid))  # 为后续方便拼接字段名
            num_now = [i for i in num_valid if i not in now_invalid]
            num_recent = [i for i in num_valid if i not in recent_invalid]
            print(f"数值型字段：有效字段{len(num_valid)}个，其中当月自动衍生字段{len(num_now)}个，近n月自动衍生字段{len(num_recent)}个")
            if '__' in str(num_valid):
                w = Series(num_valid)[Series(num_valid).str.contains('__')].values
                s = f"数值型有效字段的字段名中包括__， 注意后续生成宽表时可能会发生字段名拆分错误:{w}"
                warnings.warn(s); time.sleep(seconds)

            pa = '^' + '~|^'.join(col_char) + '~'
            char_validvalue = res['table'].columns[res['table'].columns.str.contains(pa)]

            # 添加人为强制加入的自动衍生_py字段
            con_valuecnt = (field_base.field_src == '自动衍生_py') & (field_base.field_name.str.startswith('valuecnt__'))
            field_auto = field_base.loc[con_valuecnt, 'field_name']
            field_auto_value = set()
            for auto in field_auto:
                c_v = auto.replace('valuecnt__', '')
                c, v = c_v.split('~')
                if (data_now[c].astype(str) == v).sum() == 0:
                    u = list(data_now[c].unique())
                    v_maybe = Series(u).loc[Series(u).apply(lambda x: len(set(str(x)) & set(str(v)))).idxmax()]
                    ad = c + '~' + str(v_maybe)
                    v_ad = 'valuecnt__' + ad
                    s = f"请确认：{c}字段不包括取值{v}, 其取值范围为：{u}，删除{auto}, 添加{v_ad}"
                    warnings.warn(s); time.sleep(seconds)
                    field_auto_value = field_auto_value | {ad}
                    field_base.loc[field_base.field_name == auto, 'field_name'] = v_ad
                else:
                    field_auto_value = field_auto_value | {c_v}

            col_must = field_base.loc[field_base.must_remain.fillna('').str.contains('是')].field_name

            count_auto = len(set(field_auto_value) - set(char_validvalue))
            char_validvalue = char_validvalue.union(field_auto_value)
            char_now = [i for i in char_validvalue if re.sub('~.*', '', i) not in now_invalid]
            char_recent = [i for i in char_validvalue if re.sub('~.*', '', i) not in recent_invalid]
            print(f"类别型字段：有效字段{len(char_validvalue)}个（强制加入{count_auto}个）:")
            print(f"    当月自动衍生_py字段{len(char_now)}个: {char_now}")
            print(f"    近n月自动衍生_py字段{len(char_recent)}个: {char_recent}")
            if '__' in str(char_validvalue):
                w = Series(char_validvalue)[Series(char_validvalue).str.contains('__')].values
                s = f"类别型有效字段的字段名中包括__， 注意后续生成宽表时可能会发生字段名拆分错误:{w}"
                warnings.warn(s); time.sleep(seconds)

            col_now_paste = field_base[field_base.field_name.str.contains('paste__') &
                                       (field_base.field_src == '衍生')].field_name
            pair2_dict = {k: field_pair_fun(num_now, 2) for k in now_methods_desc.keys() if k != 'paste'}
            pair2_dict['paste'] = [tuple(i[1:]) for i in col_now_paste.str.split('__')]
            na_to_value = {}
            num_recent_dict = {k: num_recent for k in recent_methods_desc}

            morecnt_thre_dict = dict()
            morecnt_com = rencent_n + '大于%s的月份数'
            morecnt_thre_dict[('more0cnt', morecnt_com % 0)] = Series([0] * len(num_recent), index=num_recent)
            morecnt_thre_dict[('moreq25cnt', morecnt_com % '第一四分位数')] = data_now[num_recent].quantile(q=0.25)

            col_valuecnt = char_recent

        elif stage == 'create':
            pair2_dict = {k: col_df[col_df.suf == k].col if k in col_df.suf.values else [] for k in now_methods_desc.keys()}
            na_to_value = Info.na_to_value

            num_recent_dict = {k: col_df[col_df.suf == k].col if k in col_df.suf.values else [] for k in recent_methods_desc}

            col_morecnt = col_df.loc[col_df.suf.str.contains('more.*?cnt')]
            morecnt_thre_dict = {k: v[col_morecnt.loc[col_morecnt.suf == k[0], 'col']]
                                 for k, v in Info.morecnt_thre_dict.items()
                                 if k[0] in col_morecnt.suf.values}

            col_valuecnt = col_df[col_df.suf == 'valuecnt'].col
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # 当月 两个字段之间 衍生
        if Info.auto_pair2:
            for method_desc in now_methods_desc.items():
                method = method_desc[0]
                if len(pair2_dict[method]) > 0:
                    print(f'\n----------------------------------------- 当月 两个字段: {method_desc[1]} ')
                    print(f'字段对个数：{len(pair2_dict[method])}, 添加字段：{col_add}')
                    res_pair2 = opera_pair_fun(data_now, pair2_dict[method], method_desc, col_add, comment)
                    res, comment_auto = stage_fun(Info, res, comment_auto, res_pair2, stage, pipeline)

        # 近三月均值等
        for method_desc in recent_methods_desc.items():
            method = method_desc[0]
            if len(num_recent_dict[method]) > 0:
                print(f'\n----------------------------------------- 近n月 数值型 {method_desc[1]} ')
                print(f'字段个数：{len(num_recent_dict[method])}, 添加字段：{col_add}')
                res_recent = recent_num_fun(data_recent, num_recent_dict[method], Info, method_desc, col_add, comment)
                res, comment_auto = stage_fun(Info, res, comment_auto, res_recent, stage, pipeline)

        if len(morecnt_thre_dict) > 0:
            print('\n----------------------------------------- 近n月 数值型 大于x的月份数 ')
            res_morecnt = recent_morecnt_fun(data_recent, Info, morecnt_thre_dict, col_add, comment)
            res, comment_auto = stage_fun(Info, res, comment_auto,  res_morecnt, stage, pipeline)

        if len(col_valuecnt) > 0:
            print('\n----------------------------------------- 近n月 类别型 取某值的月份数 ')
            print(f'字段个数：{len(col_valuecnt)}, 添加字段：{col_add}')
            res_valuecnt = recent_valuecnt_fun(data_recent, col_valuecnt, Info, col_add, comment)
            res, comment_auto = stage_fun(Info, res, comment_auto, res_valuecnt, stage, pipeline)

        # ------------------------ <editor-fold desc="汇总最终宽表、处理、保存"> ---------------------------------------
        print('\n----------------------------------------- 汇总最终宽表 ')
        if stage == 'explore':
            table = res['table'].copy()
            na_inf_examine(table)

            print('计算字段之间的相关性系数')
            table_r = table_r_fun(table)
            print(f"{add_printindent(table_r.abs().describe().round(3))}\n")

            iv_X = res['iv_accum'][set(table.columns) - set([Info.col_id])]
            print(f'iv分布：{add_printindent(iv_X.describe().round(3))}\n')

            # woe 整理
            woe_accum_df = DataFrame()
            if res['woe_accum']:
                for v in res['woe_accum'].values():
                    v = v.copy()
                    v.columns.name = None
                    k = v.index.name
                    v.index.name = 'value'
                    v = v.reset_index()
                    v.index = [k] * len(v)
                    woe_accum_df = pd.concat([woe_accum_df, v])
                woe_accum_df['ALL'] = woe_accum_df[Info.Pcase] + woe_accum_df[Info.Ncase]
                woe_accum_df = woe_accum_df.sort_values(by='woe', ascending=False)
                print(f'woe_accum(累计)概览：{add_printindent(woe_accum_df[woe_accum_df.ALL > 500].head())}\n')

            print('最终宽表字段列表col_all:')
            col_all = list(table.columns)
            print(f'    len(col_all): {len(col_all)}')

            # 非宽表探索所得的特征，基础数据字典field_base中强制保留在宽表中的字段；特征字段
            supply_must = set(col_must) - set(col_all)
            if supply_must:
                print(f"    补充{len(supply_must)}个的字段（must_remain=是）：{supply_must}")
                col_all = col_all + list(supply_must)
                print(f'    len(col_all): {len(col_all)}')

            # 非宽表探索所得的特征，账期、数据集名称、用户标识等，目标字段；非特征字段
            print(f"    补充账期、数据集名称、用户标识等，目标字段(Info.col_mark、Info.col_target)")
            col_all = Info.col_mark + col_all + [Info.col_target]
            print(f'    len(col_all): {len(col_all)}\n')

            # 非宽表探索所得的特征，在此之外的条件字段、预测分数排序字段、预测分数输出字段；非特征字段
            supply_other = set(col_other) - set(col_all)
            if supply_other:
                print(f"    补充{len(supply_other)}个的字段（col_mark、col_target、condition、dict_sortscore、col_out）：{supply_other}")
                col_all = col_all + list(supply_other)
                print(f'    len(col_all): {len(col_all)}')

            print('整理宽表数据字典')
            # 此处 类别字段~取值 字段修改为自动衍生_py
            char_validvalue_df = DataFrame(char_validvalue, columns=['field_name'])
            char_validvalue_df['col'] = char_validvalue_df.field_name.str.replace('~.*?$', '')
            char_validvalue_df['value'] = char_validvalue_df.field_name.str.replace('^.*?~', '')
            char_validvalue_df['com'] = char_validvalue_df.col.map(dict(field_base.set_index('field_name').comment))
            char_validvalue_df['comment'] = char_validvalue_df.com + '：取值是否为' +char_validvalue_df.value
            char_validvalue_df['field_src'] = '自动衍生_py'
            char_validvalue_df['is_cause'] = char_validvalue_df.col.map(dict(field_base.set_index('field_name').is_cause))

            char_validvalue_df = char_validvalue_df[char_validvalue_df.columns.intersection(field_base.columns)]

            comment_auto['field_src'] = '自动衍生_py'
            comment_auto['field_name'] = comment_auto.index
            comment_auto['is_cause'] = comment_auto.field_name.str.replace('^.*?__|~.*?$', '').map(dict(field_base.set_index('field_name').is_cause))
            comment_all = pd.concat([Info.field_base[Info.field_base.field_src != '自动衍生_py'], char_validvalue_df, comment_auto, field_base[field_base.field_src == '自动衍生_py']])
            comment_all = comment_all[(comment_all.field_src != '其他') | (comment_all.field_name.isin([Info.col_month]))]   # "其他"字段仅为了记录字段信息，而与宽表无关
            comment_all = comment_all.loc[~comment_all.field_name.duplicated()]

            # onehot衍生字段
            if_onehot = comment_all.field_name.str.contains('~') & (~comment_all.field_name.str.contains('valuecnt'))
            onehot1 = comment_all.comment.isnull() & if_onehot
            onehot2 = comment_all.dtype_classify.isnull() & if_onehot
            comment_all.loc[onehot1, 'comment'] = comment_all.loc[onehot1, 'field_name'].str.split('~').map(lambda x: field_base.set_index('field_name').comment[x[0]] + '-是否为：' + x[1])
            comment_all.loc[onehot2, 'dtype_classify'] = '数值型'  # 类别型

            # 补齐field_base之外字段, 注释, 类型
            col_beyondfieldbase = ['user_acct_month', 'data_use', Info.col_month, Info.col_target]
            comment_all = pd.concat([DataFrame({'field_name': [i for i in col_beyondfieldbase if i not in comment_all.field_name.values]}), comment_all])
            comment_beyond = {'user_acct_month': '观察期最后账期', 'data_use': '数据集名称', Info.col_month: '账期', Info.col_target: '预测目标'}
            comment_all.loc[comment_all.field_name.isin(col_beyondfieldbase), 'comment'] = comment_all.loc[comment_all.field_name.isin(col_beyondfieldbase), 'field_name'].map(comment_beyond)
            comment_all.loc[comment_all.field_name.isin(col_beyondfieldbase), 'dtype_classify'] = '类别型'
            comment_all.loc[comment_all.field_name.isin(col_beyondfieldbase), 'field_src'] = '手动衍生_sql'

            # 确定每个字段所需的基础字段(原始/手动衍生_sql)
            con_self = (comment_all.field_src.isin(['原始', '手动衍生_sql'])) | comment_all.field_name.isin(col_beyondfieldbase)
            comment_all.loc[con_self, 'base_init'] = comment_all.loc[con_self, 'field_name']
            comment_all.loc[comment_all.field_src == '手动衍生_py', 'base_init'] = comment_all.loc[comment_all.field_src == '手动衍生_py', 'field_name'].map(col_need_ad)
            # 自动衍生_py字段 <- 原始/手动衍生_sql|(手动衍生_py字段 <- 原始/手动衍生_sql)：若基于手动衍生_py字段，在加工宽表环节应先加工手动衍生_py字段
            auto = comment_all.loc[comment_all.field_src == '自动衍生_py', ['field_name', 'base_init']].copy()
            auto['base0'] = auto.field_name.str.replace('^.*?__|~.*?$', '')
            auto['base0_field_src'] = auto['base0'].map(comment_all.set_index('field_name').field_src)
            auto.loc[auto.base0_field_src.isin(['原始', '手动衍生_sql']), 'base_init'] = auto.loc[auto.base0_field_src.isin(['原始', '手动衍生_sql']), 'base0']
            auto.loc[auto.base0_field_src == '手动衍生_py', 'base_init'] = auto.loc[auto.base0_field_src == '手动衍生_py', 'base0'].map({k: [k] + v for k, v in col_need_ad.items()})
            comment_all.loc[comment_all.field_src == '自动衍生_py', 'base_init'] = comment_all.loc[comment_all.field_src == '自动衍生_py', 'field_name'].map(auto.set_index('field_name')['base_init'])
            # 确定有遗漏字段
            base_na = comment_all['base_init'].isnull() & (~comment_all.field_name.isin(col_notavail))
            if base_na.sum():
                s = f'{base_na.sum()}个字段所需的原始/手动衍生_sql字段base_init缺失，会影响后续宽表加工，请检查：\n{comment_all.loc[base_na]}'
                raise Exception(s)

            # 将目标字段移至最后，方便查看
            comment_all = pd.concat([comment_all.loc[comment_all.field_name != Info.col_target],
                                     comment_all.loc[comment_all.field_name == Info.col_target]])

            comm_lack = comment_all.comment.isnull() | comment_all.dtype_classify.isnull() | comment_all.field_src.isnull()
            if comm_lack.sum():
                s = f'结果宽表的数据字典: {comm_lack.sum()}个字段的dtype_classify、comment或field_src缺失，会影响后续宽表加工，请检查：\n{comment_all[comm_lack]}'
                warnings.warn(s); time.sleep(seconds)

            lack_valid = set(col_all) - set(comment_all.field_name)
            if lack_valid:
                s = f'宽表数据字典中缺少{len(lack_valid)}个字段，会影响后续宽表加工，请检查'
                warnings.warn(s); time.sleep(seconds)

            comment_all.index = [''] * len(comment_all)
            comment_all['是否宽表字段'] = '否'
            comment_all.loc[comment_all.field_name.isin(col_all), '是否宽表字段'] = '是'
            comment_all = comment_all[['是否宽表字段'] + [i for i in comment_all.columns if i != '是否宽表字段']]
            comment_valid = comment_all[comment_all.是否宽表字段 == '是']
            print(f'宽表数据字典概览 {comment_valid.shape}:\n{add_printindent(comment_valid.head())}\n')

            tabexp_col_obj = {
                'supply_must': supply_must,
                'supply_other': supply_other,
                'comment_all': comment_all,
                'iv_valid': iv_X,
                'iv_init': iv_init,
                'iv_all': res['iv_accum'],
                'woe_accum_df': woe_accum_df,
                'na_to_value': na_to_value,
                'morecnt_thre_dict': morecnt_thre_dict,
                'now_methods_desc': now_methods_desc,
                'recent_methods_desc': recent_methods_desc,
                'dis_exam': dis_exam,
                'num_valid': num_valid,
                'char_validvalue': char_validvalue}

            Info = choradd_namedtuple(Info, tabexp_col_obj)
            file_info = Info.model_wd_traintest + '/Info~tabexp.pkl'
            print(f'保存Info至：{file_info}')
            privy_fileattr_fun(file_info, 'unhide')
            joblib.dump(Info._asdict(), file_info)
            privy_fileattr_fun(file_info)

        elif stage == 'create':
            table = res
            col_lack = set(col_all) - set(table.columns)
            col_more = set(table.columns) - set(col_all)
            if col_lack:
                raise Exception(f'缺少{len(col_lack)}个字段：{col_lack}')
            if col_more:
                warnings.warn(f'多出{len(col_more)}个字段：{col_more}'); time.sleep(seconds)
            table = table[col_all]
            print(f'table.shape:{table.shape}')

            f = Info._asdict()[f"table_{step}"]
            print('\n', end='')
            print(f'保存宽表结果至：{f}\n')
            table.to_csv(f, index=False, encoding="utf_8_sig")

            # 缺失值 inf 检验
            # 无需检验supply_other字段，其有缺失值是正常可接受的
            # 因为在前文的处理过程中：其中的数值型和类别型字段缺失值会被处理
            #                         但如果包括日期型字段，则不会被处理
            supply_other_date = Info.comment_all.loc[Info.comment_all.field_name.isin(Info.supply_other) &
                                                     (Info.comment_all.dtype_classify == '日期型'), 'field_name']
            supply_other_date = set(supply_other_date) & set(table.columns)  # 预测时可能缺少condition字段
            na_inf_examine(table.drop(columns=supply_other_date))

            col_count = table.columns.value_counts()
            col_count = col_count[col_count > 1]
            if len(col_count) > 0:
                raise Exception(f'存在重复字段：\n{col_count}')

            if Info.table_r:
                print('计算字段之间的相关性系数')
                # 仅统计宽表探索结果特征的相关性，与宽表探索的时的相关性对比
                # 差异：宽表探索阶段基于woe编码的字段统计相关性，宽表生成阶段针对未woe编码的字段统计
                col_no_r = (Info.supply_must | Info.supply_other | set(Info.col_mark) | {Info.col_target}) & set(table.columns)
                table_r = table_r_fun(table.drop(columns=col_no_r))  # 预测集无col_target，故& set(table.columns)
                print(f"{add_printindent(table_r.abs().describe().round(3))}\n")

            psi_mark = re.sub('^.*/|\.csv|\.txt', '', Info._asdict()[f"table_{step}"])
            psi_mark = psi_mark[psi_mark.index(str(Info._asdict()[f"month_{step}"])):]
            psi_mark = str(Info.month_train) + ('' if step == 'train' else '~' + psi_mark)  # ~训练账期[~当前账期]
            f_psi = f"{wd}/{step}_Psi~{psi_mark}.pkl"
            if Info.table_psi:
                print('\n------------------------------- 计算宽表字段psi ------------------------------------ ')
                Psi = None
                if step == 'train':
                    print('训练集 data_train')
                    # 仅对特征检验稳定度（宽表探索的结果特征 + 基础数据字典field_base中must_remain强制保留的特征字段）
                    col_no_x = (Info.supply_other | set(Info.col_mark) | {Info.col_target}) & set(table.columns)
                    data_train = table.loc[table.data_use == 'data_train', col_all]
                    X_train = data_train.drop(columns=col_no_x)
                    X_train.data_name ='data_train'
                    y_train = data_train[Info.col_target]

                    Psi = PsiTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, psi_limit=0.1)
                    Psi.fit(X_train, y_train)  # 预测集无col_target，故& set(table.columns)

                    data_timein = table.loc[table.data_use == 'data_timein']
                    if len(data_timein) > 0:
                        print('\n')
                        print('时间内验证集 data_timein')
                        X_timein = data_timein.drop(columns=Info.col_target)
                        X_timein.data_name ='data_timein'
                        X_timein.y_carrier = data_timein[Info.col_target]
                        _ = Psi.transform(X_timein)

                    print('\n')
                    Info = choradd_namedtuple(Info, {'train_Psi': Psi})
                    file_info2 = Info.model_wd_traintest + '/Info~tabcre_train.pkl'
                    print(f'保存Info至：{file_info2}')
                    privy_fileattr_fun(file_info2, 'unhide')
                    joblib.dump(Info._asdict(), file_info2)
                    privy_fileattr_fun(file_info2)
                else:
                    if 'train_Psi' not in Info._asdict().keys():
                        s = f"step='train'时Info.table_psi=False，{step}无PsiTransformer_DF可供计算psi，跳过，不计算!"
                        print(s)
                        warnings.warn(s); time.sleep(seconds)
                    else:
                        Psi = Info.train_Psi
                        X_new = table.copy()
                        if step == 'test':
                            X_new.data_name = 'data_timeout'
                            X_new.y_carrier = X_new[Info.col_target]
                        elif step == 'predict':
                            X_new.data_name = 'data_predict'

                        col_rid = [i for i in Psi.fit_in_colnames_ if i not in X_new.columns]
                        for i in col_rid:
                            # 随便一个非空取值，让DataFram中补齐这些列就行（np.nan将通不过transform结尾通用的缺失值检验）
                            X_new[i] = 1
                        Psi.col_ignore = col_rid
                        _ = Psi.transform(X_new)

                        print(f'保存Psi至：{f_psi}')
                        privy_fileattr_fun(f_psi, 'unhide')
                        joblib.dump(Psi, f_psi)
                        privy_fileattr_fun(f_psi)

                    save_mark = re.sub('^.*/|\.csv|\.txt', '', f)
                    save_mark = get_onlyvalue(re.findall('~.*$', save_mark))
                    f2 = f"{wd}/{step}_dis_exam{save_mark}.pkl"
                    print(f'保存dis_exam至：{f2}')
                    joblib.dump(dis_exam, f2)

        end_time = datetime.datetime.now()
        print(f"结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time_cost = (end_time - start_time).seconds
        print(f"耗时：{time_cost} s")
        # </editor-fold> -----------------------------------------------------------------------------------------------
    filter_warnings(w)
    return table, Info


# 统一不同user_acct_month取值的数据
# 统一示例：# 统一前 # 统一后
# user_acct_month  acct_month   user_id   X    Y    # user_acct_month acct_month  user_id    X    Y
#       202103       202101       1      ,,,  ...   #       202105      202103    202103_1  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202103       202102       2      ,,,  ...   #       202105      202104    202103_1  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202103       202103       3      ,,,  ...   #       202105      202105    202103_1  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...

#       202104       202102       1      ,,,  ...   #       202105      202103    202104_2  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202104       202103       2      ,,,  ...   #       202105      202104    202104_2  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202104       202104       3      ,,,  ...   #       202105      202105    202104_2  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...

#       202105       202103       1      ,,,  ...   #       202105      202103    202105_3  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202105       202104       2      ,,,  ...   #       202105      202104    202105_3  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...
#       202105       202105       3      ,,,  ...   #       202105      202105    202105_3  ,,,  ...
#         ...          ...        ...    ,,,  ...   #         ...         ...       ...     ,,,  ...

