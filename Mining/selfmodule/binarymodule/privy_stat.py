import os
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import datetime
import time
from collections import OrderedDict
pd.set_option('display.width', 100000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)

from Mining.selfmodule.tablemodule.basestr import *
from Mining.selfmodule.toolmodule.datatrans import *
from Mining.selfmodule.tablemodule.tablefun import month_add, month_list_fun, wherefun, field_pair_fun
from Mining.selfmodule.binarymodule.traintest import pre_result_fun
from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, choradd_namedtuple, get_onlyvalue, dropdup
from Mining.selfmodule.toolmodule.strtotable import infos_to_table
seconds = 3


def privy_target_stat(sqltype, infos, start=None, end=None, wave_limit=10/100, sqlcomment='----'):
    """
    统计目标字段表中所有目标字段的分布（未限制模型的目标用户，初步查看各账期之间分布是否平稳）
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param infos: DataFrame，个人维护的所有模型的信息
    :param start: 起始账期（包含）
    :param end: 结束账期（包含）
    :param wave_limit: 两个账期之间数据相差太大，发出警告的阈值
    :param sqlcomment: sql的注释字符
    :return: 目标字段的分布统计结果
    """
    con = []
    if start is not None:
        con.append(f'#m# >= {sql_value(start)}')
    if end is not None:
        con.append(f'#m# <= {sql_value(end)}')
    where = wherefun(con)

    stat_set = {}

    table_info = infos_to_table(s_table_info, col_eval=['tableXday_desc', 'tableXscore_desc'], col_index=None)
    table_info = table_info[table_info.s_field_base.isin(infos.s_field_base)]
    tableys = table_info.loc[table_info.tabletype == 'tableY', ['s_field_base', 'tablename']]
    for i in tableys.tablename.unique():
        print(f"\n\n{sqlcomment}--------------------- {i} ----------------")
        infos_sub = infos.loc[infos.s_field_base.isin(tableys.loc[tableys.tablename == i, 's_field_base'])]
        print(f"{sqlcomment} {list(infos_sub.index)}")
        col_month = get_onlyvalue(infos_sub.col_month.unique())
        s_count = "sum(case when " + infos_sub.col_target + "=" + infos_sub.Pcase.apply(sql_value) + " then 1 else 0 end) " + infos_sub.col_target
        s_count = ',\n'.join(s_count)
        sql_stat = sql_format(f"""
            select {col_month},
            count(1) count_all,
            {s_count}
            from {i}
            {where.replace('#m#', col_month)}
            group by {col_month}
            order by {col_month}
        """)
        sql_show(f'\n{sqlcomment} sql语句：', sql_stat)
        if sqltype == 'execute':
            count_stat = my_sql_fun(sql_stat, method='read')
            print(f"\n统计正例样本量：\n{count_stat}")
            for model_name in infos_sub.model_name:
                col_target, month_train = list(infos_sub.loc[model_name, ['col_target', 'month_train']])
                data_month = count_stat[infos_sub.loc[model_name, 'col_month']]
                mlist = data_month[data_month >= month_train]  # 账期字段强制统一按类别型处理

                wave_m = ''
                for idx in range(len(mlist.iloc[:-1])):
                    m1, m2 = mlist.iloc[idx: (idx + 2)]
                    count_m12 = count_stat.loc[count_stat[col_month].isin([m1, m2]), col_target]
                    cmin, cmax = count_m12.min(), count_m12.max()
                    wave = (cmax - cmin) / cmin
                    if wave >= wave_limit:
                        wave_m += f'{m1}与{m2} '
                if wave_m:
                    s = f"\n{model_name}: {wave_m}正例数据量相差较大，请确认！"
                    warnings.warn(s); time.sleep(seconds)

            stat_set[i] = count_stat
    if sqltype == 'execute':
        return stat_set


def privy_score_evaluate1(sqltype, month_perform, infos, db_score, score_id=None, n_ev=6, col_exesam='exe_sam',
                          sqlcomment='---', db_score_flag=prefix + 'table_score_flag_month'):
    """
    模型分数监控函数1 (在数据库中生成分数监控所需中间表)
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param month_perform: 表现期账期
    :param infos: DataFrame，个人维护的所有模型的信息
    :param db_score: 分数表表名
    :param score_id: 分数表的用户标识字段名，如果无需设置，根据各自模型的Info.col_id获取用户标识字段，即score_id=None
    :param n_ev: 分数需要跟踪n_ev个账期
    :param col_exesam: 分数表中区分执行组与对比组的字段名，若无该字段，可令col_exesam=None，不区分，都是执行组
    :param sqlcomment: sql的注释字符
    :param db_score_flag: 分数监控所需的 分数+目标字段中间表表名
    :return:
    """
    print(f'---------------------------------------- 关联个账期分数与目标字段 ----------------------------------------')
    table_info = infos_to_table(s_table_info, col_eval=['tableXday_desc', 'tableXscore_desc'], col_index=None)

    col_es = f',x.{col_exesam}'if col_exesam else ''
    sqls = []  # 每个模型每个统计周期的 分数与其对应账期的table_y的关联语句
    for model_name in infos.index:
        print(f"\n\n{sqlcomment}~ {model_name}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ~".replace('~', '-'*60))
        Info = to_namedtuple(infos.loc[model_name])
        table_y = table_info.loc[(table_info.s_field_base == Info.s_field_base) & (table_info.tabletype == 'tableY'), "tablename"].iloc[0]

        month_score = month_add(month_perform, -Info.target_lag)
        slist = month_list_fun(month_score, periods=-n_ev)
        slist = [str(i) for i in slist]  # 统一账期字段的类型
        print(f'{sqlcomment} 确定监控的分数账期列表(近{n_ev}个账期)：{slist}')

        score_id = score_id if score_id else Info.col_id
        for m in slist:
            print(f"    {sqlcomment}准备sql：关联{m}账期分数与其{month_perform}账期目标字段")
            stat_batch = len(month_list_fun(m, month_perform)) - 1  # 统计周期，即监控分数在stat_batch月后的的实际表现
            sql_m = sql_format(f'''
            select * from(
                select x.*, 
                {sql_value(month_perform)} ymonth, {stat_batch} stat_batch,  '{Info.col_target}' col_flag, y.{Info.col_target} flag_value {col_es}
                from (select * from {db_score} where {Info.col_month}={sql_value(m)} and model_name='{model_name}') x
                left join (select * from {table_y} y where {Info.col_month}={sql_value(month_score)}) y on x.{score_id}=y.{Info.col_id}
            ) %s''')
            sqls.append(sql_m)

    sql_union = '#blank#union all#blank#'.join(sqls)
    sql_create = sql_format(f'''
        drop table if exists {db_score_flag};
        create table {db_score_flag} as
        {sql_union}
        ''')
    sql_create = sql_create.replace('#blank#', '\n\n') % tuple('t' + Series(range(sql_create.count('%s'))).astype(str))
    sql_show(f"\n{sqlcomment}-------------------------- 生成最终sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:", sql_create)
    if sqltype == 'execute':
        my_sql_fun(sql_create, method='execute')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def privy_score_evaluate2(infos, n_ev, col_exesam="exe_sam", db_score_flag=prefix + 'table_score_flag_month'):
    """
    模型分数监控函数2(读取privy_score_evaluate1生成的中间表进行分数监控)
    :param infos: DataFrame，个人维护的所有模型的信息
    :param n_ev: 分数需要跟踪n_ev个账期
    :param col_exesam: 目标字段表中区分执行组与对比组的字段名， 若无该字段，可令col_exesam=None，不区分，都是执行组
    :param db_score_flag: privy_score_evaluate1加工的中间表表名
    :return: 打印分数监控效果，并返回各模型、各账期、执行组/对比组的测试效果
    """
    col_exesam_none = col_exesam is None
    eval_tab = DataFrame()
    score_result = {}
    top_idx = {}
    top_idx['Top20%'] = 3  # 位置下标
    top_idx['Top40%'] = 4  # 位置下标
    for model_name in infos.index:  # 遍历各模型
        Info = to_namedtuple(infos.loc[model_name])
        for b in range(1, n_ev+1):  # 遍历近n_ev个分数账期
            print(f"\n\n* {model_name} stat_batch={b}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *".replace('*', "*"*80))
            condition = f"model_name='{model_name}' and stat_batch={b}"  # 限定特定模型特定周期的
            col_char = [Info.col_month, Info.col_id, 'flag_value', 'ymonth']
            data_sub = read_data(db_score_flag, col_char=col_char, condition=condition)
            if len(data_sub) == 0:
                print(f"该账期分数缺失，跳出循环")
                break
            elif len(data_sub) > 0:
                if col_exesam_none:
                    print('col_exesam为None，默认全部为执行组')
                    col_exesam = 'exe_sam'
                    data_sub[col_exesam] = '执行组'

                na_flag = data_sub.flag_value.isnull().sum()
                if na_flag:
                    month_xy = get_onlyvalue(data_sub[Info.col_month].astype(str) + ' -> ' + data_sub['ymonth'].astype(str))
                    nrowold = len(data_sub)
                    s = f"{model_name} stat_batch={b}: {month_xy} 的目标字段存在{na_flag}个缺失值"
                    if Info.score_targetnull == 'del':
                        data_sub = data_sub.loc[data_sub.flag_value.notnull()]
                        nrownew = len(data_sub)
                        s_ad = s + f'，将目标字段缺失记录删除（{nrowold} -> {nrownew}）！'
                    elif Info.score_targetnull == 'Pcase':
                        to = str(Info._asdict()[Info.score_targetnull])
                        data_sub["flag_value"] = data_sub["flag_value"].fillna(str(to))
                        s_ad = s + f"，将此目标字段缺失值赋值为Pcase({to})！"
                    elif Info.score_targetnull == 'Ncase':
                        to = str(Info._asdict()[Info.score_targetnull])
                        data_sub["flag_value"] = data_sub["flag_value"].fillna(str(to))
                        s_ad = s + f"，将此目标字段缺失值赋值为Ncase({to})！"
                    warnings.warn(s_ad); time.sleep(seconds)

                for es in data_sub[col_exesam].unique():  # 遍历 执行组、样本组
                    print(f"监控{es}效果")
                    data_es = data_sub.loc[data_sub[col_exesam] == es]
                    pre_result_predict = pre_result_fun(data_es.score, data_es.flag_value.astype(str), str(Info.Pcase), str(Info.Ncase))
                    row = Series(dtype=object)
                    row['模型名称'] = model_name
                    row['统计周期'] = b
                    row['分数账期'] = get_onlyvalue(data_es.acct_month)
                    row['表现期'] = get_onlyvalue(data_es.ymonth)
                    row['执行/对比组'] = es
                    row['目标用户池_人数'] = len(data_es)
                    row['目标用户池_正例人数'] = len(data_es.loc[data_es.flag_value == Info.Pcase])
                    row['目标用户池_自然率'] = row['目标用户池_正例人数'] / row['目标用户池_人数']

                    for top, idx in top_idx.items():
                        row[f'{top}_人数'] = pre_result_predict.累计人数.iloc[idx]
                        row[f'{top}_正例人数'] = pre_result_predict[f'累计人数_{Info.Pcase}'].iloc[idx]
                        row[f'{top}_查准率'] = pre_result_predict['累计查准率'].iloc[idx]
                        row[f'{top}_查全率'] = pre_result_predict['累计查全率'].iloc[idx]
                        row[f'{top}_提升度'] = pre_result_predict['累计提升度'].iloc[idx]
                    eval_tab = pd.concat([eval_tab, DataFrame(row).T])
                    score_result[f"{model_name}-{b}-{es}"] = pre_result_predict
    print("#"*180)
    eval_tab.模型名称[eval_tab.模型名称.duplicated()] = ' '
    eval_tab = eval_tab.set_index(['模型名称'])
    c = eval_tab.columns.str.split('_')
    c1 = Series(c.map(lambda x: x[0]))
    c1[c1.duplicated()] = [('-' + ' ' * i) for i in range(c1.duplicated().sum())]
    c2 = c.map(lambda x: '-' if len(x) == 1 else x[-1])
    eval_tab.columns = [c1, c2]
    print(f"\n监控效果汇总：\n{eval_tab}")
    return score_result, eval_tab


def privy_table_all_stat(infos, prefix=prefix):
    """
    统计所有特征目标汇总表
    :param infos: DataFrame，个人维护的所有模型的信息
    :param prefix: 数据库表名前缀 如：ml.tablename中的“ml.”
    :return:
    """
    Infos = Series(infos.index).apply(lambda x: to_namedtuple(infos.loc[x]))

    # 与 privy_basedatafun函数 中的table_all命名规则保持一致
    tabs_t = {f"dm_zc{Info.user_type}_moxing{Info.col_id_type}_info_{'target'}_": Info.col_month for Info in Infos}
    tabs_a = {f"dm_zc{Info.user_type}_moxing{Info.col_id_type}_info_{'add'}_": Info.col_month for Info in Infos}

    for i, col_m in dict(tabs_t, **tabs_a).items():
        print(f'\n------------------------ {i} -------------------------------')
        like = f"{i}{wildcard_any}"
        tables = sorted(table_exists(like=like))
        fields = {}
        shapes = Series(dtype=object)
        for j in tables:
            fields[j] = ', '.join(get_field(j))
            shapes[j] = (get_onlyvalue(get_month_count(j, col_m)['count']), len(fields[j]))
        if tables:
            print(f"\n{shapes}")
            rows = shapes.apply(lambda x: x[0])
            sep = rows.std() / rows.mean()
            if sep > 10 / 100:
                s = f"\n{prefix}{like} 系列表的行数浮动较大，请确认（如有已失效的表，删除）！"
                warnings.warn(s); time.sleep(seconds)
        if len(set(fields.values())) > 1:
            s = f"\n{prefix}{like} 系列表的表结构不同，请删除其中已失效的表"
            warnings.warn(s); time.sleep(seconds)


def privy_select_tables(infos, step, mode='history'):
    """
    查询建模数据集的表名
    :param infos: DataFrame，个人维护的所有模型的信息
    :param step: 取值 traintest：查询训练集、测试集的所有表名
                 取值 predict：查询预测集的所有表名
    :param mode: 取值 history：查询小于目前账期的历史表名
                 取值 current：查询等于目前账期的当前表名
                 取值 total：  查询所有账期的表名
    :return: DataFrame，table_full_name字段为查询的表名
    """
    sql = """
    select * from (
        SELECT
            table_schema || '.' || table_name AS table_full_name,
            pg_size_pretty(pg_total_relation_size('"' || table_schema || '"."' || table_name || '"')) AS size
        FROM information_schema.tables
        ORDER BY
        pg_total_relation_size('"' || table_schema || '"."' || table_name || '"') DESC 
    ) t 
    where table_full_name like '%s'"""

    table_select = DataFrame()
    for i in {'traintest': ['train', 'test'], 'predict': ['predict']}[step]:
        for model_name, short_name, month in infos[['model_name', 'short_name', f"month_{i}"]].values:
            like = f"ml.mid_{short_name}_recent_{i}_%"
            sql_i = sql % like
            tables = my_sql_fun(sql_i, method='read')  # 两列: table_full_name（必须） size（可选）
            tables.index = [model_name] * len(tables)
            if len(tables) > 0:
                if mode == 'history':
                    tab = tables.loc[tables.table_full_name < like.replace('%', str(month))]
                elif mode == 'current':
                    tab = tables.loc[tables.table_full_name == like.replace('%', str(month))]
                elif mode == 'total':
                    tab = tables
                else:
                    raise Exception(f"mode参数取值为{mode}，应为'history', 'current', 'total'")
                table_select = pd.concat([table_select, tab])
    if len(table_select):
        print(table_select)
    else:
        print('未查询到相关表')
        table_select = DataFrame(columns = ['table_full_name'])
    return table_select


def privy_drop_tables(x):
    """
    删除表
    :param x: 表名字符串、或表名序列
    :return: None
    """
    if isinstance(x, str):
        x = [x]
    drops = [f'drop table if exists {i}' for i in x]
    if drops:
        for i in drops:
            sql_show('删表：', i)
            my_sql_fun(i, method='execute')
    else:
        print('无表可删除')


def privy_selectfile(allfile, filecontain, dirupto='', mode=None):
    """
    查询符合一定条件的文件
    :param allfile: DataFrame，所有文件的信息列表，getAllFilesize函数的返回结果
    :param filecontain: 文件名称包含filecontain
    :param dirupto: 文件夹截至，用于判断各自模型目录下，同类文件的新旧
                    新的文件夹：当前账期的文件下
                    旧的文件夹：旧账期的文件夹下
    :param mode: 取值 old：   查询旧的文件名
                 取值 newest：查询新的文件名
                 取值 total： 查询所有文件名
    :return: print所有文件名，返回mode对应的文件名
    """
    select_file = allfile.loc[allfile.filename.str.contains(filecontain) &
                              (allfile.dir.str.contains(dirupto))].sort_values(by='dir').copy()
    if len(select_file) == 0:
        print('未查询到符合条件的文件，返回None')
        return None

    if dirupto:
        if mode is None:
            print('mode为None，默认其为total')
            mode = 'total'
        dirs = select_file.dir.str.replace(f'{dirupto}.*$', dirupto).drop_duplicates()

        for i in dirs:
            con_dir = select_file.dir.str.contains(i)
            dirs_i = sorted(select_file.loc[con_dir, 'dir'].unique())
            oldnew = dict(zip(dirs_i, ['old']*(len(dirs_i)-1) + ['newest']))
            select_file.loc[con_dir, 'dirstatus'] = select_file.dir.map(oldnew)

        select_file['是否删除'] = '否'
        if mode == 'old':
            select_file.loc[select_file.dirstatus == 'old', '是否删除'] = '是'
        elif mode == 'newest':
            select_file.loc[select_file.dirstatus == 'newest', '是否删除'] = '是'
        elif mode == 'total':
            select_file.loc[select_file.dirstatus.isin(['old', 'newest']), '是否删除'] = '是'
        else:
            raise Exception(f"mode参数取值为{mode}，应为'old', 'newest', 'total'")
        select_file.loc[select_file.是否删除 == '是']
    else:
        if mode is not None:
            warnings.warn("dirupto=''时，mode失效"); time.sleep(seconds)
        select_file['是否删除'] = '是'
    print(select_file)
    return select_file.loc[select_file.是否删除 == '是']


def privy_delfile(file_df):
    """
    批量删除模型保存路径下的无用文件
    :param file_df: privy_selectfile函数的返回结果
    :return: None，仅删除文件
    """
    if file_df is None:
        print("输入为None，无需操作")
    elif len(file_df) == 0:
        print('输入为0行，无需操作')
    else:
        files = file_df.dir + '/' + file_df.filename
        for i in files:
            if os.path.exists(i):
                print(f"删除 {i}")
                os.remove(i)
            else:
                print(f"不存在 {i}")

