import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame
import time
import datetime
import re

from Mining.selfmodule.toolmodule.dataprep import add_printindent

seconds = 5

# 数据库
# 本机、内蒙古云平台：db = 'gp'
# 内蒙古集团专区：db = 'hive'
# 内蒙古亚信jupyter: db = 'hive'
db = 'gp'


def sql_value(x):
    """
    为拼接sql中的取值：字符型加引号
    :param x: python中取值
    :return: sql中其加入筛选条件中的取值
    """
    return ("'%s'" if type(x) == str else '%s') % x


# ----------------- <editor-fold desc="为不同数据库配置：连接数据库、py与数据库字段类型对应、随机函数等"> ---------------------
if db == 'gp':
    # 连接数据库
    import psycopg2
    from sqlalchemy import create_engine
    dbname = 'postgres'
    user = 'postgres'
    pwd = 'postgres'  # 本机:' xingjing' 云主机: 'admin'
    port = '5432'
    host = '192.168.100.112'
    paradict = {'dbname': dbname, 'user': user, 'pwd': pwd, 'port': port, 'host': host}
    # prefix = 'ml.'  # 表名前缀
    prefix = 'kehujingyingbudb.'
    type_py_sql = {str: 'text', int: 'int', float: 'numeric'}  # 用于sql中转换字符型 cast(字段 as ?)
    sqlrandfun = 'random()'  # 随机函数
    wildcard_any = '%'       # 通配符：任意字符
    percentile_fun = lambda colname, p: f"percentile_cont({p}) within group (order by {colname})"

    # 分区操作
    def part_sql(tablename, partinfo):
        """
        数据库分区语句的统一接口
        :param tablename: 表名
        :param partinfo: 分区信息，列表 [(分区字段1， 字段1取值，字段1类型), (分区字段2， 字段1取值，字段2类型), ...]
        :return: sql语句字符：设置分区键、清空分区、向分区插入数据
        """
        part = f"partition by list({partinfo[0][0]})"  # 设置分区键sql，col为分区键字段名
        if len(partinfo) == 1:
            partcol, partvalue, parttype = partinfo[0]
            parttablename = f"{tablename}_{partvalue}"
            clear = sql_format(f"""
                create table if not exists {parttablename} partition of {tablename} for values in ({sql_value(partvalue)});
                truncate table {parttablename};""") + ';'  # 创建/清空分区 sql_format会把结尾分号删除（为兼容某些环境）
        elif len(partinfo) > 1:
            clear = ""
            tablename_last = tablename
            parttablename = tablename
            for i in range(len(partinfo)):
                partvalue = partinfo[i][1]
                if i < len(partinfo)-1:
                    partcol_next = f"partition by list({partinfo[i+1][0]})"
                else:
                    partcol_next = ""
                parttablename += "_" + re.sub('^.*\.', '', partvalue)
                clear += f"\ncreate table if not exists {parttablename} partition of {tablename_last} for values in ({sql_value(partvalue)}) {partcol_next};"
                tablename_last = parttablename
            clear += f"\ntruncate table {parttablename};"
        insert = f"insert into {tablename}"  # 向分区insert 数据
        return part, clear, insert

elif db == 'hive':
    from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField
    from pyspark.sql import HiveContext, SparkSession
    from pyspark import SparkContext,SparkConf
    from pyspark.sql import HiveContext, SparkSession

    # # 内蒙古集团专区
    # _SPARK_HOST = ""
    # _APP_NAME = ""
    # spark_session = SparkSession.builder.master(_SPARK_HOST).appName(_APP_NAME).getOrCreate()
    # sqlContext = HiveContext(spark_session)
    # prefix = 'cq_pro1_role2.'  # 表名前缀

    # 内蒙古亚信jupyter
    spark_session = SparkSession.builder.appName("Test")                          \
        .config("spark.hive.mapred.supports.subdirectories", "true")           \
        .config("mapreduce.input.fileinputformat.input.dir.recursive", "true") \
        .config("spark.network.timeout", "10000001")                           \
        .config("spark.executor.heartbeatInterval", "10000000")                \
        .config("spark.rpc.askTimeout", "10000000")                            \
        .config("spark.executor.extraJavaOptions", "-XX:+UseConcMarkSweepGC")  \
        .config("spark.driver.memory", '10g')                                  \
        .config("spark.driver.memoryOverhead", '6g')                           \
        .config("spark.executor.memory", '10g')                                \
        .config("spark.executor.memoryOverhead", '6g')                         \
        .config("spark.sql.debug.maxToStringFields", 1000000)                  \
        .config('spark.sql.legacy.allowNonEmptyLocationInCTAS', "true")        \
        .enableHiveSupport().getOrCreate()

    prefix = 'kehujingyingbudb.'  # 表名前缀

    paradict = {}

    type_py_sql = {str: 'string', int: 'int', float: 'double'}  # 用于sql中转换字符型 cast(字段 as ?)
    sqlrandfun = 'rand()'  # 随机函数
    wildcard_any = '*'     # 通配符：任意字符
    percentile_fun = lambda colname, p: f"percentile_approx({colname}, {p})"

    # 分区操作
    def part_sql(tablename, partinfo):
        """
        数据库分区语句的统一接口
        :param tablename: 表名
        :param partinfo: 分区信息，列表 [(分区字段1， 字段1取值，字段1类型), (分区字段2， 字段1取值，字段2类型), ...]
        :return: sql语句字符：设置分区键、清空分区、向分区插入数据
        """
        p1 = ', '.join([f"{i[0]} {i[2]}" for i in partinfo])
        part = f"partitioned by ({p1})"  # 设置分区键sql，col为分区键字段名
        clear = ""  # 清空分区
        p2 = ', '.join([f"{i[0]}={sql_value(i[1])}" for i in partinfo])
        insert = f"insert overwrite table {tablename} partition ({p2})"  # 向分区insert 数据
        return part, clear, insert

    def df_to_hive(df, destination, if_exists=None):
        """
        将数据导入hive
        :param df: DataFrame
        :param destination: 数据库表名
        :param if_exists => mode:
                  `append` => `append`: Append contents of this :class:`DataFrame` to existing data.
                  `replace` => `overwrite`: Overwrite existing data.
                  `fail`  => `error`: Throw an exception if data already exists.
        :return: None
        """
        if if_exists is None:
            if_exists = 'fail'
        mode_convert = {'fail': 'error', 'replace': 'overwrite', 'append': 'append'}  # 保持不同数据库参数取值一致
        mode = mode_convert[if_exists]

        df = df.copy()
        pre, tablename = destination.split('.')

        # ------------- DataFrame 有权限直接写入hive（saveAsTable）：
        df = df.apply(lambda x: x.where(x.notnull(), None))

        if table_exists(tablename=tablename, prefix=pre) & (mode == 'append'):  # DataFrame 较 数据库表 字段多错少补
            f = get_field(destination).index
            col_more = set(df.columns) - set(f)
            if col_more:
                s = f"数据库表{destination}的 {', '.join(col_more)} 字段不存在！"
                raise Exception(s)
            col_lack = set(f) - set(df.columns)
            if col_lack:
                print(f"DataFrame 较 数据库表 缺少下列字段,用空值补齐：{col_lack}")
                for c in col_lack:
                    df[c] = None
                df = df[f]

        py_spark_type = {'float': DoubleType(), 'int': IntegerType(), 'object': StringType()}
        dt = df.dtypes.astype(str).str.replace('64.*?$|32.*?$|16.*?$|8.*?$', '')
        dt_convert = []
        for i in dt.index:
            dt_convert.append(StructField(i, py_spark_type[dt[i]], True))
        schema = StructType(dt_convert)
        cre = spark_session.createDataFrame(df, schema=schema)
        cre.write.saveAsTable(destination, mode=mode)
        # </editor-fold> -----------------------------------------------------------------------------------------------


def my_sleep_sql(seconds=seconds, db=db):
    """
    适应不同数据库的延时操作sql
    :param seconds: 延时时长
    :param db: 数据库名称，如gp、hive等
    :return: sql字符串
    """
    if seconds is None:
        return ''
    if db == 'gp':
        return f'select pg_sleep({seconds});'
    elif db == 'hive':
        return ''


def my_monthadd_sql(colname, m_add, db=db):
    """
    适应不同数据库的账期增加月份数的sql
    :param colname: 字段名，形如yyyymm
    :param m_add: 增加的月份数，可以为负数
    :param db: 数据库名称，如gp、hive等
    :return: sql字符串
    """
    if db == 'gp':
        return f"to_char(to_date({colname}::text, 'yyyymm') + interval '{m_add} month', 'yyyymm') "
    elif db == 'hive':
        return f"from_unixtime(unix_timestamp(add_months(from_unixtime(unix_timestamp(cast({colname} as string), 'yyyyMM'), 'yyyy-MM-dd HH:mm:ss'), {m_add}), 'yyyy-MM-dd'), 'yyyyMM')"


def my_sql_fun(sql, method, index_col=None, db=db, paradict=paradict, seconds=seconds, scope=None, indent='', **kwargs):
    """
    适应不同数据库中：读取sql、执行sql、将python中数据导入数据库
    :param sql: 当method取值为read、execute时，该参数为sql语句字符串；
                当method取值为todb时，insert into tablename select * from df，df为python中DataFrame、tablename为数据库表名
                                      目前仅支持 select * from df，如需更灵活的句式，请扩展。
    :param method: 取值为read：将数据库中数据读取到python中；
                   取值为execute：执行sql；
                   取值为todb，将python中的数据导入数据库
    :param index_col: 索引列，若取值为None，则不设置索引列
    :param db: 数据库名称，如gp、hive等
    :param kwargs: 关键字参数
    :return: 取值为read：返回DataFrame；取值为execute：执行sql，无返回
    """
    def get_df_dest(sql, scope):
        # method='todb'时，用于从sql字符串中获取DataFrame和数据库表名
        sql_ad = re.sub(' +', ' ', sql.strip(' '))
        destination = sql_ad[:sql_ad.index(' select')].replace('insert into ', '')
        df = sql_ad[(sql_ad.rfind(' ') + 1):]
        df = scope[df]
        return df, destination

    if db == 'gp':
        # sql执行错误后，再执行sql将报错：当前事务被终止。故每次执行sql前重新定义连接
        dbname, user, pwd, port, host = paradict['dbname'], paradict['user'], paradict['pwd'], paradict['port'], paradict['host']
        conn1 = psycopg2.connect(dbname=dbname, user=user, password=pwd, port=port, host=host, client_encoding='UTF-8')
        conn2 = create_engine(f'postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{dbname}')

        if method == 'read':
            data_res = pd.read_sql(sql, conn1, index_col=index_col, **kwargs)
            return data_res
        elif method == 'execute':
            conn1.cursor().execute(sql)
            conn1.commit()
            print(f'{indent}sql执行完毕')
        elif method == 'todb':
            df, destination = get_df_dest(sql, scope)
            schema, tablename = destination.split('.')
            pd.io.sql.to_sql(df, tablename, conn2, schema=schema, index=False, **kwargs)
            print(f'{indent}导入完毕')
        else:
            raise Exception(f"method取值{method}有误，应为read、execute、todb")
        conn1.close()
    elif db == 'hive':
        if method == 'read':
            data_res = spark_session.sql(sql).toPandas()
            data_res = data_res.rename(columns={'count(1)': 'count'})  # 为与gp的读取结果保持一致
            if index_col is not None:
                data_res = data_res.set_index(index_col)
            return data_res
        elif method == 'execute':
            sqls = [i.strip('\n') for i in sql.split(';')]
            for i in sqls:
                spark_session.sql(i)
            print(f'{indent}sql执行完毕')
            if seconds is not None:
                time.sleep(seconds)
        elif method == 'todb':
            df, destination = get_df_dest(sql, scope)

            df_to_hive(df, destination, **kwargs)
            print(f'{indent}导入完毕')
        else:
            raise Exception(f"method取值{method}有误，应为read、execute、todb")


def table_exists(tablename=None, like=None, prefix=prefix, db=db, indent=''):
    """
    判断数据库中的表名(按本地数据库修改)
    :param tablename: 按表名搜索(不带 xxx. 前缀)
    :param like: 模糊匹配
     :param prefix: 表名前缀，即数据库名称/模式名称 + 点号
    :param db: 数据库名称，如gp、hive等
    :return: 表名列表（如果无则返回空列表）
    """
    x = f"like '{like}'" if like else f"prefix='{prefix}'' tablename='{tablename}'"
    print(f'查询表是否存在（{x}）')
    if (tablename is not None) & (like is not None):
        s = "tablename、like参数同时设置，仅关注like参数，忽略tablename参数"
        warnings.warn(s); time.sleep(seconds)

    if db == 'gp':
        where = '' if (prefix is None) & (tablename is None) & (like is None) else 'where '
        ts = f"table_schema = '{prefix.replace('.', '')}' and " if prefix else ''
        if like is not None:
            tn = f"table_name like '{like}'"
        elif tablename is not None:
            tn = f"table_name='{tablename}'"
        else:
            tn = ''
        sql = sql_format(f"""
            select concat(table_schema, '.', table_name) full_name
            from information_schema.tables 
            {where}{ts}{tn}""")
        sql_show(f'{indent}查询表：', add_printindent(sql, indent+'    '))
        tables = list(my_sql_fun(sql, method='read').full_name)
        if tables:
            print(add_printindent(f'存在{len(tables)}个表 ', indent+'    '))
        else:
            print(add_printindent('表不存在 ', indent+'    '))

    elif db == 'hive':
        # like、tablename参数其实一样，因为只有like的用法，所以约定（非语法强制）：
        #     tablename：不使用通配符时设置
        #     like：使用通配符时设置
        if like is not None:
            tn = f"like '{like}'"
        elif tablename is not None:
            tn = f"like '{tablename}'"
        else:
            tn = ''
        if prefix is not None:
            ts = f"use {prefix.replace('.', '')}"
            sql = f"show tables {tn}"
            # my_sql_fun(ts, method='execute')
            spark_session.sql(ts)
            sql_show(f'{indent}查询：', add_printindent(sql, indent + '    '))
            tables = my_sql_fun(sql, method='read')
            tables = list(tables.namespace + tables.tableName)
            if tables:
                print(add_printindent(f'存在{len(tables)}个表 ', indent + '    '))
            else:
                print(add_printindent('表不存在 ', indent + '    '))
        else:
            # 理论上可以获取所有数据库名称，遍历查询每个数据库下的表，如果确定要这么操作，可扩充该分支
            s = '请指定数据库（prefix参数）！'
            raise Exception(s)

    return tables


def get_field(tablename, db=db):
    """
    统计数据库表某表字段名列表,并附带字段类型，
    :param tablename: 表名
    :return: Series，索引为字段名，值为字段类型
    """
    if tablename is None:
        return []
    # # 这种方法可兼容各类数据库
    # sql_field = f"select * from {tablename} limit 1"  # 读取的类型可能与数据库类型不一致
    # field = my_sql_fun(sql_field, method='read').columns

    if db == 'gp':
        sql = sql_format(f"""
        select column_name col_name, data_type 
        from information_schema.columns
        where table_schema||'.'||table_name = '{tablename}'
        """)
        field = my_sql_fun(sql, method='read', index_col='col_name').data_type
    elif db == 'hive':
        sql = f'desc {tablename}'
        field = my_sql_fun(sql, method='read', index_col='col_name').data_type
        # 删除分区字段
        par_index = (field.index == '# Partition Information').argmax()
        if par_index > 0:
            field = field.loc[~field.index.isin(field.index[par_index:])]
    field.index = field.index.str.lower()
    return field


def get_month_count(tablename, col_month,day_next=None, start=None, end=None,day_datetye=2):
    """
    统计数据表[各账期]数据量
    :param tablename: 表名
    :param col_month: 账期字段
    :param start: 起始账期（包含）
    :param end: 结束账期（包含）
    :return: 各账期数据量
    """
    if day_datetye==2:
        con = ([] if start is None else [f"cast({col_month} as {type_py_sql[str]}) >= '{start}'"]) + \
              ([] if end is None else [f"cast({col_month} as {type_py_sql[str]}) <= '{end}'"])
        where = '' if con == [] else f"where {' and '.join(con)}"
        sql_mlist = f'select cast({col_month} as {type_py_sql[str]}) {col_month}, count(1) from {tablename} {where} group by {col_month}'
    elif day_datetye==1:
        con = ([f" cast({col_month[0]} as {type_py_sql[str]})='{day_next[:6]}'"] if start is None else [f"cast({col_month[0]} as {type_py_sql[str]})='{day_next[:6]}' and cast({col_month[1]} as {type_py_sql[str]}) >= '{start[-2:]}'"]) + \
              ([f"cast({col_month[1]} as {type_py_sql[str]}) <= '{day_next[-2:]}'"] if end is None else [f"cast({col_month[1]} as {type_py_sql[str]}) <= '{end[-2:]}'"])
        where = '' if con == [] else f"where {' and '.join(con)}"
        sql_mlist = f'select cast({col_month[1]} as {type_py_sql[str]}) {col_month[1]}, count(1) from {tablename} {where} group by {col_month[1]}'
        col_month=col_month[1]
        print(col_month)
    print(sql_mlist)
    mlist = my_sql_fun(sql_mlist, method='read', index_col=col_month).sort_values(by=col_month)
    mlist.index = mlist.index.astype(str)
    return mlist


def sql_format(x):
    """
    取消sql长字符串的缩进，使print输出更美观（识别首行缩进，sql的开端必须在三个引号后另起一行）
    :param x: sql长字符串
    :return: 调整后的sql
    """
    indent = re.findall('^ *\n *', x)
    if len(indent) == 1:
        indent = indent[0]
        indent = re.sub('.*\n', '\n', indent)
        x_new = x.replace(indent, '\n')
    else:
        x_new = x

    x_new = re.sub('\n *\n', '\n', x_new)
    x_new = re.sub('\n$', '', x_new)
    x_new = re.sub('\n*;$', '', x_new)  # 内蒙古电信的某些环境python执行的sql结尾有分号易报错
    return x_new


def sql_show(text, sql, db=db):
    """
    打印sql
    :param text: 提示字符串
    :param sql: sql语句字符串
    :param db: 数据库名称，如gp、hive等
    :return:
    """
    if db == 'gp':
        sho = re.sub(' *$', '', f'{text}{sql}')   # conn.cursor().execute执行时不会自动打印sql
    elif db == 'hive':
        # sho = re.sub(' *$', '', text)           # 有的环境：spark_session.sql执行时自动打印sql
        sho = re.sub(' *$', '', f'{text}{sql}')  # 有的环境：spark_session.sql执行时不打印sql
    print(sho)


def reduce_mem_usage(data):
    """
    精简DaFrame字段类型，减小其内存占用
    :param data: DataFrame
    :return: 修改过字段类型的DataFrame
    # 转换类型需要小心：
    Series([0, -999, -999],dtype=np.float16).std()  # inf
    Series([0, -999, -999],dtype=np.float32).std()  # 576.77294921875

    np.float16(576.772919)  # 577.0
    np.float16(576.625)     # 576.5

    np.float32(576.772919)  # 576.77295
    np.float32(576.625)     # 576.625
        """
    # 计算当前内存
    start_mem_usg = data.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"处理前内存 :{round(start_mem_usg, 5)}MB")

    # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
    NAlist = []
    for col in data.columns:
        # 这里只过滤了objectd格式，如果你的代码中还包含其他类型，请一并过滤
        if data.dtypes.astype(str)[col] not in ('object', 'datetime64[ns]'):
            # 判断是否是int类型
            isInt = False
            mmax = data[col].max()
            mmin = data[col].min()

            # # Integer does not support NA, therefore Na needs to be filled 修改！！！！
            # if not np.isfinite(data[col]).all():
            #     NAlist.append(col)
            #     data[col].fillna(-999, inplace=True)  # 用-999填充

            # test if column can be converted to an integer
            asint = data[col].fillna(0).astype(np.int64)
            result = np.fabs(data[col] - asint)
            result = result.sum()
            if result < 0.01:  # 绝对误差和小于0.01认为可以转换的，要根据task修改
                isInt = True

            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0:  # 最小值大于0，转换成无符号整型
                    if mmax <= np.iinfo(np.uint8).max:
                        data[col] = data[col].astype(np.uint8)
                    elif mmax <= np.iinfo(np.uint16).max:
                        data[col] = data[col].astype(np.uint16)
                    elif mmax <= np.iinfo(np.uint32).max:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.uint64)
                else:  # 转换成有符号整型
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)
            else:  # 注意：这里对于float都转换成float16，需要根据你的情况自己更改
                if mmin > np.finfo(np.float16).min and mmax < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif mmin > np.finfo(np.float32).min and mmax < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    mem_usg = data.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"处理后内存 :{round(mem_usg, 5)}MB")
    print(f"This is {100 * mem_usg / start_mem_usg}% of the initial size")
    return data


# ------------------------------ <editor-fold desc="数据导入函数"> -----------------------------------------------------
def read_data(name, src=db, condition=None, col_need=None, col_del=None, col_char=None, col_num=None, col_date=None,
              nrows=None, if_coltolower=True, **kwargs):
    """
    数据导入函数（保持各环境下该接口的一致性）
    将src的默认值设置为本地环境设置
    :param name: src='file'时：文件名（带路径）； src='yaxin'时：数据源名称； src='某数据库'时：数据库表名
    :param src: 数据来源，取值为 'file'、'某数据库（hive、pg...）'等，根据需要扩充
    :param condition: 用户范围筛选条件,可以取值为‘limit n’，用于查看前几条数据
    :param col_need: 需要读取的字段名称列表（list）,不可多填
    :param col_del:  需要在导入前过滤掉的字段名称列表（list）,可多填，只处理数据中包括的列
    :param col_char: 需限定为类别型的字段列表，可多填，只处理数据中包括的列
    :param col_num: 需限定为数值型的字段列表，可多填，只处理数据中包括的列(整型无需转换为float)
    :param col_date: 需限定为日期型的字段列表，可多填，只处理数据中包括的列
    :param nrows: 读取nrows行，若取值为None，则全量读取
    :param if_coltolower: 是否将读取的DataFrame的字段名转换为小写，True：转换为小写，False：不转换，保持原字段名不动
    :param kwargs: 关键字参数
    :return: 读取的数据(DataFrame)
    """
    print(f"-- 读取数据: {name} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} --".replace('--', '-'*25))
    def my_len(x):
        if x is None:
            return ''
        else:
            return f"({len(x)})"
    print(f"    src: {src}")
    print(f"    condition: {condition}")
    print(f"    col_need{my_len(col_need)}: {col_need}")
    print(f"    col_del{my_len(col_del)}: {col_del}")
    print(f"    col_char{my_len(col_char)}: {col_char}")
    print(f"    col_num{my_len(col_num)}: {col_num}")
    print(f"    col_date{my_len(col_date)}: {col_date}")
    print(f"    nrows: {nrows}")
    print(f"    if_coltolower: {if_coltolower}")
    print(f"    kwargs: {kwargs}\n")

    # ------------------------------- 前置条件准备 -------------------------------
    if (src == 'file'):
        if condition is not None:
            print('    src参数取值为"file"时，不支持直接筛选，数据全量读取后再按condition筛选！')
        all_cols = pd.read_csv(name, nrows=0, **kwargs).columns
    if src == 'gp':
        if not if_coltolower:
            print('    当src取值为gp时，读取数据的字段名均为小写（即使数据库中字段名为大写）')
        all_cols = my_sql_fun(f'select * from {name} limit 0', method='read', db=src).columns  # 该方式可兼容多种数据库

    # ------------------------------- 确定读取字段 -------------------------------
    if (col_need is not None) & (col_del is not None):
        raise Exception('参数col_need与col_del不可以同时设置！')
    elif (col_need is None) & (col_del is None):  # col_need、col_del都没特别设置，则读取全部字段
        cols = None
    elif col_need is not None:
        cols = list(col_need)
    elif col_del is not None:  # 全部字段 剔除 col_del 为读取字段
        cols = [i for i in all_cols if i not in col_del]

    # 处理col_char、col_num、col_date，并检验字段名称是否有重复
    to_dict = lambda x, y: {} if x is None else {i: y for i in x if i in (cols if cols else all_cols)}
    dict_char = to_dict(col_char, str)
    dict_num = to_dict(col_num, float)
    dict_date = to_dict(col_date, '')
    type_dict = dict(dict_char, **dict_num)
    if (len(dict_char) + len(dict_num) + len(dict_date)) != len(dict(type_dict, **dict_date)):
        raise Exception('参数col_char、col_num、col_date之间存在重复字段！')

    # ------------------------------- 读取数据 -------------------------------
    print('    读取')
    if src == 'file':
        data = pd.read_csv(name, usecols=cols, dtype=type_dict, parse_dates=col_date, nrows=nrows, **kwargs)
        if condition is not None:
            print(f'    shape: {data.shape}\n')
            data = sqldf(f"select * from data where {condition}")
            print(f'   按condition筛选后{data.shape}: {condition}')
    elif src in ['gp', 'hive']:
        if dict_char:
            cols_sql = ', '.join([(f"cast({i} as {type_py_sql[str]}) {i}" if i in dict_char.keys() else i) for i in (cols if cols else all_cols)])
        else:
            cols_sql = '*' if cols is None else ', '.join(cols)

        con_sql = ('' if condition is None else f' where {condition}').replace(' and', '\n and')
        con_nrows = '' if nrows is None else f' limit {nrows}'
        sql = f"select {cols_sql} \nfrom {name}\n{con_sql}\n{con_nrows}"
        data = my_sql_fun(sql, method='read', db=src, parse_dates=col_date, **kwargs)
        print(f'    shape: {data.shape}\n')
    else:
        raise Exception(f'src参数取值{src}，未实现，请补充！')
    if if_coltolower:
        data.columns = data.columns.str.lower()

    # 字段类型纠正
    for c, t in type_dict.items():
        dtype_actual = data.dtypes.astype(str).str.replace('64.*?$|32.*?$|16.*?$|8.*?$', '')[c]
        dtype_expect = {str: 'object', float: 'float'}[t]
        if (dtype_actual != dtype_expect) & (not ((dtype_actual == 'int') & (dtype_expect == 'float'))):  # 整型无需转换为float
            t_name = str(t).replace('class', '').strip("<|>|'| '")
            print(f"将{c}字段类型({data.dtypes[c]}): .astype({t_name})")
            isna = data[c].isnull()
            data[c] = data[c].astype(t)
            data.loc[isna, c] = np.nan
    if dict_date:
        d = data[dict_date.keys()].dtypes.astype(str)
        d_error = d.loc[d != 'datetime64[ns]']
        if len(d_error) > 0:
            line = f"    日期型字段转换 {'-' * 25}"
            print(line)
            print('    下列字段并未读取为日期型，读取后转换为日期型')
            print(f'    转换前的字段类型:{add_printindent(d_error)}\n')
            compare = DataFrame()
            for i in d_error.index:
                print(f'    转换: {i}')
                data_i = data[i].copy()

                # --- 临时措施 start
                con_more = data[i] > '22620411'
                if con_more.sum():
                    s = f'{i} 字段存在{con_more.sum()}个取值过大，被赋值为22620401'
                    warnings.warn(s); time.sleep(seconds)
                    data.loc[con_more, i] = '22620401'
                if set(data[i].value_counts().index.map(len)) == {6}:
                    s = f'{i} 字段长度为6（yyyymm），统一添加01（dd）'
                    warnings.warn(s); time.sleep(seconds)
                    data.loc[data[i].notnull(), i] = data.loc[data[i].notnull(), i] + '01'
                # --- 临时措施 end

                data[i] = pd.to_datetime(data[i])
                com = pd.concat([data_i, data[i]], axis=1)
                com = com.drop_duplicates()
                com.columns = ['转换前', '转换后']
                com = com if len(com) < 3 else com.head(3)
                com.index = [[i] * len(com), [''] * len(com)]
                compare = pd.concat([compare, com])
            print('\n', end='')
            print(f'    转换后的字段类型：{add_printindent(data[d_error.index].dtypes)}\n')
            print(f"    转换前后取值对比：\n{add_printindent(compare)}\n")
            print(f"    转换后数据概览：\n{add_printindent(data[d_error.index].head(3))}")
            print(line.replace('转换', '转换完毕'))
    print(f"--读取完毕: {data.shape} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} --\n".replace('--', '-'*25))
    # data = reduce_mem_usage(data) # 慎用，精度会出现问题
    return data
# </editor-fold> -------------------------------------------------------------------------------------------------------


# ------------------------------------ <editor-fold desc="数据账期校验"> ------------------------------------------------
def month_check(data, col_month, month_value):
    """
    数据账期校验
    主要针对亚信平台，以防配置数据源的人为失误,可直连数据库的环境中可不用此函数
    不严格区分 yyyymm 与 'yyyymm'
    :param data: 待检验的数据（DataFrame）
    :param col_month: 数据账期字段名称(str)
    :param month_value: 正确取值（str，'yyyymm'）
    :return: None， 直接将检验结果打印在屏幕，如果账期与预期不符则报错跳出，以免基于错误数据输出结果而无察觉
    """

    if col_month not in data.columns:
        print(f'数据中不包括 {col_month} 字段，无法完成检验！')
    else:
        col_month_unique = list(data[col_month].unique())
        if len(col_month_unique) != 1:
            raise Exception(f'{col_month}字段应取唯一值，但实际：{col_month_unique}')
        else:
            if str(col_month_unique[0]) != str(month_value):  # 兼容 取值类型不一致的情况（字符、数值）
                raise Exception(f'{col_month}字段取值应为 {month_value}，但实际：{col_month_unique[0]}')
        print('    通过')
# </editor-fold> -------------------------------------------------------------------------------------------------------


# --------------------------------------- <editor-fold desc="数据导出"> -------------------------------------------------
def save_data(data, destination, to='file', if_exists=None,  **kwargs):
    """
    数据导出函数（保持各环境下该接口的一致性）
    将src的默认值设置为本地环境设置
    :param data: 待导出的数据(DataFrame)
    :param destination: to='file'时：文件名（带路径）； to='某数据库'：数据库表名
    :param to: 数据的导出地，取值为 'file'、'某数据库（hive、pg...）'，分别代表 文件、某数据库
    :param f_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    index : boolean, default True
    :param kwargs: 关键字参数
    :return: None，直接将数据导出，无需返回结果
    """
    print(f"保存结果数据至：{destination}")

    if if_exists is None:
        if_exists = {'file': 'replace', 'gp': 'fail', 'hive': 'fail'}[to]
        print(f"to={to}时，if_exists默认为'{if_exists}'")
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    if to == 'file':
        data.to_csv(destination, sep=',', index=False, **kwargs)
        if if_exists != 'replace':
            raise Exception('save_data 待填充！')  # 需要时再填充
    elif to in['gp', 'hive']:
        sql = f'insert into {destination} select*from data'
        scope = {'data': data}  # key 与 sql中取值保持一致
        my_sql_fun(sql, method='todb', db=db, scope=scope, if_exists=if_exists)

        # # 新增数据库需要确保：
        # to = 'hive';
        # destination = 'cq_pro1_role2.tmp_xj_20211129'
        # from pandas import DataFrame
        # data = DataFrame([
        #     [1, 2, 3],
        #     [3, 4, 5]], columns=['id', 'col1', 'col2'])
        # data
        #
        # save_data(data, destination, to=to, if_exists='append')                  # 首次导入：无表则创建表
        # save_data(data.iloc[:, [0, 1]], destination, to=to, if_exists='append')  # 再次导入部分列：插入到对应列，缺失列为空值
        # save_data(data.iloc[:, [0, 2]], destination, to=to, if_exists='append')  # 再次导入部分列：插入到对应列，缺失列为空值
        # save_data(data.rename(columns={'col2': 'col3'}), destination, to=to, if_exists='append')  # 再次导入新增列：列名不对应，插入报错
# </editor-fold> -------------------------------------------------------------------------------------------------------


# ------------------ <editor-fold desc="导入sqldf，或创建sqldf（若本地建模环境未安装）"> -----------------------------------
try:
    from pandasql import sqldf
except:
    print('创建sqldf函数')
    import inspect
    from contextlib import contextmanager
    from pandas.io.sql import to_sql, read_sql
    from sqlalchemy import create_engine
    import re
    from warnings import catch_warnings, filterwarnings
    from sqlalchemy.exc import DatabaseError, ResourceClosedError
    from sqlalchemy.pool import NullPool

    class PandaSQLException(Exception):
        pass


    class PandaSQL:
        def __init__(self, db_uri='sqlite:///:memory:', persist=False):
            """
            Initialize with a specific database.

            :param db_uri: SQLAlchemy-compatible database URI.
            :param persist: keep tables in database between different calls on the same object of this class.
            """
            self.engine = create_engine(db_uri, poolclass=NullPool)
            if self.engine.name not in ('sqlite', 'postgresql'):
                raise PandaSQLException('Currently only sqlite and postgresql are supported.')

            self.persist = persist
            self.loaded_tables = set()
            if self.persist:
                self._conn = self.engine.connect()
                self._init_connection(self._conn)

        def __call__(self, query, env=None):
            """
            Execute the SQL query.
            Automatically creates tables mentioned in the query from dataframes before executing.

            :param query: SQL query string, which can reference pandas dataframes as SQL tables.
            :param env: Variables environment - a dict mapping table names to pandas dataframes.
            If not specified use local and global variables of the caller.
            :return: Pandas dataframe with the result of the SQL query.
            """
            if env is None:
                env = get_outer_frame_variables()

            with self.conn as conn:
                for table_name in extract_table_names(query):
                    if table_name not in env:
                        # don't raise error because the table may be already in the database
                        continue
                    if self.persist and table_name in self.loaded_tables:
                        # table was loaded before using the same instance, don't do it again
                        continue
                    self.loaded_tables.add(table_name)
                    write_table(env[table_name], table_name, conn)

                try:
                    result = read_sql(query, conn)
                except DatabaseError as ex:
                    raise PandaSQLException(ex)
                except ResourceClosedError:
                    # query returns nothing
                    result = None

            return result

        @property
        @contextmanager
        def conn(self):
            if self.persist:
                # the connection is created in __init__, so just return it
                yield self._conn
                # no cleanup needed
            else:
                # create the connection
                conn = self.engine.connect()
                conn.text_factory = str
                self._init_connection(conn)
                try:
                    yield conn
                finally:
                    # cleanup - close connection on exit
                    conn.close()

        def _init_connection(self, conn):
            if self.engine.name == 'postgresql':
                conn.execute('set search_path to pg_temp')


    def get_outer_frame_variables():
        """ Get a dict of local and global variables of the first outer frame from another file. """
        cur_filename = inspect.getframeinfo(inspect.currentframe()).filename
        outer_frame = next(f
                           for f in inspect.getouterframes(inspect.currentframe())
                           if f.filename != cur_filename)
        variables = {}
        variables.update(outer_frame.frame.f_globals)
        variables.update(outer_frame.frame.f_locals)
        return variables


    def extract_table_names(query):
        """ Extract table names from an SQL query. """
        # a good old fashioned regex. turns out this worked better than actually parsing the code
        tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
        tables = [tbl
                  for block in tables_blocks
                  for tbl in re.findall(r'\w+', block)]
        return set(tables)


    def write_table(df, tablename, conn):
        """ Write a dataframe to the database. """
        with catch_warnings():
            filterwarnings('ignore',
                           message='The provided table name \'%s\' is not found exactly as such in the database' % tablename)
            to_sql(df, name=tablename, con=conn,
                   index=not any(name is None for name in df.index.names))  # load index into db if all levels are named


    def sqldf(query, env=None, db_uri='sqlite:///:memory:'):
        """
        Query pandas data frames using sql syntax
        This function is meant for backward compatibility only. New users are encouraged to use the PandaSQL class.

        Parameters
        ----------
        query: string
            a sql query using DataFrames as tables
        env: locals() or globals()
            variable environment; locals() or globals() in your function
            allows sqldf to access the variables in your python environment
        db_uri: string
            SQLAlchemy-compatible database URI

        Returns
        -------
        result: DataFrame
            returns a DataFrame with your query's result

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
            "x": range(100),
            "y": range(100)
        })
        >>> from pandasql import sqldf
        >>> sqldf("select * from df;", globals())
        >>> sqldf("select * from df;", locals())
        >>> sqldf("select avg(x) from df;", locals())
        """
        return PandaSQL(db_uri)(query, env)
# </editor-fold> -------------------------------------------------------------------------------------------------------
