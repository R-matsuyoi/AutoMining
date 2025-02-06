import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings
import time
import re

seconds =3

def string_to_table(s, line_break='\n', sep='\t', header=True, col_type=None, na_values='', dtype=None):
    """
    将字符串转化为DataFrame，通常用于将excel表格中的内容作为一个大字符串赋值给s，然后转成DataFrame
    用于不方便上传文件至工作环境中的项目
    :param s: 字符串
    :param line_break: 行分隔符
    :param sep: 列分隔符
    :param header: 数据是否包含表头
    :param col_type: 字段类型列的字段名，若不为None，则将根据该列取值将字段类型归类为类型型、数值型、日期型
    :param na_values: 将na_values的取值转化为缺失值NAN,若na_values为None，则什么都不做
    :param dtype: 规定字段类型的字典，键为字段名，值为字段类型
    :return: DataFrame
    """
    rows = s.split(line_break)
    if rows[0] == '':
        rows = rows[1:]
    if rows[-1] == '':
        rows = rows[:-1]

    table = DataFrame()
    for r in rows:
        if r:
            r_split = r.split(sep)
            table = pd.concat([table, DataFrame(r_split).T])

    if header:
        table.columns = table.iloc[0, ].values
        table = table.iloc[1:, ]
        table = table.copy()
    table.index = range(len(table))

    table = table.loc[:, table.columns.notnull()]

    if col_type is not None:
        if col_type not in table.columns:
            raise Exception("不存在字段：%s ！" % col_type)

        if_num = table[col_type].str.lower().str.contains('numeric|int|double|float|decimal')  # 按本地实际情况扩充
        if_char = table[col_type].str.lower().str.contains('character|text|varchar|string')    # 按本地实际情况扩充
        if_date = table[col_type].str.lower().str.contains('date|datetime|timestamp')          # 按本地实际情况扩充
        table.loc[if_num, 'dtype_classify'] = '数值型'
        table.loc[if_char, 'dtype_classify'] = '类别型'
        table.loc[if_date, 'dtype_classify'] = '日期型'
        not_classified = table.loc[table.dtype_classify.isnull(), :]

        if len(not_classified) > 0:
            raise Exception('有%d个字段的字段类型未归类！：\n%s' % (len(not_classified), not_classified))

    if na_values is not None:
        table[table == na_values] = np.float64('nan')
    if dtype is not None:
        d = {str: 'object', float: 'float'}
        type_more = set(dtype.values()) - set(d.keys())
        if type_more:
            warnings.warn('dtype参数中的下列类型暂未实现，将忽略，需要修改string_to_table函数：%s' % type_more); time.sleep(seconds)
            dtype = {k: v for k, v in dtype.items() if v in d.keys()}
        for c, t in dtype.items():
            if table.dtypes.astype(str)[c] != d[t]:
                table[c] = table[c].astype(t)
    return table


def strtotable_valuefun(x):
    """
    从excel复制的字符串转换为DataFrame后，将每列取值正确转换为：字符型、浮点型、整型
    :param x: Series
    :return:
    strtotable_valuefun(Series('1'))      # dtype: int32
    strtotable_valuefun(Series("'1'"))    # dtype: object
    strtotable_valuefun(Series('1.0'))    # dtype: object
    strtotable_valuefun(Series("'1.0'"))  # dtype: float64
    """
    # 纠正字符串转为DataFrame后的取值
    x = x.copy()
    x_delquot = x.str.strip("'|\"")
    if_quot = x_delquot.apply(len) != x.apply(len)
    if_digit = (~if_quot) & x.str.replace('.', '').str.isdigit()
    if_float = if_digit & x.str.contains('\.')
    if_int = if_digit & (~if_float)
    x.loc[if_quot] = x_delquot.loc[if_quot]
    x.loc[if_float] = x.loc[if_float].astype('float').copy()
    x.loc[if_int] = x.loc[if_int].astype('int').copy()
    return x


def infos_to_table(s_info, col_dealvalue=None, col_eval=None, col_lower=None, col_index='model_name', col_char=None,
                   del_pound=True, del_model=None):
    """
    将excel粘贴的长字符串转为DataFrame
    长字符串：由excel复制而来，其中的空单元格将转换为'', 缺失值取值需要在单元格中明确的写成np.nan
    若不想要长字符串中某些模型的信息，可令del_pound=True，且在相应的模型名称开头添加#
    :param s_info: 长字符串
    :param col_dealvalue: 需要 转换取值的字段名列表 'ALL'或指定字段名列表
    :param col_eval: 向这些列的每个取值应用eval函数
    :param col_lower: 将取值转换为小写的字段列表
    :param col_index: 将该列取值设置为索引
    :param col_char: 将这些列转换为类别型
    :param del_pound: 是否删除以#开头的模型名称的模型信息
    :return:
    """
    print('将excel粘贴的长字符串转为DataFrame')
    infos = string_to_table(s_info, na_values=None)
    infos = infos.apply(lambda x: x.str.strip('|  '))
    infos = infos.apply(lambda x: x.where(x != 'None', None))
    for c in ([] if col_dealvalue is None else (infos.columns if col_dealvalue == 'ALL' else col_dealvalue)):
        infos[c] = strtotable_valuefun(infos[c])
    if col_lower is not None:
        infos[col_lower] = infos[col_lower].apply(lambda x: x.str.lower())
    if col_char:
        for i in col_char:
            infos[i] = infos[i].astype(str)
    infos = infos.apply(lambda x: x.where(~x.isin(['np.nan', 'nan', 'NaN']), np.nan))
    infos.columns = infos.columns.str.strip(' ')
    if col_eval is not None:
        for i in col_eval:
            print(f'    eval: {i}')
            # 如果不是缺失值或数个空格，则eval
            infos[i] = infos[i].apply(lambda x: x if (str(x) == 'nan') | (re.sub('^ *$', '', str(x)) == '') else eval(x))

    if 'model_name' in infos.columns:
        con_pound = infos.model_name.str.startswith('#')
        if con_pound.sum():
            if del_pound:
                print(f'删除model_name以#开头的模型信息：{list(infos[con_pound].model_name)}\n')
                infos = infos.loc[~con_pound]
            else:
                print('保留model_name以#开头的模型信息，并替换掉#字符\n')
                infos.model_name = infos.model_name.str.replace('^# *', '')
    if col_index is not None:
        infos.index = infos[col_index].values
    if del_model:
        con_name = infos.model_name.isin(del_model)
        if con_name.sum():
            print(f'删除下列模型的信息：{list(infos[con_name].model_name)}\n')
            infos = infos.loc[~con_name]
    return infos