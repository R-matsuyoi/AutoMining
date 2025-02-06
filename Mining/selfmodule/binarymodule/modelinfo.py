
# field_comment_fun函数、addcol_cover函数可能需要手动修改：不同模型处理不同

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import warnings
import re

try:
    from sklearn.externals import joblib
except:
    import joblib

from Mining.selfmodule.tablemodule.basestr import *
from Mining.selfmodule.toolmodule.strtotable import infos_to_table
from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, get_onlyvalue
from Mining.selfmodule.tablemodule.tablefun import field_base_fun, month_mark


def add_preinfos(infos):
    """
    补充预测所需信息
    :param Info: 命名元组，单个模型信息
    :param month_predict: 预测账期，yyyymm
    :return:
    """
    infos = infos.copy()
    infos['month_predict'] = [j(i) for i, j in zip(infos['month_predict'], infos['month_train'].apply(type))]  # 同步取值类型
    mark_tabexp = (infos.month_tabexp.astype(str) + 'exp~').where(infos.month_tabexp.astype(str) != infos.month_train.astype(str), '')
    model_wd = infos.model_wd_traintest.str.replace('traintest~.*$', '')
    infos['model_wd_predict'] = model_wd + '/predictscore~' + infos.month_predict.astype(str)
    infos['table_predict'] = infos.model_wd_predict + '/predict_data~' + mark_tabexp + infos.month_predict.astype(str) + '.csv'
    infos['table_score'] = infos.model_wd_predict + '/predict_score_data~' + mark_tabexp + infos.month_train.astype(str) + '~' + infos.month_test.astype(str) + '~' + infos.month_predict.astype(str) + '.csv'

    # 账期类型检验
    er = infos.loc[infos.month_train.apply(type) != infos.month_predict.apply(type), ['month_train', 'month_predict']]
    if len(er) > 0:
        er = er.apply(lambda x: x.apply(type))
        raise Exception(f"下列模型的month_predict取值类型与month_train不同，请纠正：\n{er}")
    return infos


def type_exam(x, dtype_classify, mark=''):
    """
    检查字段取值类型与field_base中对应类型是否一致
    :param x: 字段的具体取值
    :param dtype_classify: field_base中字段类型分类：'类别型', '数值型
    :return: 返回报错信息字符
    """
    mark2 = mark if mark else '参数'
    if (dtype_classify == '类别型') & (not isinstance(x, str)):
        return f"{mark2}取值类型为数值型（{type(x).__name__}）, field_base中对应为类别型，请统一！"
    elif (dtype_classify == '数值型') & (isinstance(x, str)):
        return f"{mark2}取值类型为类别型（{type(x).__name__}），field_base中对应为数值型，请统一！"
    elif dtype_classify not in ('类别型', '数值型'):
        return f"dtype_classify应在('类别型', '数值型')中, 或扩充此处代码"


def privy_modelsummary(info_changed, if_coltolower=True):
    """
    模型信息汇总函数，记录不同模型之间的差异点，作为复合参数使用
    :param info_changed: 训练、测试、打分的数据账期等动态变动参数的DataFrame
    :param if_coltolower: 列名是否转小写
    :return: 模型信息汇总（DataFrame）
    备注：table_train、table_test、table_predict、table_score的设置：
          读取文件: 文件名（带路径）
          连数据库: 数据库表名
    """
    def to_lower(x):  # 字段名统一为小写（与数据传输函数保持一致）
        if x is None:
            res = None
        if str(x) == 'nan':
            return np.nan
        elif isinstance(x, list):
            res = [i.lower() for i in x]  # lower
        elif isinstance(x, str):
            res = x.lower()
        return res

    col_dealvalue = ['Pcase', 'Ncase', 'target_lag']
    col_lower = ['col_month', 'col_id', 'col_target', 'col_stratified', 'col_out']
    col_eval = ['col_out', 'dict_sortscore']
    # 为避免类型不一致导致错误，在账期字段、目标字段出现的所有地方，将其统一强转类别型（数据库中、模型信息中、读取到python中）
    infos_quiet = infos_to_table(s_info_quiet, col_dealvalue, col_eval, col_lower=col_lower)
    infos = pd.merge(info_changed, infos_quiet, on='model_name', how='left')  # 根据参数info_changed的index进一步限定指定模型
    infos.index = infos.model_name.values

    # 检查Pcase、Ncase取值类型是否一致
    PN_diff = infos[~(infos.Pcase.apply(type) == infos.Ncase.apply(type))].index
    if len(PN_diff) > 0:
        raise Exception(f"下列模型的Pcase、Ncase取值类型不同，请纠正：{PN_diff}")

    print("\n\n转换、检查、纠正field_base")
    type_error = []
    for i in infos.s_field_base.unique():
        print(f"\n- {i} -".replace('-', '-'*60))
        field_base = field_base_fun(i)
        field_base_array = np.array(field_base)

        # 检查你是否选择了正确数量的行（93行）
        if field_base_array.shape[0] == (infos.s_field_base == i).sum():
            infos.loc[infos.s_field_base == i, 'field_base'] = field_base_array
        else:
            # 如果形状不匹配，则需要处理，确保两者形状一致
            print(
                f"Shape mismatch: field_base_array shape is {field_base_array.shape} but expected {(infos.s_field_base == i).sum()}")
        # 检查字段类型是否矛盾
        col_month = get_onlyvalue(infos.col_month)
        type_col_month = field_base.loc[field_base.field_name == col_month, 'dtype_classify'].iloc[0]
        for j in infos.loc[infos.s_field_base == i, ].index:
            print(j)
            if str(infos.loc[j, "month_train"]) != 'nan':
                train_er = type_exam(infos.loc[j, "month_train"], type_col_month, mark=f'模型信息中 {j} 的 month_train ')
                if train_er:
                    type_error.append(train_er)
            if str(infos.loc[j, "month_test"]) != 'nan':
                test_er = type_exam(infos.loc[j, "month_test"], type_col_month, mark=f'模型信息中 {j} 的 month_test ')
                if test_er:
                    type_error.append(test_er)
            col_target = infos.loc[j, 'col_target']
            type_col_target = field_base.loc[field_base.field_name == col_target, 'dtype_classify'].iloc[0]

            Pcase_er = type_exam(infos.loc[j, "Pcase"], type_col_target, mark=f'模型信息中 {j} 的 Pcase ')
            if Pcase_er:
                type_error.append(Pcase_er)
            Ncase_er = type_exam(infos.loc[j, "Ncase"], type_col_target, mark=f'模型信息中 {j} 的 Ncase ')
            if Ncase_er:
                type_error.append(Ncase_er)
        if type_error:
            type_error = '\n'.join(type_error)
            raise Exception(f"\n{type_error}")
    print('-' * 120, '\n')

    month_train_str = infos.month_train.astype(str)
    month_test_str = infos.month_test.astype(str)

    # 用户群基本限制条件
    infos.condition_base = infos.condition_base.apply(lambda x: condition_base_dict[x] if str(x) != 'nan' else np.nan)

    # 模型的最终限制条件(基本+模型特定)
    infos['condition'] = (infos.condition_add.fillna('') + ' and ' + infos.condition_base.fillna('')).str.strip(' ').str.replace('^and|and$', '').str.strip(' ')

    # 宽表探索账期,若未特别指定宽表探索账期，则默认等于每期更新时的训练账期
    if 'month_tabexp' not in infos.columns:
        infos['month_tabexp'] = infos.month_train

    # 训练结果路径
    mark_traintest = Series(infos.index, index=infos.index).apply(lambda x: month_mark(to_namedtuple(infos.loc[x]), 'traintest'))
    infos['model_wd_traintest'] = './binaryclassify/' + infos.short_name + '/traintest' + mark_traintest

    # 建模宽表数据字典对应文件名
    infos['f_field_comment'] = infos.model_wd_traintest + '/tabexp_col_obj~' + infos.month_tabexp.astype(str) + '.pkl'

    # 对于训练模型无用的字段名称列表（直接不读取），可多填，只处理数据中包括的列
    # 包括：T+1账期字段（其他模型的目标字段）、前期无数据字段、无用字段等
    # 作为 field_base中into_model列的补充，主要针对field_base不包括的字段
    coldel = list(infos.col_target.unique()) + ['user_acct_month', 'rn']  # 手动填充

    # 剔除目标模型的目标字段
    infos['col_del'] = infos.index.map(lambda x: [i for i in coldel if i != infos.col_target[x]])

    # 训练数据（可能含时间内验证集data_timein）、测试数据（时间外data_timeout）、预测数据、分数数据的文件名/数据库表名；
    #     若为方便传输数据将table_test数据合并至训练集table_train中、或未设置data_timeout数据集，则对应的table_test为NaN
    #     训练、测试、预测数据的账期标识后缀：[~宽表探索账期]~数据所属账期,若如果宽表探索账期=训练账期，则省略"~宽表探索账期"
    #     分数数据账期标识后缀： [~宽表探索账期]~训练账期~测试账期~预测账期
    mark_tabexp = (infos.month_tabexp.astype(str) + 'exp~').where(infos.month_tabexp.astype(str) != month_train_str, '')
    month_test_strna = month_test_str.where(infos.month_test.notnull(), np.nan)
    infos['table_train'] = infos.model_wd_traintest + '/train_data~' + mark_tabexp + month_train_str + '.csv'
    infos['table_test'] = infos.model_wd_traintest + '/test_data~' + mark_tabexp + month_test_strna + '.csv'

    # 分数排序字典: 将字符串转换为字典，并且将键对应d字段名转换为小写
    infos.dict_sortscore = infos.dict_sortscore.apply(lambda x: {k.lower(): v for k, v in x.items()} if str(x) != 'nan' else x)

    # base_data_fun、traintest_fun等函数内变量的默认值（若各模型取值不同，根据需在info_changed中设置）
    if 'timein_count' not in infos.columns:   # 时间内验证集的样本量，若为NaN，则不设置验证集
        print('infos中不包括timein_count， 统一赋值为默认值：np.nan')
        infos['timein_count'] = np.nan
    if 'traintable_ratio' not in infos.columns:     # 训练数据集正负例样本比例，即负例样本数 = 正例样本数 * traintable_ratio
        print('infos中不包括traintable_ratio， 统一赋值为默认值：1')
        infos['traintable_ratio'] = 1
    if 'Pcase_limit' not in infos.columns:          # 正例样本上限，实际正例量超出则随机抽样
        print('infos中不包括Pcase_limit， 统一赋值为默认值：10000')
        infos['Pcase_limit'] = 10000
    if 'timeout_limit' not in infos.columns:        # 时间外测试样本的上限，实际样本量超出则随机抽样
        print('infos中不包括timeout_limit， 统一赋值为默认值：100000')
        infos['timeout_limit'] = 100000
    if 'Pcumsum_limit' not in infos.columns:        # 向前累计正例样本的账期数
        print('infos中不包括Pcumsum_limit， 统一赋值为默认值：np.nan')
        infos['Pcumsum_limit'] = np.nan
    if 'trainproc_ratiolist' not in infos.columns:  # 训练过程遍历的正负例抽样比例列表， 即负例样本数 = 正例样本数 * 该列表元素
        print("infos中不包括trainproc_ratiolist， 统一赋值为默认值：['actual']")
        infos['trainproc_ratiolist'] = [['actual']] * len(infos)
    if 'freq_limit' not in infos.columns:           # 数据预处理时字段取值的集中度阈值
        print('infos中不包括freq_limit， 统一赋值为默认值：0.95')
        infos['freq_limit'] = 0.95
    if 'unique_limit' not in infos.columns:         # 数据预处理时类别字段的类别个数上限
        print('infos中不包括unique_limit， 统一赋值为默认值：5000')
        infos['unique_limit'] = 5000
    if 'valuecount_limit' not in infos.columns:    # 数据预处理时类别字段的类别取值样本数下限
        print('infos中不包括valuecount_limit， 统一赋值为默认值：50')
        infos['valuecount_limit'] = 50
    if 'iv_limit' not in infos.columns:             # 字段的iv阈值，过小则剔除字段
        print('infos中不包括iv_limit， 统一赋值为默认值：0.2')
        infos['iv_limit'] = 0.2
    if 'r_limit' not in infos.columns:              # 字段之间相关性系数阈值，超过则剔除字段
        print('infos中不包括r_limit， 统一赋值为默认值：0.8')
        infos['r_limit'] = 0.8
    if 'random_state' not in infos.columns:         # 随机种子
        print('infos中不包括random_state， 统一赋值为默认值：None')
        infos['random_state'] = None
    if 'n_recent' not in infos.columns:             # 加工近n月基础数据的月份数
        print('infos中不包括n_recent， 统一赋值为默认值：3')
        infos['n_recent'] = 3
    if if_coltolower:  # 字段名统一转换未小写
        for i in infos.columns[infos.columns.str.contains('^col_')]:
            infos[i] = infos[i].apply(to_lower)
    # 调整字段顺序
    order_new = ['model_name'] + [i for i in infos.columns if i not in ['model_name']]
    infos = infos[order_new]

    model_lack = set(info_changed.index) - set(infos.index)
    if model_lack:
        raise Exception(f'返回结果infos缺少模型：{model_lack}')
    if infos.model_wd_traintest.duplicated().sum() > 0:
        warnings.warn('model_wd_traintest有重复，请检查！')
    if set(infos.index) != set(info_changed.index):
        warnings.warn('infos的index 与 model_names 不同，请检查！')
    test_warn1 = infos.loc[infos.month_test.isnull() & infos.table_test.notnull()].index.values
    if len(test_warn1):
        s = f'下列模型未设置month_test，将忽略table_test，不设置时间外测试集，请确认:{test_warn1}'
        warnings.warn(s)
    test_warn2 = infos.loc[infos.month_test.notnull() & infos.timeout_limit.isnull()].index.values
    if len(test_warn2):
        s = f'\n下列模型设置了month_test，但未设置timeout_limit，时间外测试集将取全量样本，请确认:{test_warn2}'
        warnings.warn(s)

    evaluate_na = infos.month_test.isnull() & infos.timein_count.isnull()
    if evaluate_na.sum() > 0:
        s = f"下列模型只有训练数据：\n{infos.loc[evaluate_na, ['model_name', 'month_test', 'timein_count']]}"
        raise Exception(s)

    return infos

