import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import Callable, Dict, Optional, Union
import datetime
from collections import OrderedDict
import re
from scipy import sparse
from collections import namedtuple
import string
from string import punctuation
from sklearn.tree import DecisionTreeClassifier, _tree
import inspect
import warnings
import builtins
import math



# 'NumStrSpliter', 'FeaturePrefilter', 'Mdlp_dt', 'NewValueHandler', 'WoeTransformer',
__all__ = [
    'na_inf_examine',
    'colvalue_exam',
    'dropdup',
    'create_pipeline',
    'create_featurenion',
    'get_index',
    'interval_to_str',
    'col_pop',
    'my_div',
    'to_namedtuple',
    'choradd_namedtuple',
    'get_onlyvalue',
    'add_printindent',
    'NumStrSpliter_DF',
    'FeaturePrefilter_DF',
    'Mdlp_dt_DF',
    'NewValueHandler_DF',
    'WoeTransformer_DF',
    'Pipeline_DF',
    'StandardScaler_DF',
    'OneHotEncoder_DF',
    'MinMaxScaler_DF',
    'FeatureUnion_DF',
    'LogisticRegression_DF',
    'DecisionTreeClassifier_DF',
    'RandomForestClassifier_DF',
    'LGBMClassifier_DF',
    'XGBClassifier_DF',
    'OutlierHandler_DF',
    'SimpleImputer_DF',
    'CategoricalEncoder_DF',
    'PsiTransformer_DF',
    'ValueConverter_DF',
    'MinMaxScaler_DF',
    'ObjectBacktoInt_DF'
    #'ColFilter_DF',
    #'NochangeTranformer_DF'
    ]

from sklearn.utils.sparsefuncs import _get_median


def Pipeline_DF(steps, X, y=None, verbose=False):
    """
    根据给定的步骤列表依次对DataFrame数据进行转换
    :param steps: 包含多个转换步骤的列表，每个元素是一个元组 (step_name, transformer)
    :param X: 输入的数据（DataFrame）
    :param y: 标签（可选，用于某些变换器）
    :param verbose: 是否输出每个步骤的详细信息
    :return: 转换后的DataFrame数据
    """
    for name, transformer in steps:
        if verbose:
            print(f"Applying step: {name}")
        X = transformer.fit_transform(X, y) if y is not None else transformer.fit_transform(X)

    return X

# ----------------------------------------------------------------------------------------------------------------------

def Mdlp_dt_DF(precision=4, print_indent='', data=None):
    """
    使用MDLP算法进行数据离散化
    :param precision: 离散化精度
    :param print_indent: 输出的缩进级别
    :param data: 输入的数据（NumPy数组或DataFrame）
    :return: 离散化后的数据
    """

    # 这里只是模拟一个简单的离散化过程，具体的MDLP算法需要更复杂的实现
    if data is None:
        raise ValueError("Data should be provided for discretization.")

    print(f"{print_indent}开始离散化操作...")

    # 假设我们将数据四舍五入到指定精度
    discretized_data = np.round(data, precision)

    print(f"{print_indent}离散化操作完成。")

    return discretized_data

# ----------------------------------------------------------------------------------------------------------------------

def PsiTransformer_DF(Pcase=None, Ncase=None, psi_limit=None, print_indent='', warn_mark=False):
    """
    假设的PsiTransformer_DF函数，用于处理Pcase和Ncase数据并应用psi_limit约束。
    :param Pcase: 输入的数据集Pcase
    :param Ncase: 输入的数据集Ncase
    :param psi_limit: 用于限制操作的阈值
    :param print_indent: 控制输出格式的缩进
    :param warn_mark: 是否打印警告标志
    :return: 转换后的数据
    """

    # 检查是否提供了Pcase和Ncase
    if Pcase is None or Ncase is None:
        raise ValueError("Pcase和Ncase必须提供")

    print(f"{print_indent}开始PsiTransformer处理...")

    # 进行一些操作，假设这里是基于Pcase和Ncase的处理
    # 比如对两个数据集进行一些简单的加法、减法或其他操作
    result = Pcase - Ncase

    # 应用psi_limit限制条件，如果提供了psi_limit
    if psi_limit is not None:
        result[result > psi_limit] = psi_limit  # 将大于psi_limit的值限制为psi_limit

    # 打印警告信息（如果需要）
    if warn_mark:
        print(f"{print_indent}警告: 超过psi_limit的值已经被限制。")

    print(f"{print_indent}PsiTransformer处理完成。")

    return result

# ----------------------------------------------------------------------------------------------------------------------

def MissingIndicator(X, missing_values=np.nan):
    """
    该函数接受一个数据集 X，返回一个与 X 同形状的缺失值指示矩阵。
    1 表示缺失值，0 表示非缺失值。

    :param X: 输入数据（numpy 数组或 pandas DataFrame）。
    :param missing_values: 用于表示缺失值的标识，默认为 np.nan。
    :return: 缺失值指示矩阵（numpy 数组），1 表示缺失值，0 表示非缺失值。
    """
    # 创建一个与输入数据 X 相同形状的布尔矩阵，指示缺失值的位置
    indicator_matrix = np.isin(X, missing_values)

    # 将布尔值转换为 1（缺失）和 0（非缺失）
    return indicator_matrix.astype(int)

# ----------------------------------------------------------------------------------------------------------------------

def FeatureUnion_DF(transformers, verbose=False):
    """
    Combine multiple transformers into a single transformer that applies each transformer to the data.

    Parameters:
    transformers: List of tuples [(name, transformer), ...]
        - name: A string name for the transformer.
        - transformer: A transformer object that follows the fit-transform API.

    verbose: bool, optional (default=False)
        If True, prints information about the transformers being applied.

    Returns:
    transformer: A transformer that applies each of the individual transformers.
    """
    # Using an ordered dictionary to store the transformers, so that they are applied in order
    transformers_dict = OrderedDict(transformers)

    class CombinedTransformer:
        def __init__(self, transformers):
            self.transformers = transformers

        def fit(self, X, y=None):
            """
            Fit each of the transformers to the data.
            """
            for name, transformer in self.transformers.items():
                if verbose:
                    print(f"Fitting transformer: {name}")
                transformer.fit(X, y)
            return self

        def transform(self, X):
            """
            Apply each transformer to the data and concatenate the results.
            """
            transformed_data = []
            for name, transformer in self.transformers.items():
                if verbose:
                    print(f"Transforming data with: {name}")
                transformed_data.append(transformer.transform(X))
            # Concatenate all transformed data (you can modify this depending on how you want the output)
            return np.concatenate(transformed_data, axis=1)  # Example: concatenating along columns

        def fit_transform(self, X, y=None):
            self.fit(X, y)
            return self.transform(X)

    return CombinedTransformer(transformers_dict)

# ----------------------------------------------------------------------------------------------------------------------

class WoeTransformer_DF:
    def __init__(self, Pcase, Ncase, to_woe=True):
        """
        Initialize the WoeTransformer_DF class.

        Parameters:
        Pcase (int): Number of positive cases (e.g., default cases).
        Ncase (int): Number of negative cases (e.g., non-default cases).
        to_woe (bool): Whether to transform the data to WOE values.
        """
        self.Pcase = Pcase
        self.Ncase = Ncase
        self.to_woe = to_woe

    def _woe_iv(self, df, feature, target):
        """
        Calculate WOE and IV for a given feature.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        feature (str): The feature column name.
        target (str): The target column name.

        Returns:
        pd.DataFrame: A dataframe containing WOE and IV values for each bin.
        """
        # Group the data by the feature and calculate the number of positive and negative cases
        grouped = df.groupby(feature)[target].agg(['sum', 'count'])
        grouped.columns = ['Pcase_bin', 'Total_bin']
        grouped['Ncase_bin'] = grouped['Total_bin'] - grouped['Pcase_bin']

        # Calculate the distribution of positive and negative cases
        grouped['P_dist'] = grouped['Pcase_bin'] / self.Pcase
        grouped['N_dist'] = grouped['Ncase_bin'] / self.Ncase

        # Calculate WOE and IV
        grouped['WOE'] = np.log(grouped['P_dist'] / grouped['N_dist'])
        grouped['IV'] = (grouped['P_dist'] - grouped['N_dist']) * grouped['WOE']

        # If to_woe is True, replace the feature values with WOE values
        if self.to_woe:
            woe_dict = grouped['WOE'].to_dict()
            df[feature] = df[feature].map(woe_dict)

        return grouped

    def transform(self, df, features, target):
        """
        Transform the input dataframe by calculating WOE and IV for each feature.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        features (list): List of feature column names.
        target (str): The target column name.

        Returns:
        pd.DataFrame: The transformed dataframe with WOE values.
        """
        for feature in features:
            self._woe_iv(df, feature, target)
        return df

# ----------------------------------------------------------------------------------------------------------------------

def na_inf_examine(data):
    """
    检查DataFrame是否存在缺失值/无穷值
    """
    col_na = data.isnull().sum()
    col_na = col_na.loc[col_na > 0]
    s = '数据存在缺失值或±inf，请检查！'
    if len(col_na) > 0:
        s += f'\n\n缺失值：\n{col_na}'

    col_inf = ((data == np.float('Inf')) | (data == np.float('-Inf'))).sum()
    col_inf = col_inf.loc[col_inf > 0]
    if len(col_inf) > 0:
        s += f'\n\n±inf：\n{col_inf}'

    if (len(col_na) > 0) | (len(col_inf) > 0):
        raise Exception(s)
# ----------------------------------------------------------------------------------------------------------------------


def dropdup(x):
    """
    # 列表去重函数
    :param x: 列表
    :return: 元素去重后的列表
    """
    return sorted(set(x), key=x.index)


# ----------------------------------------------------------------------------------------------------------------------

def colvalue_exam(data, colname, valuerange):
    """
    检查字段取值的合规性
    :param data: DataFrame
    :param colname: 字段名
    :param valuerange: 取值范围
    :return: None
    """""
    # valueerr = data.loc[~data[colname].isin(valuerange), colname].unique()  正常如此操作即可，有些环境貌似存在某种bug：
    xj = data[colname].drop_duplicates()
    if (xj.isnull().sum() > 0) & any(xj.astype(str) == 'nan'):  # 过滤掉None等
        xj = list(xj[xj.notnull()]) + [np.nan]
    valueerr = set(xj) - set(valuerange)
    if len(valueerr):
        s = f"{colname}列的取值范围为{valuerange}，请修改不合规取值：{valueerr}"
        raise Exception(s)


# ----------------------------------------------------------------------------------------------------------------------



def create_pipeline(keys, tran_dict, indent=None):
    """
    创建Pipeline_DF（目的：不同流水线之间可复用相同阶段的fit结果，避免重复计算）
    :param keys: 需要组合的tran_dict中转换器的键
    :param tran_dict: 键值对，value为各个转换器
    :param indent: 信息打印时的缩进
    :return:
    """
    for i in range(len(keys)):
        k = keys[i]
        if not hasattr(tran_dict[k], 'tran_chain_expect'):
            class_name = display_classname(str(tran_dict[k]))
            tran_dict[
                k].tran_chain_expect = class_name if i == 0 else f"{tran_dict[keys[i - 1]].tran_chain_expect} ==> {class_name}"
    pipe = Pipeline_DF([(i, tran_dict[i]) for i in keys], verbose=indent)
    pipe.tran_chain_expect = tran_dict[keys[-1]].tran_chain_expect
    return pipe


# ----------------------------------------------------------------------------------------------------------------------


def create_featurenion(keys, tran_dict, indent=None):
    """
    创建FeatureUnion_DF（目的：不同流水线之间可复用相同阶段的fit结果，避免重复计算）
    :param keys: 需要组合的tran_dict中Pipeline的键
    :param tran_dict: 键值对，value为各个Pipeline
    :param indent: 信息打印时的缩进
    :return:
    """
    union = FeatureUnion_DF([(i, tran_dict[i]) for i in keys], verbose=indent)
    tran_union = [tran_dict[i].tran_chain_expect for i in keys]
    union.tran_chain_expect = '[' + ' + '.join(tran_union) + ']'
    return union


# ----------------------------------------------------------------------------------------------------------------------


def get_index(data, level=None, name=None):  # 获取层次化索引
    index = data.index
    res = index.to_frame()
    if set(index.names) != {None}:
        res.columns = index.names
    if name is not None:
        if level is not None:
            warnings.warn('get_index函数 level、name参数同时设置，将忽略level参数，以name参数获取结果！')
        select = name
        return Series(res.loc[:, select].values)
    elif level is not None:
        select = level
        return Series(res.iloc[:, select].values)
    elif level is None:
        return Series(res.values)


# ----------------------------------------------------------------------------------------------------------------------


def display_classname(CLASS):
    """
    精简类名，为打印时更清晰的展示信息
    :param CLASS: 类
    :return: 精简后的类名字符
    """
    if isinstance(CLASS, (Pipeline_DF, FeatureUnion_DF)):
        return re.sub('.*?\.', '', CLASS.__name__)
    class_to_str = str(CLASS)
    s = re.sub("print_indent=' *'", '', class_to_str)
    # s = re.sub('\( *, *', '(', s)
    s = re.sub('\n *', ' ', s)
    return s


# ----------------------------------------------------------------------------------------------------------------------


def col_pop(col):
    """
    训练测试、预测打分函数中使用：各分数段用户量及占比统计
    :param col: 分数分箱字段
    :return: 统计结果
    """
    col_pop = col.value_counts()
    if type(col.values).__name__ == 'Categorical':
        col_pop.index = [interval_to_str(i) for i in col_pop.index]  # astype(str)为最好，此处为兼容多版本
        order = [interval_to_str(i) for i in col.values.categories]
        col_pop = col_pop[order]
    col_pop = DataFrame(col_pop)
    col_pop.columns = ['用户量']
    col_pop['占比'] = (col_pop.用户量 / len(col))
    return col_pop


# ----------------------------------------------------------------------------------------------------------------------


def interval_to_str(x):
    """
    将区间值（Interval）转换为字符串（str）
     # 列.astype(str)为最好，本函数为兼容多版本（某些项目的包版本不支持使用astype实现）
    :param x: 区间值
    :return: 字符串
    """
    if type(x).__name__ == 'Interval':
        left = x.left
        right = x.right
        sign_left = {True: '(', False: '['}[x.open_left]
        sign_right = {True: ')', False: ']'}[x.open_right]
        return f'{sign_left}{left}, {right}{sign_right}'
    else:
        return str(x)


# ----------------------------------------------------------------------------------------------------------------------


def enctry(s, k=None):
    """
    字符串加密函数
    :param s: 待加密的字符串
    :param k: 密钥
    :return: 加密后的字符串
    """
    if k is None:
        k = ''.join(np.arange(10).astype(
            str)) + string.ascii_letters + punctuation + 'ΑΒΓΔΕΖΗΘΙΚ∧ΜΝΞΟ∏Ρ∑ΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωόά' + 'āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ' + '·￥…（）—【】、；：’“《，》。？'
    if len(s) > len(k):
        ss = 'enctry: 参数s的长度小于k，可能无法保证一一对应！'
        warnings.warn(ss)
    encry_str = ""
    for i, j in zip(s, k):
        # i为字符，j为秘钥字符
        temp = str(ord(i) + ord(j)) + '_'  # 加密字符 = 字符的Unicode码 + 秘钥的Unicode码
        encry_str = encry_str + temp
    return encry_str


def dectry(p, k=None):
    """
    字符串解密函数
    :param s: 待解密的字符串
    :param k: 密钥
    :return: 解密后的字符串
    """
    if k is None:
        k = ''.join(np.arange(10).astype(
            str)) + string.ascii_letters + punctuation + 'ΑΒΓΔΕΖΗΘΙΚ∧ΜΝΞΟ∏Ρ∑ΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωόά' + 'āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ' + '·￥…（）—【】、；：’“《，》。？'
    dec_str = ""
    for i, j in zip(p.split("_")[:-1], k):
        # i 为加密字符，j为秘钥字符
        temp = chr(int(i) - ord(j))  # 解密字符 = (加密Unicode码字符 - 秘钥字符的Unicode码)的单字节字符
        dec_str = dec_str + temp
    return dec_str


# ----------------------------------------------------------------------------------------------------------------------


def to_namedtuple(x):
    """
    将字典转换为命名元组
    :param x: 字典，或可以被dict函数转换为字典的类型
    :return: 命名元组
    """
    x_dict = dict(x)
    return namedtuple('Infotuple', ' '.join(x_dict.keys()))(**x_dict)


def choradd_namedtuple(x, choradd_dict=None, rep_dict=None):
    """
    修改或添加命名元组的属性（有则修改，无则添加）
    :param x: 命名元组
    :param choradd_dict: 字典，{键：替换值值（初始值）}
    :param rep_dict: 字典{k: v}，将所有值中的 k 替换为 v
    :return: 新的命名元组
    """
    # to_namedtuple = lambda x: namedtuple('Infotuple', ' '.join(dict(x).keys()))(**dict(x))
    x_dict = x._asdict()
    if rep_dict is not None:
        for k in x_dict.keys():
            for kk in rep_dict.keys():
                if isinstance(x_dict[k], str):
                    x_dict[k] = x_dict[k].replace(kk, str(rep_dict[kk]))
    if choradd_dict is not None:
        for name_attr, new_value in choradd_dict.items():
            x_dict[name_attr] = new_value
    return to_namedtuple(x_dict)


# ----------------------------------------------------------------------------------------------------------------------


def my_div(x1, x2):
    """
    自定义除法
    :param x1: 被除数
    :param x2: 除数
    :return: 商（自定义）
    """
    x1, x2 = x1.fillna(0).copy(), x2.fillna(0).copy()
    x2_not0_unique = x2[x2 != 0].drop_duplicates().sort_values()
    if (x2_not0_unique > 0).sum():  # 除数有正数，将[绝对值]过小的取值赋值为可接受的[绝对值]最小值
        con = x2 > 0
        to_value = x2[con].iloc[min(1000, math.ceil(con.sum() * 0.01))]  # 可接受的[绝对值]最小值
        x2[x2 < to_value] = to_value
    elif (x2_not0_unique < 0).sum():  # 除数全是负数，将绝对值过小的取值赋值为可接受的绝对值最小值
        con = x2 < 0
        to_value = x2[con].iloc[-min(1000, math.ceil(con.sum() * 0.01))]  # 可接受的绝对值最小值
        x2[x2 > to_value] = to_value
    elif len(x2_not0_unique) == 0:  # 除数取值恒为0
        s = f"my_div: {x2.name + '字段' if x2.name else '除数'}取值全为0，暂将其值全部赋值为1，确保程序正常执行，请检查!"
        warnings.warn(s)
        x2 = 1
    return x1 / x2


# ----------------------------------------------------------------------------------------------------------------------


def get_onlyvalue(x):
    """
    检验并获取长度为1的序列的唯一元素
    :param x: 输入变量
    :return: 如果x长度不为1则报错，若长度为1则返回
    """
    u = np.unique(x)
    if len(u) == 1:
        return u[0]
    else:
        s = f'存在多个取值：{u}'
        raise Exception(s)


# ----------------------------------------------------------------------------------------------------------------------


def add_printindent(x, indent='    '):
    """
    打印DataFrame、Series、str(以及可以通过str函数转换为字符的类型)时缩进
    :param x: DataFrame、Series、str等待打印的变量
    :param indent: 缩进
    :return: None，仅打印
    """
    if isinstance(x, (DataFrame, Series)):
        if len(x) == 0:  # 打印空对象
            return '\n' + '\n'.join([indent + i for i in str(x).split('\n')])
        x_ad = x.copy()
        idx = list(DataFrame(list(x_ad.index)).astype(str).T.values)  # 为适应层次化索引
        idx[0] = (indent + Series(idx[0]).astype(str)).values
        x_ad.index = idx
        if isinstance(x, DataFrame):
            return x_ad
        elif isinstance(x, Series):
            p = ''
            to_str = x_ad.index.map(lambda x: '  '.join(x)) + '    ' + x_ad.values.astype(str)
            for i in to_str:
                p += '\n' + i
            p += f"\n{indent}dtype: {x_ad.dtype}"
            return p
        else:
            s = "参数x的类型必须为DataFrame、Series"
            raise Exception(s)
    else:
        x_ad = str(x)
        crlf_start = x_ad[0] if x_ad[0] == '\n' else ''  # 为保留第一行开头的换行符
        crlf_end = x_ad[-1] if x_ad[-1] == '\n' else ''  # 为保留最后一行的换行符
        x_ad = '\n'.join([(indent + i) for i in x_ad.strip('\n').split('\n')])
        x_ad = f"{crlf_start}{x_ad}{crlf_end}"
        return x_ad


# ----------------------------------------------------------------------------------------------------------------------


class NumStrSpliter(BaseEstimator, TransformerMixin):
    def __init__(self, select, ob_to_cat=False):
        """
        筛选数值、非数值字段的DataFrame（为DataFrame设计，而非数组）
        :param select: 取值‘num’时筛选数值型字段，取值‘notnum’时筛选欸数值型数据
        :param ob_to_cat: 是否将非数值字段转化为分类型（暂无落实必要）
        """
        self.select = select
        self.ob_to_cat = ob_to_cat

    def fit(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        col_type = X.dtypes.astype(str).str.replace('64.*?$|32.*?$|16.*?$|8.*?$', '')
        if self.select == 'num':
            self.col_select_ = col_type[col_type.isin(['float', 'int', 'uint'])]
            print(f'    提取数值型字段名：{len(self.col_select_)}列')
            if len(self.col_select_) == 0:
                s = '    NumStrSpliter 数据中未出现数值型字段！'
                print(s)
                warnings.warn(s)
        if self.select == 'notnum':
            self.col_select_ = col_type[col_type == 'object']
            print(f'    提取非数值型字段名：{len(self.col_select_)}')
            if len(self.col_select_) == 0:
                s = '    NumStrSpliter 数据中未出现非数值型字段！'
                print(s)
                warnings.warn(s)
        col_rest = col_type[~col_type.isin(['float', 'int', 'uint', 'object'])].index
        col_type[col_rest]
        if len(col_rest) > 0:
            s = f'NumStrSpliter 下列字段未被归类到数值型和类别型字段集合中，请确认：\n{col_type[col_rest]}'
            warnings.warn(s)
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        # 筛选所需字段
        X = X[self.col_select_.index]

        if self.select == 'num':
            print('    提取数值型数据')
            for i in X.columns:
                try:  # 为解决新数据由于错位导致数值字段中误入字符的情况，这样可及早发现异常
                    X.loc[:, i] = X.loc[:, i].astype(float)
                except Exception as er:
                    raise Exception(f'NumStrSpliter {i}: {str(er)}')

        elif self.select == 'notnum':
            print('    提取非数值型数据')
            for i in X.columns:
                try:
                    na_idx = X.loc[:, i].isnull()
                    X.loc[:, i] = X.loc[:, i].astype(str)
                    X.loc[na_idx, i] = np.nan
                except Exception as er:
                    raise Exception(f'NumStrSpliter {i}: {str(er)}')

        self.transform_out_colnames_ = X.columns  # 提前设定免，否则进入DF转换环节会报错
        return X


# ----------------------------------------------------------------------------------------------------------------------


class FeaturePrefilter(BaseEstimator, TransformerMixin):
    def __init__(self, freq_limit=1, unique_limit=np.float64('inf'), valuecount_limit=np.float64('inf')):
        """
        字段预筛选（为DataFrame设计，而非数组）
        :param freq_limit: 剔除取值过于集中（>=freq_limit）的字段
        :param unique_limit: 1. 剔除取值个数大于等于unique_limit 的字段，取值无穷大时不处理
                             2. 只应用于类别型字段，数值型字段该参数应设置为无穷大（数值字段取值个数自然很多）
        :param valuecount_limit: 1. 将样本数 <= valuecount_limit的取值替换为字符“其他”，取值无穷大时不处理
                                 2. valuecount_limit=None时，不做处理
                                 仅作用于类别型字段，数值型字段该参数应设置为无穷大（数值字段取值分散，样本数少很正常）
        """
        self.freq_limit = freq_limit
        self.unique_limit = unique_limit
        self.valuecount_limit = valuecount_limit

    def fit(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()

        def max_freq(x):
            if x.notnull().sum() == 0:
                max_count = len(x)
            else:
                max_count = max(x.value_counts())
            return max_count / len(x)

        self.col_freq_ = X.apply(max_freq)
        col_del_freq = self.col_freq_[self.col_freq_ >= self.freq_limit].index
        if len(col_del_freq) > 0:
            self.col_del_freq_ = col_del_freq
            print(f'    删除{len(self.col_del_freq_)}个取值集中的字段(>={self.freq_limit})：{list(self.col_del_freq_)}')
            X = X.drop(columns=self.col_del_freq_)

        con_char = (X.dtypes == object)
        col_del_unique = list()
        if self.unique_limit < np.float64('inf'):  # 仅针对类别字段， 数值字段unique_limit应为无穷大
            if con_char.sum() > 0:  # 存在类别字段
                if (~con_char).sum():
                    print(f'    unique_limit仅针对类别字段，跳过{(~con_char).sum()}个数值字段，仅处理类别字段')
                unique_limit_actual = self.unique_limit
                if self.unique_limit > len(X):
                    r_max = 0.99
                    unique_limit_actual = int(len(X) * r_max)
                    s = f"    FeaturePrefilter_DF unique_limit({self.unique_limit})小于X行数({len(X)}), 添加unique_limit_actual：len(X)*{r_max}={unique_limit_actual}"
                    print(s)
                    warnings.warn(s)
                    self.unique_limit_actual = unique_limit_actual
                self.col_unique_ = X.loc[:, con_char].apply(lambda x: x.unique().shape[0])  # 仅针对类别字段
                col_del_unique = self.col_unique_[self.col_unique_ >= unique_limit_actual].index
                if len(col_del_unique) > 0:
                    self.col_del_unique_ = col_del_unique
                    print(
                        f'    删除{len(self.col_del_unique_)}个取值过多的字段(>={unique_limit_actual})：{list(self.col_del_unique_)}')
                    X = X.drop(columns=self.col_del_unique_)
            else:  # 不存在类别字段
                print('    unique_limit仅针对类别字段，但数据中不存在类别字段，跳过')

        col_del_valuecount = []
        self.convert_ = {}
        con_char = (X.dtypes == object)
        if self.valuecount_limit < np.float64('inf'):  # 仅针对类别字段， 数值字段valuecount_limit应为无穷大
            print(f'    合并字段中样本数 <= {self.valuecount_limit}的取值（替换为“其他_countlow”）')
            if con_char.sum() > 0:  # 存在类别字段
                if (~con_char).sum():
                    print(
                        f'    unique_limit仅针对类别字段，跳过{(~con_char).sum()}个数值字段，仅处理{con_char.sum()}个类别字段')
                for i in X.columns[con_char]:
                    v_c = X[i].value_counts()
                    v_replace = v_c[v_c <= self.valuecount_limit].index
                    if len(v_replace) == len(v_c):  # 某类别字段所有取值样本数都过少，则删除该字段
                        col_del_valuecount.append(i)
                    elif len(v_replace) > 0:
                        self.convert_[i] = Series('其他_countlow', index=v_replace)
                print(f'        合并了{len(self.convert_)}个字段的取值')
            else:  # 不存在类别字段
                print('    valuecount_limit仅针对类别字段，但数据中不存在类别字段，跳过')

            if len(col_del_valuecount) > 0:
                self.col_del_valuecount_ = col_del_valuecount
                print(
                    f'        删除{len(col_del_valuecount)}个字段，所有取值样本数都 <= {self.valuecount_limit}：{list(self.col_del_valuecount_)}')
                X = X.drop(columns=self.col_del_valuecount_)

        del_colnames = col_del_freq | col_del_unique | col_del_valuecount
        if len(del_colnames) > 0:
            self.del_colnames_ = del_colnames
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        if hasattr(self, 'del_colnames_'):
            print(f'    剔除{len(self.del_colnames_)}个字段：{list(self.del_colnames_)}')
            X.drop(self.del_colnames_, axis=1, inplace=True)

        if len(self.convert_) > 0:
            print(
                f"    合并{len(self.convert_)}个字段中样本数 <= {self.valuecount_limit}的取值（替换为“其他_countlow”）:{list(self.convert_.keys())}")
            for i in self.convert_.keys():
                X[i] = X[i].map(lambda x: self.convert_[i][x] if x in self.convert_[i].keys() else x)

        self.transform_out_colnames_ = X.columns
        return X


# ----------------------------------------------------------------------------------------------------------------------


class Mdlp_dt(BaseEstimator, TransformerMixin):
    def __init__(self, bin_uplimit=None, if_del=True, precision=6, min_binsize=30):
        """
        mdlp离散（为DataFrame设计，而非数组）
        :param bin_uplimit: 分箱数上限
        :param if_del: 是否剔除未切分开的字段（即计算切点后，只有最小值、最大值两个切点）
        :param precision: cut时的精度参数
        :param min_binsize: DecisionTreeClassifier(min_samples_split=self.min_binsize)
        """
        self.bin_uplimit = bin_uplimit
        self.if_del = if_del
        self.precision = precision
        self.min_binsize = min_binsize

    def cp_mdlp(self, cps, x, y, bin_uplimit=None):
        """
        确定切点
        :param cps: 预选切点
        :param x: 特征Series
        :param y: 目标Series
        :param bin_uplimit: 分箱数上限
        :return: 切点
        """
        import math
        def mylog(x):
            """
            对log函数稍作修改
            :param x: numpy数组 或 pandas Series对象 或 int标量
            :return: 对于numpy数组、pandas Series对象：调用修改后的np.log函数，返回对数化（修正）后的数组、Series
                     对于int：调用修改后的math.log函数，返回对数化（修正）后的int
            """
            if type(x) in (np.ndarray, Series):
                x[x <= 1e-10] = 1
                return np.log(x)
            if type(x) == int:
                x = 1 if x < 1e-10 else x
                return math.log(x)

        def ent(y):
            """
            熵计算函数
            :param y: 分类数据Series（numpy数组应该也可），一般情况下为因变量字段
            :return: 熵值（标量）
            """
            p = y.value_counts() / len(y)
            e = -sum(p * mylog(p))
            return e

        def mdlStop(cp, x, y):
            """
            判断切点是否有效
            :param cp: 切点
            :param x: 自变量字段Series
            :param y: 因变量字段Series
            :return: 有效则返回该切点的增益(标量)，无效则返回None
            """
            n = len(y)
            wx = (x <= cp).values
            wn = wx.sum() / n
            e1 = wn * ent(y[wx])
            e2 = (1 - wn) * ent(y[~wx])
            entropy = e1 + e2
            es = ent(y)
            y_left = y[(x <= cp).values]
            y_right = y[(x > cp).values]
            gain = es - entropy
            l0 = y.unique()
            l1 = y_left.unique()
            l2 = y_right.unique()
            k = len(l0)
            k1 = len(l1)
            k2 = len(l2)
            delta = mylog(3 ** k - 2) - (k * es - k1 * ent(y_left) - k2 * ent(y_right))
            cond = mylog(n - 1) / n + delta / n
            if gain < cond:
                return None
            else:
                return gain

        cp = Series([float('-Inf'), float('Inf')])  # cp = Series([x.min(), x.max()])
        cp_add = Series([None])
        if_continue = Series(dtype=object)
        while len(cp_add) > 0:
            loop = Series(np.arange(len(cp) - 1))
            loop = loop.loc[loop.index.difference(if_continue)]
            cp_new = Series(dtype=np.float64)
            for i in loop:
                if i == 0:
                    filter_bool = ((x >= cp.iloc[i]) & (x <= cp.iloc[i + 1])).values
                    xx = x[filter_bool]
                    yy = y[filter_bool]
                else:
                    filter_bool = ((x > cp.iloc[i]) & (x <= cp.iloc[i + 1])).values
                    xx = x[filter_bool]
                    yy = y[filter_bool]

                cp_local = cps[(cps > cp.iloc[i]) & (cps < cp.iloc[i + 1])]
                if len(cp_local) > 0:
                    if mdlStop(cp_local[0], xx, yy) is not None:
                        cp_new.loc[i] = cp_local[0]
                else:
                    cp_new.loc[i] = None
            cp_add = cp_new[cp_new.notnull()]
            if_continue = cp_new[cp_new.isnull()].index
            p = pd.concat([cp, cp_add], axis=0).sort_values()
            if (False if bin_uplimit is None else len(p) > (bin_uplimit + 1)):
                break
            else:
                cp = p
        return cp.astype(float).values

    def fit(self, X, y):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        if y is None:
            raise Exception('Mdlp_dt fit函数参数y必须设置！')

        # 检验y的取值个数
        if len(y.unique()) != 2:
            raise Exception(f'Mdlp_dt 目标字段的不同取值应该为2个，但实际：{len(y.unique())}个：{y.unique()}')

        X = X.copy()
        col_types = X.dtypes.apply(str)
        col_notnum = col_types[~col_types.str.match('int|float')]
        if len(col_notnum) > 0:
            raise Exception(f'Mdlp_dt 存在非数据值字段：{list(col_notnum.index)}')

        col_na_count = X.isnull().sum()
        col_na_count = col_na_count[col_na_count > 0]
        if len(col_na_count) > 0:
            s = f'Mdlp_dt 存在缺失值：{dict(col_na_count)}'
            raise Exception(s)

        if self.precision > 6:
            p_cp = self.precision - 1
        else:
            p_cp = self.precision

        col_cutpoint = Series(dtype=object)
        for i in X.columns:
            X[i] = np.round(X[i], self.precision)
            model = DecisionTreeClassifier(min_samples_split=self.min_binsize)
            model.fit(X[[i]], y)
            tree_ = model.tree_
            cps = tree_.threshold[tree_.feature != _tree.TREE_UNDEFINED]
            col_cutpoint[i] = np.round(self.cp_mdlp(cps, X[i], y, self.bin_uplimit), p_cp)

            # 修正切点
            xj = pd.cut(X[i], col_cutpoint[i], include_lowest=True, precision=self.precision, duplicates='drop')
            cp_del = set(
                xj.value_counts()[(xj.value_counts() < self.min_binsize).values].index.map(lambda x: x.right))  # 兼容版本
            col_cutpoint[i] = [i for i in col_cutpoint[i] if i not in cp_del]

        self.col_cutpoint_ = col_cutpoint
        self.del_colnames_ = col_cutpoint[col_cutpoint.apply(len) == 2].index  # 没有分切开的字段
        if len(self.del_colnames_) > 0:
            print(f'    提取未离散的{len(self.del_colnames_)}字段名（无有效切点）：{list(self.del_colnames_)}')
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        if len(self.fit_in_colnames_) == 0:
            print('    data out: 零列数据输出！')
            self.transform_out_colnames_ = X.columns
            return X

        if hasattr(self, 'del_colnames_') & (len(self.del_colnames_) > 0):
            s = f'    剔除{len(self.del_colnames_)}个字段：{list(self.del_colnames_)}'
            if self.if_del:
                print(s)
                X.drop(self.del_colnames_, axis=1, inplace=True)
            else:
                print(s.replace('剔除', '未剔除'))

        for i in X.columns:
            cut_point = self.col_cutpoint_[i]
            X[i] = X[i].astype(float)  # 曾经报错：实属无奈之举，规避训练数据与新数据字段类型冲突，应从源头统一
            X[i] = np.round(X[i], self.precision)
            X[i] = np.where(X[i] < cut_point[0], cut_point[0], X[i])
            X[i] = np.where(X[i] > cut_point[-1], cut_point[-1], X[i])
            X[i] = pd.cut(X[i], self.col_cutpoint_[i], include_lowest=True, precision=self.precision, duplicates='drop')
            X[i] = [interval_to_str(interval) for interval in X[i]]  # astype(str)为最好，此处为兼容多版本
        self.transform_out_colnames_ = X.columns
        return X


# -----------------------------------------------------------------------------------------------------------------------


class NewValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        """
        用于查找并替换新数据集中的新水平值（非数值型字段）,为DataFrame设计，而非数组
        :param strategy: 替换新水平值的策略，目前只支持替换为训练集对应字段的众数，其他策略用时再添加
        """
        self.strategy = strategy

    def fit(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        d_types = X.dtypes.apply(str)
        col_num = d_types[d_types.str.contains('int|float')].index
        if len(col_num) > 0:  # 操作不可用于数值型字段
            raise Exception('NewValueHandler 不能包括数值型字段：%s', list(col_num))

        if X.shape[1] == 1:
            self.col_value_ = Series([set(X.iloc[:, 0].unique())], index=[X.columns[0]])
        else:
            try:
                self.col_value_ = X.apply(lambda a: set(a.unique()))  # 修改如果慢，则修改为一次聚合，在结果数据上统计众数
            except:  # 某些情况下好像有bug，解决：
                self.col_value_ = X.apply(lambda a: a.unique()).apply(lambda x: ','.join(x)).apply(
                    lambda x: set(x.split(',')))

        if self.strategy == 'most_frequent':  # 暂未设置其他选项，可扩充
            self.col_mode_ = X.mode().iloc[0, :]
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        for i in X.columns:
            col_value = set(X[i].unique())
            value_new = col_value - self.col_value_[i]
            if len(value_new) > 0:
                if_new = X[i].isin(value_new)
                mode = self.col_mode_[i]
                X.loc[if_new, i] = mode
                print(f"    字段 {i} 的下列{if_new.sum()}个新取值替换为众数({mode})：{value_new}")
        self.transform_out_colnames_ = X.columns
        return X


# ----------------------------------------------------------------------------------------------------------------------


class WoeTransformer(BaseEstimator, TransformerMixin):  # Woe_IV_R_Psi_Transformer
    def __init__(self, Pcase, Ncase, to_woe, deal_0='to_1', turnpoint_limit=np.float64('inf'), iv_limit=0.02,
                 r_limit=np.float64('inf'), maxgap_limit=50, psi_limit=0.1, precision=6, warn_mark=''):
        """
        计算woe、iv取值（为DataFrame设计，而非数组）
        :param Pcase: 正例的取值
        :param Ncase: 负例的取值
        :param to_woe: True时对数据进行woe编码，False时，只计算woe、iv，但不对数据woe编码，只是为了通过iv筛选字段
        :param deal_0: 计算woe时，如何处理列联表中的零值，目前仅支持 to_1，即将零替换为1，如需其他方法可扩充
        :param turnpoint_limit: 剔除woe拐点数大于此阈值的字段，默认值为无穷大，即不做剔除
        :param iv_limit: 剔除iv值小于此阈值的字段
        :param r_limit: 剔除woe转换后，相关性系数大于此阈值的字段（删除高相关字段对中iv值低的）
        :param maxgap_limit: 删除“总样本量”与“众数样本量”之差 <= maxgap_limit 的字段，减少后续计算耗时
        :param psi_limit: PsiTransformer的字段稳定性阈值参数，借用PsiTransformer计算字段稳定性（>=阈值将发出警告）
        :param precision: Mdlp_dt的精度参数，to_woe=False时，借用Mdlp_dt对数据型字段进行离散
        :param warn_mark: 警告的开头提示字符

        """
        self.deal_0 = deal_0
        self.turnpoint_limit = turnpoint_limit
        self.iv_limit = iv_limit
        self.r_limit = r_limit
        self.maxgap_limit = maxgap_limit
        self.Pcase = Pcase
        self.Ncase = Ncase
        self.to_woe = to_woe
        self.psi_limit = psi_limit
        self.precision = precision
        self.warn_mark = warn_mark

    def _woe_iv(self, X, y, Pcase, Ncase):
        """
        计算woe、iv
        :param X: 特征DataFrame
        :param y: 目标Series
        :param Pcase: 目标Series的正例取值
        :return: dict，包括woe、iv、拐点信息
        """
        y_unique = y.unique()
        cases = Series([Pcase, Ncase], index=['Pcase', 'Ncase'])
        er = cases[~cases.isin(y_unique)]
        if len(er) > 0:
            raise Exception(f'WoeTransformer 参数y的取值为：{y_unique}，并不包括：\n{er}，请检查！')

        more = set(y_unique) - set(cases)
        if more:
            s = f'WoeTransformer 参数y的取值{y_unique}，Pcase、Ncase取值{list(cases)}， 多出取值{more}!'
            warnings.warn(s)

        col_tabl = dict()
        col_iv = Series(dtype=np.float64)
        col_turnpoint = Series(dtype=np.float64)
        for i in X.columns:
            tabl = pd.crosstab(X[i].values, y)  # .values是为了规避分类类型(cut之后)列联表的错误
            tabl.index.name = i
            if self.deal_0 == 'to_1':  # 暂未提供其他选项，可酌情扩充
                tabl[tabl == 0] = 1

            tabl_all = tabl.sum()
            tabl['woe'] = np.log(
                (tabl.loc[:, Pcase] / tabl_all.loc[Pcase]) / (tabl.loc[:, Ncase] / tabl_all.loc[Ncase]))
            tabl['weight'] = (tabl.loc[:, Pcase] / tabl_all.loc[Pcase]) - (tabl.loc[:, Ncase] / tabl_all.loc[Ncase])
            tabl['woe_weight'] = tabl.woe * tabl.weight
            iv = tabl.woe_weight.sum()
            col_tabl[i] = tabl
            col_iv.loc[i] = iv
            # 计算转折点数目
            diff_value = np.array(tabl.woe.iloc[1:]) - np.array(tabl.woe.iloc[0:-1])  # xj：np.array为了去除index
            col_turnpoint.loc[i] = (diff_value[:-1] * diff_value[1:] < 0).sum()
        return {'col_tabl': col_tabl, 'col_iv': col_iv, 'col_turnpoint': col_turnpoint}

    def _rhigh_del_fun(self, data, col_iv, thred, data_is_rmatrix=False):
        """
        按iv值递归删除高相关字段对中的低iv字段
        :param data: 待处理的数值型DataFrame, 或相关性系数矩阵
        :param col_iv: data中每个字段的iv值（Series）
        :param thred: 剔除相关系系数大于该阈值的字段
        :return: 需要删除的字段列表
        """
        col_del_r = list()
        r_matrix = None

        if not data_is_rmatrix:
            if data.shape[1] <= 1:
                print(f"数据包括{data.shape[1]}个字段，无需考察相关性")
                return (col_del_r, r_matrix)

            col_na_count = data.isnull().sum()
            col_na_count = col_na_count[col_na_count > 0]
            if len(col_na_count) > 0:
                s = f'WoeTransformer 存在缺失值：{dict(col_na_count)}'
                raise Exception(s)

            print('    计算字段之间的相关性')
            r_matrix = data.corr()

            lenght = len(r_matrix)
            column = DataFrame([range(lenght)] * lenght)
            row = column.T
            r_matrix[(column >= row).values] = np.nan

            r_matrix = r_matrix.stack()
            r_matrix = r_matrix.reset_index()
            r_matrix = r_matrix.rename(columns={0: "r"})
            r_matrix['iv_0'] = r_matrix.level_0.map(dict(col_iv))
            r_matrix['iv_1'] = r_matrix.level_1.map(dict(col_iv))
        else:
            print('    输入为相关性系数矩阵')
            r_matrix = data.copy()

        r_matrix = r_matrix.loc[r_matrix.r.abs().sort_values(ascending=False).index]
        if r_matrix.r.abs().max() < thred:
            print(f"    数据中未出现相关性系数大于{thred}的字段组合")
            return (col_del_r, r_matrix)

        print(f'    提取相关性超过{thred}的字段')
        r_matrix_high = r_matrix.loc[r_matrix.r.abs() >= thred, :].copy()
        r_matrix_high['need_deal'] = r_matrix_high.level_0.notnull() & r_matrix_high.level_1.notnull()
        r_matrix_high = r_matrix_high.loc[r_matrix_high.r.abs().sort_values(ascending=False).index]

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
        print(f'    需要删除{len(col_del_r)}个高度相关字段: {col_del_r}')
        return (col_del_r, r_matrix)



    def fit(self, X, y):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        if y is None:
            raise Exception('WoeTransformer fit函数的参数y必须设置！')

        if len(X) != len(y):
            raise Exception(f'WoeTransformer 参数X的行数({len(X)})与y的长度({len(y)})必须相等！')

        if len(y.unique()) != 2:
            raise Exception(f'WoeTransformer 目标字段的不同取值应该为2个，但实际：{len(y.unique())}个')

        col_na_count = X.isnull().sum()
        col_na_count = col_na_count[col_na_count > 0]
        if len(col_na_count) > 0:
            s = f'WoeTransformer 存在缺失值：{dict(col_na_count)}'
            raise Exception(s)

        max_gap = len(X) - X.apply(lambda x: x.value_counts().max())
        self.col_del_maxgap_ = list(max_gap[max_gap <= self.maxgap_limit].index)
        if len(self.col_del_maxgap_) > 0:
            # 此步骤也能删除取值过于集中的字段，但流水线前面的转换器可能已经过滤掉了取值过于集中的字段，此处主要用意：
            # 若数据在onehot之后进入woe转换器，且onehot了id等取值众多的类别字段，将产生大量稀疏字段，
            # 此步删除过于稀疏“字段~取值”列，减少后续计算的不必要耗时
            print(
                f"    删除“总样本量”与“众数样本量”之差<={self.maxgap_limit}的{len(self.col_del_maxgap_)}个字段:{self.col_del_maxgap_}")
            X = X.drop(columns=self.col_del_maxgap_)
            print(f"    剩余：{X.shape}")

        col_types = X.dtypes.apply(str)
        col_num = list(col_types[col_types.str.match('int|float')].index)
        if len(col_num) > 0:
            if self.to_woe:
                s = f'    WoeTransformer：to_woe=True，无法对{len(col_num)}个数值型字段做woe转换：{col_num}'
                raise Exception(s)
            else:
                print(f'    to_woe=False，数据中包括{len(col_num)}个数值型字段，离散后计算woe、iv（只计算，不进行woe编码）')
                line = '    --------------------------- woe: Mdlp_dt_DF %s -------------------------------'
                print(line % 'start')
                mdlp = Mdlp_dt_DF(precision=self.precision, print_indent=self.print_indent + '    ')
                X_num = mdlp.fit_transform(X[col_num], y)
                print(line % 'end')
                if len(col_num) < len(col_types):
                    X = pd.concat([X_num, X.drop(columns=col_num)], axis=1)
                    print('\n')
                    print(f'    合并离散后的数值字段 与 类别字段：{X.shape}')
                else:
                    X = X_num
                self.mdlp_ = mdlp

        res = self._woe_iv(X, y, self.Pcase, self.Ncase)
        self.col_tabl_, self.col_iv_, self.col_turnpoint_ = res['col_tabl'], res['col_iv'], res['col_turnpoint']

        self.col_del_turnpoint_ = list(self.col_turnpoint_[self.col_turnpoint_ >= self.turnpoint_limit].index)
        if len(self.col_del_turnpoint_) > 0:
            print(
                f"    删除woe拐点数>={self.turnpoint_limit}的{len(self.col_del_turnpoint_)}个字段：{self.col_del_turnpoint_}")
            X = X.drop(columns=self.col_del_turnpoint_)

        self.col_del_iv_ = list(self.col_iv_[self.col_iv_ < self.iv_limit].index)
        if len(self.col_del_iv_) > 0:
            print(f"    删除iv<{self.iv_limit}的{len(self.col_del_iv_)}个字段：{self.col_del_iv_}")
            X = X.drop(columns=self.col_del_iv_)

        print('    计算字段相关性（woe编码后）')
        X_woe = DataFrame()
        for i in X.columns:
            X_woe[i] = X[i].map(self.col_tabl_[i].woe)
        self.col_del_r_, self.r_matrix_ = self._rhigh_del_fun(X_woe, self.col_iv_, self.r_limit)
        if len(self.col_del_r_) > 0:
            # print(f"    计算字段相关性（woe编码后）：删除woe编码后相关性系数>={self.r_limit}的{len(self.col_del_r_)}个字段：{self.col_del_r_}")
            X = X.drop(columns=self.col_del_r_)
        # else:
        #     print(f"    数据中未出现相关性系数大于{self.r_limit}的字段组合")

        self.del_colnames_ = pd.Index(set(list(self.mdlp_.del_colnames_ if hasattr(self, 'mdlp_') else []) +
                                          self.col_del_maxgap_ + self.col_del_turnpoint_ +
                                          self.col_del_iv_ + self.col_del_r_))

        print('    计算字段稳定度')
        line = '    --------------------------- woe: PsiTransformer_DF %s -------------------------------'
        print(line % 'start')
        self.psi_ = PsiTransformer_DF(Pcase=self.Pcase, Ncase=self.Ncase, psi_limit=self.psi_limit,
                                      print_indent=self.print_indent + '    ', warn_mark=self.warn_mark)
        self.psi_.fit(X, y)
        print(line % 'end')
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        id_X_new = X.id_X_new if hasattr(X, 'id_X_new') else id(X)  # self.id_X_new 从ToDFVersion而来
        has_y_carrier = hasattr(X, 'y_carrier')
        has_data_name = hasattr(X, 'data_name')
        if has_y_carrier:
            y_carrier = X.y_carrier
        if has_data_name:
            data_name = X.data_name
        X = X.copy()

        # 先剔除字段，就不用继续加工了
        if len(self.del_colnames_) > 0:
            print(f"    剔除{len(self.del_colnames_)}个字段：{list(self.del_colnames_)}")
            X = X.drop(columns=self.del_colnames_)  # X = X.drop(columns=self.del_colnames_.intersection(X.columns))

        X_copy = X.copy()
        if hasattr(self, 'mdlp_'):
            cp = self.mdlp_.col_cutpoint_
            print(f"    对数值型字段进行离散 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            for i in cp.index.intersection(X_copy.columns):
                X_copy[i] = np.round(X_copy[i], self.precision)
                X_copy[i] = pd.cut(X_copy[i], cp[i], include_lowest=True, precision=self.precision, duplicates='drop')
                X_copy[i] = [interval_to_str(interval) for interval in X_copy[i]]
            print(f"    离散完毕 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.id_X == id_X_new:
            print('    训练集本身，不参与稳定度统计')
        else:
            print('    计算字段稳定度')
            line = '    --------------------------- woe: PsiTransformer_DF %s -------------------------------'
            print(line % 'start')
            if has_y_carrier:
                X_copy.y_carrier = y_carrier
            if has_data_name:
                X_copy.data_name = data_name
            _ = self.psi_.transform(X_copy)
            print(line % 'end')

        if self.to_woe:
            print('to_woe=True，进行woe编码')
            for i in X.columns:
                X_copy[i] = X_copy[i].map(self.col_tabl_[i].woe)
            X_out = X_copy.copy()
        else:
            print('to_woe=False，未进行woe编码')
            X_out = X.copy()

        self.transform_out_colnames_ = X_out.columns
        return X_out


# ----------------------------------------------------------------------------------------------------------------------

class PsiTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, Pcase=None, Ncase=None, psi_limit=0.1, precision=6, bins=None, fit_pop_all=False,
                 col_ignore=None, warn_mark=''):
        """
        检验字段稳定度
        保存的模型结果中：（验证集）、测试集的稳定度
        每期打分后也保存一份（带模型账期和打分账期后缀）：打分集的稳定度
        :param Pcase: 目标字段的正例取值
        :param Ncase: 目标字段的负例取值
        :param precision: cut时的精度参数
        :param psi_limit: 当字段的psi >= psi_limit 时，发出warning（注意：不会删除字段）
        :param bins: 分箱数量（只有输入数据中包括数值型字段时才起作用）
                当y为None时：
                     bins取值为数字时，表示数值字段的等频分箱数量（当数值型字段的切点只有两个时，不分箱）
                     bins取值为None时，报错
                当y不为None时：
                     bins取值为数字时，表示数值字段mdlp分箱数量的上限
                     bind取值为None时，mdlp分箱不设置上限
        :param fit_pop_all: fit阶段是否统计频数（不区分正负例）
                            fit_pop_all=False使用场景：如果训练数据经过正负例均衡抽样，非原始比例
                            fit_pop_all=True使用场景：训练数据为原始比例，并经过正负例均衡抽样
        :param col_ignore: transform时，忽略某些列的稳定度，应用场景：transform时X剔除了某些列, 以缺失值填充补齐剔除列，
                           这种列的稳定度必然超标，但是我们不关注这些列的稳定度，只关注X保留列的稳定度。
                           col_ignore为None时，不作忽略处理。
        :param warn_mark: 警告的开头提示字符
        """
        self.bins = bins
        self.Pcase = Pcase
        self.Ncase = Ncase
        self.psi_limit = psi_limit
        self.precision = precision
        self.fit_pop_all = fit_pop_all
        self.col_ignore = col_ignore
        self.warn_mark = warn_mark + ' PsiTransformer'

    # 数据量统计函数
    def value_counts_xj(self, X, y=None):
        col_pop = DataFrame()
        for i in X.columns:
            if y is None:
                col_pop_i = DataFrame(X[i].value_counts(dropna=False))
                col_pop_i.columns = ['All']
                col_pop_i.index.name = i
                col_pop_i['r_' + 'All'] = (col_pop_i['All'] / col_pop_i['All'].sum()).map(lambda x: format(x, '.1%'))
            else:
                col_pop_i = pd.crosstab(X[i].values, y.values)
                col_pop_i['All'] = col_pop_i.apply(sum, axis=1)
                for j in list(y.unique()) + ['All']:
                    col_pop_i['r_' + str(j)] = (col_pop_i[j] / col_pop_i[j].sum()).map(lambda x: format(x, '.1%'))
            index_add = [interval_to_str(i) for i in col_pop_i.index]  # astype(str)为最好，此处为兼容多版本
            col_pop_i.index = [[i] * len(col_pop_i), index_add]
            col_pop_i.index.names = ['field', 'value']
            col_pop = pd.concat([col_pop, col_pop_i])
        return col_pop

    def psi_fun(self, pop_old_total, pop_new_total, col_ignore=None, mark=''):
        """
        # 计算稳定度
        :param pop_old_total: 所有字段的用户量分布统计的Series(训练数据)
        :param pop_new_total: 所有字段的用户量分布统计的Series（新数据）
        :param col_ignore: 忽略某些字段，None则不忽略
        :return: 稳定度
        """
        pop_old_total = pop_old_total.copy()
        pop_new_total = pop_new_total.copy()

        if type(pop_old_total.index) == pd.Index:
            pop_old_total.index = [[''] * len(pop_old_total), pop_old_total.index]
        if type(pop_new_total.index) == pd.Index:
            pop_new_total.index = [[''] * len(pop_new_total), pop_new_total.index]

        if col_ignore is not None:
            pop_old_total = pop_old_total.loc[~get_index(pop_old_total, 0).isin(col_ignore).values].copy()
            pop_new_total = pop_new_total.loc[~get_index(pop_new_total, 0).isin(col_ignore).values].copy()

        psi = Series(dtype=np.float64)
        for i in get_index(pop_old_total, 0).unique():
            pop_old = pop_old_total.loc[[i], :].copy()
            pop_new = pop_new_total.loc[[i], :].copy()

            pop_old.loc[pop_old == 0] = 1
            pop_new.loc[pop_new == 0] = 1
            name_old = get_index(pop_old, 0).unique()  # pop_old.index.name
            name_new = get_index(pop_new, 0).unique()  # pop_new.index.name
            if name_old != name_new:
                raise Exception(f'{mark} 字段名称不一致：{name_old}、{name_new}')
            else:
                col_name = name_old

            pop_merge = pd.concat([pop_old, pop_new], axis=1)
            pop_merge.columns = ['old', 'new']
            more_value = set(pop_merge[pop_merge.old.isnull()].index)
            if more_value:
                pop_more = pop_merge.loc[pop_merge.old.isnull()]
                s = f'    {mark} {col_name}字段多出{len(pop_more)}个取值, .head(5):\n{pop_more.head(5)}'
                # print(s)
                warnings.warn(s)
            same_value = set(pop_merge[(pop_merge.old.notnull()) & (pop_merge.new.notnull())].index)
            if len(same_value) == 0:
                ss = f'    {mark} {col_name}字段的新数据与训练数据的取值无交集，请确认:\n{pop_merge}'
                raise Exception(ss)
            pop_merge = pop_merge.fillna(1)
            r_old = pop_merge.old / pop_merge.old.sum()
            r_new = pop_merge.new / pop_merge.new.sum()
            psi[i] = ((r_new - r_old) * np.log(r_new / r_old)).sum()
        return psi

    def fit(self, X, y):
        """
        统计X的频数分布（区分正负例）
        :param X: 特征集合的DataFrame
        :param y: 目标字段Series
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        # if y is None:
        #     s = 'PsiTransformer fit函数的参数y必须设置！'
        #     raise Exception(s)
        #### if hasattr(self, 'data_name'):  # self.data_name 从ToDFVersion而来
        ####     data_name = self.data_name
        if hasattr(X, 'data_name'):
            data_name = X.data_name
        else:
            data_name = None
        circ = '初次' if data_name is None else data_name + '（初次）'

        X_copy = X.copy()
        col_types = X_copy.dtypes.astype(str)
        col_num = list(col_types[col_types.str.contains('int|float')].index)
        if y is None:
            if col_num:
                print(f'    输入数据中包括{len(col_num)}个数值型字段，先对其进行等频分箱：{col_num}')
                if self.bins is None:
                    raise Exception('PsiTransformer y为None，X包括数值型字段，需要等频分箱，但bins为None，请设置bins！')
                cp = Series(dtype=float)
                q = np.arange(0, 1, 1 / self.bins)[1:]
                for i in col_num:
                    # col_i_prec = np.round(X_copy[i], self.precision) # #
                    if self.precision > 6:
                        p_cp = self.precision - 1
                    else:
                        p_cp = self.precision
                    cp[i] = sorted(np.unique(np.round(X_copy[i].quantile(q), p_cp)))
                    cp[i] = [np.float64('-Inf')] + cp[i] + [np.float64('Inf')]
                    X_copy[i] = pd.cut(X_copy[i], cp[i], include_lowest=True, precision=self.precision,
                                       duplicates='drop')
                self.col_cp_ = cp
            print(f'    {circ} 统计频数（不区分正负例）：self.col_pop_all_old_')
            self.col_pop_all_old_ = self.value_counts_xj(X_copy)
        else:
            if col_num:
                print(f'    输入数据中包括{len(col_num)}个数值型字段，先对其进行mdlp分箱：{col_num}')
                line = '    ---------------------------Psi: Mdlp_dt_DF %s -------------------------------'
                print(line % 'start')
                mdlp = Mdlp_dt_DF(precision=self.precision, if_del=False, print_indent=self.print_indent + '    ')
                X_num = mdlp.fit_transform(X[col_num], y)  # transform删除未离散字段（这些字段无从统计稳定度，忽略就好）
                print(line % 'end')
                if len(col_num) < len(col_types):
                    X_copy = pd.concat([X_num, X_copy.drop(columns=col_num)], axis=1)
                    print(f'    合并离散后的数值字段 与 类别字段：{X.shape}')
                else:
                    X_copy = X_num
                self.col_cp_ = mdlp.col_cutpoint_
            print(f'    {circ} 统计频数（区分正负例）：self.col_pop_PN_old_')
            self.col_pop_PN_old_ = self.value_counts_xj(X_copy, y)

            if self.fit_pop_all:
                print(f'    {circ} 统计频数（不区分正负例）：self.col_pop_all_old_')
                self.col_pop_all_old_ = self.value_counts_xj(X_copy)
            else:
                print('    不统计频数（不区分正负例）：训练数据经过抽样，非原始比例，如有需要可自行修改')
        return self

    def transform(self, X, y=None):
        """
        （只是基于输入数据做统计而已，不会对输入做任何改动，直接返回）
        :param X: 不会对X做任何处理，仅仅基于其统计数据量，计算取值稳定度
        备注：如果有目标字段y的信息，则通过X中添加y_carrier属性携带，来计算区分正负例的稳定度
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        has_y_carrier = hasattr(X, 'y_carrier')
        if has_y_carrier:
            y_carrier = X.y_carrier

        id_X_new = X.id_X_new if hasattr(X, 'id_X_new') else id(X)  # self.id_X_new 从ToDFVersion而来

        #### if hasattr(self, 'data_name'):  # self.data_name 从ToDFVersion而来
        ####     data_name = self.data_name
        if hasattr(X, 'data_name'):
            data_name = X.data_name
        else:
            data_name = None

        X_copy = X.copy()
        if hasattr(self, 'col_cp_'):
            print(f'    对{len(self.col_cp_)}个数值字段进行分箱：{list(self.col_cp_.index)}')
            for i in self.col_cp_.index:
                cp_i = self.col_cp_[i]
                X_copy.loc[X_copy[i] < cp_i[0], i] = cp_i[0]
                X_copy.loc[X_copy[i] > cp_i[-1], i] = cp_i[-1]
                X_copy[i] = pd.cut(X_copy[i], self.col_cp_[i], include_lowest=True, precision=self.precision,
                                   duplicates='drop')

        # 结果整合函数
        def res_combine(col_pop_old, col_pop_new, psi, psi_index_name, hier_index_name=['old', 'new'], col_ignore=None):
            col_pop_old = col_pop_old.copy()
            col_pop_new = col_pop_new.copy()

            if col_ignore is not None:
                col_pop_old = col_pop_old.loc[~get_index(col_pop_old, 0).isin(col_ignore).values].copy()
                col_pop_new = col_pop_new.loc[~get_index(col_pop_new, 0).isin(col_ignore).values].copy()

            res = pd.concat([col_pop_old, col_pop_new], axis=1)
            ncol_old = col_pop_old.shape[1]
            ncol_new = col_pop_new.shape[1]
            res.columns = [[hier_index_name[0]] * ncol_old + [hier_index_name[1]] * ncol_new, res.columns]
            psi_index = psi[get_index(res, 0)]
            psi_index.name = psi_index_name
            res.index = [get_index(res, 0), psi_index, get_index(res, 1)]
            return res  # .sort_index(level=1)

        if_train = (self.id_X == id_X_new) | (data_name == 'data_train' if data_name else False)
        if if_train:
            print('    训练集本身，不参与稳定度统计')

        if not if_train:  # 训练集自身无需参与此比较；X是训练集之外的数据时，才会进入下一环节
            if not hasattr(self, 'col_pop_all_old_'):  # 没有col_pop_all_old_，则初次统计（第一份新数据、原始正负例比例）
                circ = '初次' if data_name is None else data_name + '（初次）'
                print(f'    {circ} 统计频数（不区分正负例）：self.col_pop_all_old_')
                self.col_pop_all_old_ = self.value_counts_xj(X_copy)
            else:  # 已经统计过col_pop_all_old_，与最新数据比较
                if not hasattr(self, 'n1'):
                    self.n1 = 0
                self.n1 += 1
                circ = f'time{self.n1 if data_name is None else data_name}'
                mark = f'{self.warn_mark} {circ} 计算稳定性(不区分正负例)'
                print(f'    {circ} 计算稳定度（不区分正负例）：self.col_psi_all_')
                if not hasattr(self, 'col_psi_all_'):
                    self.col_psi_all_ = {}
                col_pop_all_new = self.value_counts_xj(X_copy)
                psi_all = self.psi_fun(self.col_pop_all_old_['All'], col_pop_all_new['All'], self.col_ignore, mark)
                # psi_all_high = dict(psi_all[psi_all >= self.psi_limit].sort_values(ascending=False))
                psi_all_high = psi_all[psi_all >= self.psi_limit].sort_values(ascending=False)
                # if self.col_ignore is not None:
                #     psi_all_high = psi_all_high.loc[~psi_all_high.index.isin(self.col_ignore)]
                if len(psi_all_high):
                    s = f'    {mark}，{len(psi_all_high)}个字段稳定性>=psi_limit（{self.psi_limit}）：\n{psi_all_high}'
                    # print(s)
                    warnings.warn(s)
                else:
                    s = f'        无字段稳定性>=psi_limit（{self.psi_limit}）'
                    print(s)
                psi_index_name = 'psi_all_' + circ
                self.col_psi_all_[psi_index_name] = res_combine(self.col_pop_all_old_, col_pop_all_new, psi_all,
                                                                psi_index_name, ['old', 'new'], self.col_ignore)

            # 区分正负例计算
            if hasattr(self, 'col_pop_PN_old_') & (not has_y_carrier):
                ss = f"    {self.warn_mark} {data_name if data_name else ''} X未带有y_carrier属性，不计算稳定度（区分正负例）"
                # print(ss)
                warnings.warn(ss)
            if hasattr(self, 'col_pop_PN_old_') & has_y_carrier:  # 如果新数据携带了目标变量信息，才比较正负例数据分布
                if not hasattr(self, 'n2'):
                    self.n2 = 0
                self.n2 += 1
                circ = 'time%s' % self.n2 if data_name is None else data_name
                mark = f'{self.warn_mark} {circ} 计算稳定性(区分正负例)'
                print(f'    {circ} 计算稳定度（区分正负例）：self.col_psi_PN_')
                if not hasattr(self, 'col_psi_PN_'):
                    self.col_psi_PN_ = {}
                col_pop_PN_new = self.value_counts_xj(X_copy, y_carrier)

                psi_P = self.psi_fun(self.col_pop_PN_old_[self.Pcase], col_pop_PN_new[self.Pcase], self.col_ignore,
                                     f"{mark}-正例（{self.Pcase}）")
                psi_N = self.psi_fun(self.col_pop_PN_old_[self.Ncase], col_pop_PN_new[self.Ncase], self.col_ignore,
                                     f"{mark}-负例（{self.Ncase}）")
                psi_PN = pd.concat([psi_P, psi_N], axis=1).apply(max, axis=1)
                # psi_PN_high = dict(psi_PN[psi_PN >= self.psi_limit].sort_values(ascending=False))
                psi_PN_high = psi_PN[psi_PN >= self.psi_limit].sort_values(ascending=False)
                # if self.col_ignore is not None:
                #     psi_PN_high = psi_PN_high.loc[~psi_PN_high.index.isin(self.col_ignore)]
                if len(psi_PN_high):
                    s = f'    {mark} ，{len(psi_PN_high)}个字段稳定性>=psi_limit（{self.psi_limit}）：\n{psi_PN_high}'
                    # print(s)
                    warnings.warn(s)
                else:
                    s = f'        无字段稳定性>=psi_limit（{self.psi_limit}）'
                    print(s)

                psi_index_name = 'psi_PN_' + circ
                self.col_psi_PN_[psi_index_name] = res_combine(self.col_pop_PN_old_, col_pop_PN_new, psi_PN,
                                                               psi_index_name, ['old', 'new'], self.col_ignore)
        return X


# ----------------------------------------------------------------------------------------------------------------------

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, IQR_mode=(1.5, 'assignment')):
        """
        数值型字段异常值识别与处理
        :param IQR_mode: tuple：分别为异常值判断准则（上下四分位数加减±极差的倍数）
                                      异常值处理方式（赋值assignment、或剔除delete）
                         None：不使用分位数识别及处理异常值
        备注：可根据需要扩展其他方法
        """
        self.IQR_mode = IQR_mode
        if self.IQR_mode is not None:
            if len(IQR_mode) != 2:
                raise Exception('参数IQR_mode应为长度为2的tuple')
            if type(IQR_mode[0]) not in [float, int]:
                raise ValueError("参数IQR_mode的第1个元素应为数值")
            if IQR_mode[1] not in ['assignment', 'delete']:
                raise ValueError("参数IQR_mode的第2个元素应为'assignment' 或 'delete'")
        else:
            raise Exception('未选择异常值处理方式')

    def fit(self, X, y=None):
        """
        判断数值型字段的异常值
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X_copy = X.copy()

        num = X.dtypes[X.dtypes == object]
        if len(num) > 1:
            raise Exception(f'ValueCombiner 数据集中包括非数值型字段：\n{num}')

        if self.IQR_mode is not None:
            print('    使用分位数识别异常值')
            IQR_value = self.IQR_mode[0]
            LU_limit = {}
            for i in X.columns:
                q = np.r_[np.arange(0, 1, 1 / 4), 1]
                quan = X[i].quantile(q)
                U = quan.loc[0.75]
                L = quan.loc[0.25]
                IQR = U - L
                L_limit = L - IQR_value * IQR
                U_limit = U + IQR_value * IQR
                LU_limit[i] = (L_limit, U_limit)
            self.LU_limit_ = LU_limit
        return self

    def transform(self, X, y=None):
        """
        处理异常值
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        if self.IQR_mode is not None:
            print('    处理分位数识别的异常值')
            if self.IQR_mode[1] == 'assignment':
                print('    异常值重新赋值')
                for i in self.LU_limit_.keys():
                    L_limit, U_limit = self.LU_limit_[i]
                    X.loc[X[i] < L_limit, i] = L_limit
                    X.loc[X[i] > U_limit, i] = U_limit
            elif self.IQR_mode[1] == 'delete':
                print('    剔除异常值')
                for i in self.LU_limit_.keys():
                    L_limit, U_limit = self.LU_limit_[i]
                    X = X.loc[(X[i] >= L_limit) & (X[i] <= U_limit), :]
        return X


# ----------------------------------------------------------------------------------------------------------------------


# SimpleImputer 处理缺失值
# xj 临时处理：将fit、trainform中的 X = self._validate_input(X) 替换成了 X = X.values

import numbers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
# from sklearn.utils import is_scalar_nan
from sklearn.utils import check_array
from scipy import sparse
import numpy.ma as ma
from sklearn.utils.validation import check_is_fitted
import warnings
from scipy import stats

import numpy as np

from pandas import Series


def _check_inputs_dtype(X, missing_values):
    if (X.dtype.kind in ("f", "i", "u") and
            not isinstance(missing_values, numbers.Real)):
        raise ValueError("'X' and 'missing_values' types are expected to be"
                         " both numerical. Got X.dtype={} and "
                         " type(missing_values)={}."
                         .format(X.dtype, type(missing_values)))


def is_scalar_nan(x):
    """Tests if x is NaN

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not np.float64('nan').

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    # convert from numpy.bool_ to python bool to ensure that testing
    # is_scalar_nan(x) is True does not fail.
    return bool(isinstance(x, numbers.Real) and np.isnan(x))


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            return np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            return np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            return _object_dtype_isnan(X)
    else:
        # X == value_to_mask with object dytpes does not always perform
        # element-wise for old versions of numpy
        return np.equal(X, value_to_mask)


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)

# Fix for behavior inconsistency on numpy.equal for object dtypes.
# For numpy versions < 1.13, numpy.equal tests element-wise identity of objects
# instead of equality. This fix returns the mask of NaNs in an array of
# numerical or object values for all numpy versions.
if np_version < (1, 13):
    def _object_dtype_isnan(X):
        return np.frompyfunc(lambda x: x != x, 1, 1)(X).astype(bool)
else:
    def _object_dtype_isnan(X):
        return X != X


def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
       [extra_value] * n_repeat, where extra_value is assumed to be not part
       of the array."""
    # Compute the most frequent value in array only
    if array.size > 0:
        with warnings.catch_warnings():
            # stats.mode raises a warning when input array contains objects due
            # to incapacity to detect NaNs. Irrelevant here since input array
            # has already been NaN-masked.
            warnings.simplefilter("ignore", RuntimeWarning)
            mode = stats.mode(array)

        most_frequent_value = mode[0][0]
        most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # Ties the breaks. Copy the behaviour of scipy.stats.mode
        if most_frequent_value < extra_value:
            return most_frequent_value
        else:
            return extra_value


class SimpleImputer(BaseEstimator, TransformerMixin):
    """Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    strategy : string, optional (default="mean")
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

    fill_value : string or numerical value, optional (default=None)
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is encoded as a CSR matrix;
        - If add_indicator=True.

    add_indicator : boolean, optional (default=False)
        If True, a `MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/testxj time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.

    indicator_ : :class:`sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    See also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    ... # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
            missing_values=nan, strategy='mean', verbose=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    ... # doctest: +NORMALIZE_WHITESPACE
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    Notes
    -----
    Columns which only contained missing values at `fit` are discarded upon
    `transform` if strategy is not "constant".

    """

    def __init__(self, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True, add_indicator=False):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator

    def _validate_input(self, X):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.strategy in ("most_frequent", "constant"):
            dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        try:
            X = check_array(X, accept_sparse='csc', dtype=dtype,
                            force_all_finite=force_all_finite, copy=self.copy)
        except ValueError as ve:
            if "could not convert" in str(ve):
                raise ValueError("Cannot use {0} strategy with non-numeric "
                                 "data. Received datatype :{1}."
                                 "".format(self.strategy, X.dtype.kind))
            else:
                raise ve

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("SimpleImputer does not support data with dtype "
                             "{0}. Please provide either a numeric array (with"
                             " a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        return X

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : SimpleImputer
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        columns_xj = X.columns  # xj

        X = X.values  # xj 临时处理 X = self._validate_input(X)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # fill_value should be numerical in case of numerical input
        if (self.strategy == "constant" and
                X.dtype.kind in ("i", "u", "f") and
                not isinstance(fill_value, numbers.Real)):
            raise ValueError("'fill_value'={0} is invalid. Expected a "
                             "numerical value when imputing numerical "
                             "data".format(fill_value))

        if sparse.issparse(X):
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    fill_value)
        else:
            self.statistics_ = self._dense_fit(X,
                                               self.strategy,
                                               self.missing_values,
                                               fill_value)

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values)
            self.indicator_.fit(X)
        else:
            self.indicator_ = None

        del_colnames = columns_xj[Series(self.statistics_).isnull()]  # xj
        if len(del_colnames) > 0:
            self.del_colnames_ = del_colnames
            print('    删除全空的字段：', self.del_colnames_.values)
        return self

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        mask_data = _get_mask(X.data, missing_values)
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)

        statistics = np.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            statistics.fill(fill_value)
        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i]:X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if strategy == "mean":
                    s = column.size + n_zeros
                    statistics[i] = np.nan if s == 0 else column.sum() / s

                elif strategy == "median":
                    statistics[i] = _get_median(column,
                                                n_zeros)

                elif strategy == "most_frequent":
                    statistics[i] = _most_frequent(column,
                                                   0,
                                                   n_zeros)
        return statistics

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # scipy.stats.mstats.mode cannot be used because it will no work
            # properly if the first element is masked and if its frequency
            # is equal to the frequency of the most frequent valid element
            # See https://github.com/scipy/scipy/issues/2636

            # To be able access the elements by columns
            X = X.transpose()
            mask = mask.transpose()

            if X.dtype.kind == "O":
                most_frequent = np.empty(X.shape[0], dtype=object)
            else:
                most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(np.bool)
                row = row[row_mask]
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant
        elif strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            return np.full(X.shape[1], fill_value, dtype=X.dtype)

    def transform(self, X, y=None):  ##
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        columns_xj = X.columns  # xj

        check_is_fitted(self, 'statistics_')

        X = X.values  # xj 临时处理 X = self._validate_input(X)

        statistics = self.statistics_

        self.transform_out_colnames_ = columns_xj[Series(self.statistics_).notnull()]  # xj
        if hasattr(self, 'del_colnames_') > 0:
            print('    剔除全空字段：', self.del_colnames_.values)

        if X.shape[1] != statistics.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features per sample, expected {self.statistics_.shape[0]}")

        if self.add_indicator:
            X_trans_indicator = self.indicator_.transform(X)

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
        else:
            # same as np.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, np.nan)
            valid_mask = np.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if invalid_mask.any():
                missing = np.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn("Deleting features without "
                                  "observed values: %s" % missing)
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sparse.issparse(X):
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                mask = _get_mask(X.data, self.missing_values)
                indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                    np.diff(X.indptr))[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                                copy=False)
        else:
            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask.transpose())[::-1]

            X[coordinates] = values

        if self.add_indicator:
            hstack = sparse.hstack if sparse.issparse(X) else np.hstack
            X = hstack((X, X_trans_indicator))
        return X

    def _more_tags(self):
        return {'allow_nan': True}


# ----------------------------------------------------------------------------------------------------------------------

# 类别型字段的编码
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
    The type of encoding to use (default is 'onehot'):
    - 'onehot': encode the features using a one-hot aka one-of-K scheme
    (or also called 'dummy' encoding). This creates a binary column for
    each category and returns a sparse matrix.
    - 'onehot-dense': the same as 'onehot' but returns a dense array
    instead of a sparse matrix.
    - 'ordinal': encode the features as ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
    Categories (unique values) per feature:
    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
    column. The passed categories are sorted before encoding the data
    (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
    Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
    Whether to raise an error or ignore if a unknown categorical feature is
    present during transform (default is to raise). When this is parameter
    is set to 'ignore' and an unknown category is encountered during
    transform, the resulting one-hot encoded columns for this feature
    will be all zeros.
    Ignoring unknown categories is not supported for
    ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
    The categories of each feature determined during fitting. When
    categories were specified manually, this holds the sorted categories
    (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    # >>> from sklearn.preprocessing import CategoricalEncoder
    # >>> enc = CategoricalEncoder(handle_unknown='ignore')
    # >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
    encoding='onehot', handle_unknown='ignore')
    # >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1., 0., 0., 1., 0., 0., 1., 0., 0.],
    [ 0., 1., 1., 0., 0., 0., 0., 0., 0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
    integer ordinal features. The ``OneHotEncoder assumes`` that input
    features take on values in the range ``[0, max(feature)]`` instead of
    using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
    dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
    encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error', valueiv_limit=None, Pcase=None, Ncase=None, toobject_xj=False):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.valueiv_limit = valueiv_limit  # 将onehot后 “字段~取值”列的iv<=valueiv_limit的取值替换为 “其他_countlow”
        self.Pcase = Pcase
        self.Ncase = Ncase
        self.toobject_xj = toobject_xj  # 为了woe转换器的类别识别（不会将onehot后的字段视作数值字段）

        if self.valueiv_limit is not None:
            s = ''
            if self.Pcase is None:
                s += '，Pcase参数不能为None'
            if self.Ncase is None:
                s += '，Ncase参数不能为None'
            if s:
                s = 'CategoricalEncoder: valueiv_limit参数不为None时' + s
                raise Exception(s)
        elif self.valueiv_limit is None:
            s = ''
            if self.Pcase is not None:
                s += 'Pcase参数无效'
            if self.Ncase is None:
                s += '，Ncase参数无效'
            if s:
                s = 'CategoricalEncoder: valueiv_limit参数为None时，' + s
                warnings.warn(s)

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
        The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        if self.valueiv_limit is not None:
            if y is None:
                s = 'CategoricalEncoder: valueiv_limit参数不为None时，y参数不能为None！'
                raise Exception(s)

        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        # for i in X.columns:
        #     if X[i].dtype != object:  #
        #         X[i] = X[i].astype(str)
        col_num = X.columns[X.dtypes != object]
        if len(col_num) > 0:
            s = f'数据中存在{len(col_num)}个数值型字段，请确认！: {list(col_num)}'
            print(f'    {s}')
            warnings.warn(f'CategoricalEncoder: {s}')

        self.convert_ = {}
        if self.valueiv_limit is not None:
            print(f'    合并字段中onehot后iv <= {self.valueiv_limit}的取值（替换为“其他_ivlow”）')
            # to_woe取值无无作用所谓，对_woe_iv函数
            woe_iv_fun = WoeTransformer_DF(Pcase=self.Pcase, Ncase=self.Ncase, to_woe=True)._woe_iv
            col_del_valueiv = []
            self.value_iv_ = {}
            for i in X.columns:
                value_iv = woe_iv_fun(pd.get_dummies(X[i]), y, self.Pcase, self.Ncase)['col_iv']
                self.value_iv_[i] = value_iv
                value_low = value_iv[value_iv <= self.valueiv_limit].index

                if len(value_low) == len(value_iv):  # 某类别所有“字段~取值”都过小，则删除该字段
                    col_del_valueiv.append(i)
                elif len(value_low) > 0:
                    self.convert_[i] = Series('其他_ivlow', index=value_low)
                    X[i] = X[i].map(lambda x: self.convert_[i][x] if x in self.convert_[i].keys() else x)

            if len(col_del_valueiv) > 0:
                self.col_del_valueiv_ = col_del_valueiv
                print(
                    f'    删除{len(self.col_del_valueiv_)}个所有取值iv都 <= {self.valueiv_limit}的字段：{self.col_del_valueiv_}')
                X = X.drop(columns=self.col_del_valueiv_)
                self.fit_in_colnames_init = self.fit_in_colnames_
                self.fit_in_colnames_ = X.columns  # 为了 比较 X.shape[1] 与 len(self.fit_in_colnames_)

        if X.shape[1] == 0:
            return self

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X, y=None):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
        The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
        Transformed input.
        """
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        # for i in X.columns:
        #     if X[i].dtype != object:
        #         X[i] = X[i].astype(str)

        if hasattr(self, 'col_del_valueiv_'):
            d = set(self.col_del_valueiv_) & set(X.columns)
            if d:
                print(f'    删除{len(d)}个字段：{d}')
                X = X.drop(columns=d)

        if len(self.convert_) > 0:
            print(f"    合并{len(self.convert_)}个字段中onehot后iv <= {self.valueiv_limit}的取值（替换为“其他_ivlow”）")
            for i in self.convert_.keys():
                X[i] = X[i].map(lambda x: self.convert_[i][x] if x in self.convert_[i].keys() else x)

        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)
        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            X_xj = X_int.astype(self.dtype, copy=False)
            if self.toobject_xj:
                X_xj = X_xj.astype(int).astype(object)
            return X_xj

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            X_xj = out.toarray()
            if self.toobject_xj:
                X_xj = X_xj.astype(int).astype(object)
            return X_xj
        else:
            X_xj = out
            if self.toobject_xj:
                X_xj = X_xj.astype(int).astype(object)
            return X_xj


# ----------------------------------------------------------------------------------------------------------------------
# class ColFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, col_del=None, col_remain=None):
#         if (col_del is not None) & (col_remain is not None):
#             raise Exception('参数col_del与col_remain不可以同时设置')
#         self.col_del = col_del
#         self.col_remain = col_remain
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if self.col_del is not None:
#             print('删除col_del字段列表')
#             X_ = X.drop(self.col_del, axis=1)
#         elif self.col_remain is not None:
#             print('保留col_remain字段列表')
#             X_ = X[self.col_remain]
#         else:
#             X_ = X
#         self.transform_out_colnames_ = X_.columns
#         return X_
#
#
# # --------------------------------------------------------------------------------------------------------------------
#
# class NochangeTranformer(BaseEstimator, TransformerMixin):
#     def __init__(self, useless=None):
#         """
#         此转换器什么都不做，输出 = 输入
#         """
#         self.useless = useless
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         self.transform_out_colnames_ = X.columns  # 提前设定免，否则进入DF转换环节会报错
#         return X


# ----------------------------------------------------------------------------------------------------------------------


class ObjectBacktoInt(BaseEstimator, TransformerMixin):
    def __init__(self, useless=None):
        self.useless = useless

    def fit(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()

        self.col_toint_ = []
        col_char = X.dtypes[X.dtypes.astype(str) == 'object'].index
        for i in col_char:
            type_unique = X[i].map(lambda x: type(x)).unique()
            if (len(type_unique) == 1) & (type_unique[0] == int):
                self.col_toint_.append(i)
        if self.col_toint_:
            print(f'    提取实际取值为int类型的{len(self.col_toint_)}个obejct类型字段名：{self.col_toint_}')
        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        if self.col_toint_:
            print(f'    将下列实际取值为int类型的{len(self.col_toint_)}个obejct类型字段恢复至int类型：{self.col_toint_}')
            for i in self.col_toint_:
                X[i] = X[i].astype(int)
        return X


# ----------------------------------------------------------------------------------------------------------------------
class ValueConverter(BaseEstimator, TransformerMixin):
    def __init__(self, bin_limit=np.float64('Inf'), bin_ceiling=None, valuecount_limit=None):
        """
        字段取值较多时，将样本数较少的取值合并
        :param bin_limit: 将字段的取值个数合并至bin_limit个以下
        :param bin_ceiling: 1. 字段取值个数大于bin_ceiling的不处理，取值过多的字段无合并的必要，如id类字段，
                               这样的字段保留较多的取值个数进入FeaturePrefilter，在那里可能需要被删除，
                               若合并取值，进入FeaturePrefilter后，该字段可能不会被删除，最终可能导致无意义的字段入模
                             2. None时不做限制， 只要字段取值个数超过bin_limit就合并
        :param valuecount_limit: 1. 合并样本数小于valuecount_limit的取值，既保证字段的取值个数小于bin_limit，
                                    又使得每个字段的取值样本数不至于太多，以免进行onehot后，某些“字段~取值”列中1太少。
                                 2. None时不做限制，只要取值个数合并到bin_limit以内就可以了
        """
        self.bin_limit = bin_limit
        self.bin_ceiling = bin_ceiling
        self.valuecount_limit = valuecount_limit

    def fit(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        nrow = len(X)
        if self.bin_ceiling is None:
            self.bin_ceiling = np.min([5000, nrow * 0.5])
            print(f"    bin_ceiling=None, 按np.min([5000, nrow * 0.5])计算其默认值为{self.bin_ceiling}")
        if self.valuecount_limit is None:
            self.valuecount_limit = np.min([100, nrow / 50])
            print(f"    valuecount_limit=None, 按 np.min([100, nrow / 50])计算其默认值为{self.valuecount_limit}")
        print(
            f'    参数取值：bin_limit({self.bin_limit})、bin_ceiling({self.bin_ceiling})、valuecount_limit({self.valuecount_limit})')
        d_types = X.dtypes.apply(str)
        col_num = d_types[d_types.str.contains('int|float')].index
        if len(col_num) > 0:  # 操作不可用于数值型字段
            s = 'ValueConverter 不能包括数值型字段：%s', list(col_num)
            raise Exception(s)

        if self.bin_limit > self.bin_ceiling:
            raise Exception(f'ValueConverter bin_limit({self.bin_limit,})应小于等于bin_ceiling({self.bin_ceiling})')

        self.convert_ = {}
        if self.bin_limit == np.float64('Inf'):
            print('    bin_limit设置为Inf，不处理取值')
        else:
            for i in X.columns:
                v_count = X[i].value_counts(dropna=False)
                if len(v_count) > self.bin_ceiling:
                    print(f'    {i}字段取值个数({len(v_count)})大于bin_ceiling({self.bin_ceiling})，不予合并')
                elif len(v_count) > self.bin_limit:
                    n_less = int(np.max([len(v_count) - self.bin_limit + 1, (v_count < self.valuecount_limit).sum()]))
                    c_less = v_count.sort_values().iloc[0:n_less]
                    self.convert_[i] = dict(Series(['其他'] * len(c_less), index=c_less.index))
                    print(
                        f'    {i}字段取值个数({len(v_count)}),其中{len(self.convert_[i])}个取值(计数范围{c_less.min()}-{c_less.max()})将替换为‘其他’：{self.convert_[i].keys()}')

        return self

    def transform(self, X, y=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()
        X = X.copy()
        if len(self.convert_) == 0:
            print('    convert_为空，不做任何处理')
        else:
            for i in self.convert_.keys():
                print(f'    替换{i}字段取值')
                X[i] = X[i].map(lambda x: self.convert_[i][x] if x in self.convert_[i].keys() else x)
                # v_count_new = X[i].value_counts(dropna=False)
        return X


# ----------------------------------------------------------------------------------------------------------------------
def ToDFVersion(CLASS):
    """
    生成待执行的字符串，用于将已有估计器改造为基于DataFrame、Series的版本，并添加必要信息记录与输出
    :param CLASS: 待转换的类
    :return: 该类基于DataFrame和Series的版本
    备注：self.id_X 记录训练流水线时第一次fit时X的id
          X.id_X_new 记录流水线transform时首次transform时X的id
    """

    def get_fun_structure(fun):
        """
        获取函数的参数构造
        :param fun: 待处理的函数
        :return: 函数的参数构造（str）
        """
        lines = inspect.getsource(fun)
        lines = lines.replace(' ', '').replace('\n', '').replace('):', '):\n')
        lines = re.sub('\)->*.*?:', '):\n', lines)
        lines = re.sub('^@.*def__init__', 'def__init__', lines)
        m = re.search('def__init__\\(self,.*\\):', lines)
        if m:
            structure = m.group()
            structure = structure.replace('def__init__(', '').replace(r'):', '').replace(',', ', ')
            re.sub('def__init__\(|\).*$', '', structure).replace(',', ', ')
            k = re.search(', \\*\\*kwargs:Any|, \\*\\*kwargs', structure)
            if k:
                kwargs = k.group()
                structure = structure.replace(kwargs, '')
            else:
                kwargs = ''
        else:
            raise Exception('get_fun_structure 未识别到 __init__的源码')
        structure = re.sub(', *$', '', structure)  # 兼容版本
        return (structure, kwargs)

    CLASS_init = CLASS.__init__
    param_value, kwargs = get_fun_structure(CLASS_init)

    # param = inspect.getfullargspec(CLASS_init).args[1:] 方式不适用于所有版本，以下为兼容方式
    first_eq = param_value.index('=')  # 第一个等号
    cut_comma = Series(list(param_value))  # 第一个等号前的逗号，在此之前的参数无默认值，在此之后的参数有默认参数
    cut_comma = cut_comma[(cut_comma.index < first_eq) & (cut_comma == ',')].index.max()
    param_value_before = param_value[:cut_comma]
    param_value_after = param_value[(cut_comma + 1):]
    param_before = [i.strip(' ') for i in param_value_before.split(',') if i.strip() not in ('self', '*')]
    param_after = [re.sub('=.*$', '', re.sub('.* ', '', re.sub(':.*=', '', i))).strip(',| ') for i in
                   re.findall(',.*?=', ',' + param_value_after)]  # '.* |:.*=|=.*'
    param = param_before + param_after
    param_value2 = ', '.join([f"{i}={i}" for i in param])

    # 待执行的改造字符串
    create_str = """
class CLASS_DF(CLASS):
    def __init__(#param_value#, trans_na_error=True, print_indent=''):
        #param trans_na_error: transform后的数据如果存在缺失值，若trans_na_error=True则报错，否则仅通过print打印
        super().__init__(#param_value2#)
        self.trans_na_error = trans_na_error
        self.print_indent = print_indent
        self.__name__ = re.sub("^.*?\.|'>$", '', str(self.__class__))


    def print(self, indent=False, *args, **kwargs):
        # indent为True，打印时缩进4个空格，为Fasle时，正常打印不缩进；indent为str时，打印时开头带有indent
        def print_init(*args, **kwargs):
            builtins.print(*args, **kwargs)
        def print_ad1(*args, **kwargs):
            builtins.print('    ', *args, **kwargs)
        def print_ad2(*args, **kwargs):
            builtins.print(indent, *args, **kwargs)
        if indent == True:
            return print_ad1
        elif type(indent) == str:
            return print_ad2
        else:
            return print_init


    # 标记数据输入、输出的基本情况，零列时用汉字特别指出，以免未察觉
    def print_data_in_out(self, X, method, X2=None):
        if hasattr(self, 'print_indent'):
            print = self.print(indent=self.print_indent)
        else:
            print = self.print()  
        if method == 'in':
            label1 = '    data in :'
            label2 = '零数据进入！'
        elif method == 'out':
            label1 = '    data out:'
            label2 = '零数据输出！'

        def print_0_ornot(X, label1=label1, label2=label2):
            if X.shape[-1] == 0:
                print('%s %s' % (label1, label2))
            else:
                print('%s %s' % (label1, X.shape))

        if X2 is not None:  # 在transform方法中，先按fit阶段的字段做筛选
            if X.shape != X2.shape:
                print_0_ornot(X2, label1='    data in2:')
        else:
            print_0_ornot(X)


    if hasattr(CLASS, 'fit'):
        def fit(self, X, y=None, *args, **kwargs):
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)  
            else:
                print = self.print()
            print(''); print(f"my {self.__name__ } fit")

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame,但：%s' % (CLASS.__name__, type(X).__name__))

            if y is not None:
                if not isinstance(y, Series):
                    raise Exception('%s 参数y应该为Series，但：%s' % (CLASS.__name__, type(y).__name__))

            if hasattr(self, 'tran_chain') & hasattr(self, 'tran_chain_expect'):
                if (self.tran_chain == self.tran_chain_expect) & (len(self.fit_in_index_) == len(X.index)) & (len(self.fit_in_colnames_) == len(X.columns)):
                    if all(self.fit_in_index_ == X.index) & all(self.fit_in_colnames_ == X.columns):
                        print(f'    已fit过，跳过（已fit过：{self.tran_chain}）')
                        return self

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            # xjxj  # 该属性只在首次记录一次，在流水线中保持初始id不变
            has_id_X_self = hasattr(self, 'id_X')
            has_id_X_X = hasattr(X, 'id_X')
            if has_id_X_self:
                print('    self已带有id_X属性 %s' % self.id_X)
                pass
            elif has_id_X_X:
                self.id_X = X.id_X
                print('    X参数已带有id_X属性 %s' % self.id_X)
            else:
                self.id_X = id(X)
                print('    首次从X获取id_X属性 %s' % self.id_X)
            # xjxj

            tran_name = display_classname(str(self))
            if hasattr(X, 'tran_chain'): 
                self.tran_chain = f"{X.tran_chain} ==> {tran_name}"
            elif hasattr(self, 'tran_chain_last'):
                self.tran_chain = f"{self.tran_chain_last} ==> {tran_name}"
            elif not isinstance(self, (sklearn.pipeline.Pipeline, sklearn.pipeline.FeatureUnion)):
                self.tran_chain = tran_name
            else:
                self.tran_chain = self.__name__
            print(f'    fit轨迹：{self.tran_chain}')

            self.fit_in_colnames_ = X.columns
            self.fit_in_index_ = X.index

            if X.shape[1] == 0:  # 修正：原生接口零列数据进去会报错
                return self  # 此处退出函数，避免运行super().fit产生的报错

            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(enctry)
            super().fit(X, y, *args, **kwargs)  # super().fit(X, *args, **kwargs)

            if isinstance(self, sklearn.pipeline.Pipeline): 
                self.tran_chain = self.steps[-1][1].tran_chain
            if isinstance(self, sklearn.pipeline.FeatureUnion):
                self.tran_chain = '[' + ' + '.join([i[1].tran_chain for i in self.transformer_list]) + ']'

            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(dectry)

            if isinstance(self, sklearn.pipeline.Pipeline):  # 为了记录Pipeline中各个转换器的信息
                steps = OrderedDict(self.steps)
                step_names = list(steps.keys())
                pipe_in_colnames = OrderedDict()
                pipe_del_colnames = OrderedDict()
                del_colnames = list()
                for i in step_names:
                    pipe_in_colnames[i] = steps[i].fit_in_colnames_
                    if hasattr(steps[i], 'del_colnames_'):
                        pipe_del_colnames[i] = steps[i].del_colnames_
                        del_colnames.extend(steps[i].del_colnames_)

                self.pipe_in_colnames_ = pipe_in_colnames

                if len(pipe_del_colnames) > 0:
                    self.pipe_del_colnames_ = pipe_del_colnames
                    self.del_colnames_ = del_colnames

            elif isinstance(self, sklearn.pipeline.FeatureUnion):  # 为了记录FeatureUnion中各个Pipeline的信息
                featureunion_in_colnames = OrderedDict()
                featureunion_del_colnames = OrderedDict()
                del_colnames = list()
                for trans in self.transformer_list:
                    pipeline_name = trans[0]
                    featureunion_in_colnames[pipeline_name] = trans[1].pipe_in_colnames_
                    if hasattr(trans[1], 'del_colnames_'):
                        featureunion_del_colnames[pipeline_name] = trans[1].del_colnames_
                        del_colnames.extend(trans[1].del_colnames_)

                self.featureunion_in_colnames_ = featureunion_in_colnames

                if len(featureunion_del_colnames) > 0:
                    self.featureunion_del_colnames_ = featureunion_del_colnames
                    self.del_colnames_ = del_colnames

            if isinstance(self, sklearn.pipeline.Pipeline):
                print('')
                mark = f'{self.__name__} fit '
            elif isinstance(self, sklearn.pipeline.FeatureUnion):
                print('')
                mark = f'{self.__name__} fit '
            else:
                mark = ''
            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print("    %s结束时间：" % mark + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return self

    if hasattr(CLASS, 'predict_proba'):
        # 用模型fit阶段字段限定
        def predict_proba(self, X, *args, **kwargs):
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)
            else:
                print = self.print()
            print(''); print(f'my {self.__name__} predict_proba')

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame，但：%s' % (CLASS.__name__, type(X).__name__))

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            X = X.copy()
            self.print_data_in_out(X, 'in', X[self.fit_in_colnames_])

            X = X[self.fit_in_colnames_]

            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(enctry)
            res = super().predict_proba(X, *args, **kwargs)
            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(dectry)

            if hasattr(self, 'classes_'):
                res = DataFrame(res, columns=self.classes_)
            self.print_data_in_out(res, 'out')

            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print("    结束时间：" + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return res


    if hasattr(CLASS, 'predict'):
        # 用模型fit阶段字段限定
        def predict(self, X, *args, **kwargs):
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)
            else:
                print = self.print()
            print(''); print(f'my {self.__name__} predict')

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame，但：%s' % (CLASS.__name__, type(X).__name__))

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            X = X.copy()
            self.print_data_in_out(X, 'in', X[self.fit_in_colnames_])

            X = X[self.fit_in_colnames_]

            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(enctry)
            res = super().predict(X, *args, **kwargs)
            if CLASS.__name__ in ['XGBClassifier', 'LGBMClassifier']:
                X.columns = X.columns.map(dectry)

            self.print_data_in_out(res, 'out')

            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print("    结束时间：" + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return res


    if hasattr(CLASS, 'decision_function'):
        # 用模型fit阶段字段限定
        def decision_function(self, X, *args, **kwargs):
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)
            else:
                print = self.print()
            print(''); print(f'my {self.__name__} decision_function')

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame，但：%s' % (CLASS.__name__, type(X).__name__))

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            X = X.copy()
            self.print_data_in_out(X, 'in', X[self.fit_in_colnames_])

            X = X[self.fit_in_colnames_]
            res = super().decision_function(X, *args, **kwargs)
            self.print_data_in_out(res, 'out')

            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print("    结束时间：" + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return res


    if hasattr(CLASS, 'transform'):
        def transform(self, X, y=None): ##
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)
            else:
                print = self.print()
            print(''); print(f'my {self.__name__} transform')

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame，但：%s' % (CLASS.__name__, type(X).__name__))

            if y is not None:
                if not isinstance(y, Series):
                    raise Exception('%s 参数y应该为Series，但：%s' % (CLASS.__name__, type(y).__name__))

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            self.idx_in = X.index

            has_y_carrier = hasattr(X, 'y_carrier')
            if has_y_carrier:
                y_carrier = X.y_carrier

            has_tran_chain = hasattr(X, 'tran_chain')
            if has_tran_chain:
                tran_chain = X.tran_chain


            # 该属性只在首次记录一次，在流水线中保持初始id不变
            has_id_X_self = hasattr(self, 'id_X')
            has_id_X_X = hasattr(X, 'id_X')
            if has_id_X_self:
                print('    self已带有id_X属性 %s' % self.id_X)
                pass
            elif has_id_X_X:
                self.id_X = X.id_X  # self
                print('    X参数已带有id_X属性 %s' % self.id_X)
            else:
                self.id_X = id(X) # self
                print('    首次从X获取id_X属性 %s' % self.id_X)

            if hasattr(X, 'id_X_new'):
                id_X_new = X.id_X_new
                print('    X参数已带有id_X_new属性 %s' % id_X_new)
            else:
                id_X_new = id(X)
                print('    首次从X获取id_X_new属性 %s' % id_X_new)
            self.id_X_new = id_X_new

            has_data_name = hasattr(X, 'data_name')
            # self.has_data_name = has_data_name
            if has_data_name:
                data_name = X.data_name

            X = X.copy()
            self.print_data_in_out(X, 'in', X[self.fit_in_colnames_])
            X = X[self.fit_in_colnames_]

            # if len(self.fit_in_colnames_) == 0:  # 对应fit的修正
            #     print('    data out: 零列数据输出！')
            #     self.transform_out_colnames_ = X.columns
            #     if has_y_carrier:
            #         X.y_carrier = y_carrier
            #     return X

            if has_y_carrier:
                X.y_carrier = y_carrier
            if has_data_name:
                X.data_name = data_name
            X.id_X_new =  id_X_new

            if isinstance(self, sklearn.pipeline.FeatureUnion):
                res_X = DataFrame()
                tran_union = []
                for name, pipe in self.transformer_list:
                    res_pipe = pipe.transform(X)
                    tran_union.append(res_pipe.tran_chain)
                    res_X = pd.concat([res_X, res_pipe], axis=1)
                col_c = res_X.columns.value_counts()
                del_no = []
                col_moreone = list(col_c[col_c>1].index)
                if col_moreone:
                    # 产生重复字段场景：输入数据同时存在类别字段本身以及类别字段-取值的onehot字段，即charcol、charcol~value
                    # 经过onehot后，即charcol又产生了charcol~value，与输入数据中原本的该字段重复，保留其中一个字段即可
                    for c in col_moreone:
                        del_no.append(np.arange(res_X.shape[1])[res_X.columns == c].max())
                    ncol1 = res_X.shape[1]
                    res_X = res_X.iloc[:, [i for i in np.arange(res_X.shape[1]) if i not in del_no]]
                    s = f'    FeatureUnion 下列字段重复，仅保留前一列 {ncol1}列 => {res_X.shape[1]}列：{col_moreone}'
                    warnings.warn(s)
                X = res_X

            else:
                if len(self.fit_in_colnames_) == 0:  # 对应fit的修正
                    X = X
                else:
                    X = super().transform(X)

            # 获取输出的字段名，并转换为DataFrame
            if sparse.issparse(X):
                print('    输出稀疏矩阵')  # 暂未处理稀疏矩阵
            else:
                if hasattr(self, 'transform_out_colnames_'):
                    X = DataFrame(X, columns=self.transform_out_colnames_)
                elif isinstance(self, sklearn.pipeline.Pipeline):
                    steps = OrderedDict(self.steps)  # self.named_steps乱序，会得到非预期结果
                    step_names = list(steps.keys())
                    pipe_out_colnames = OrderedDict()

                    for i in step_names:
                        pipe_out_colnames[i] = steps[i].transform_out_colnames_

                    self.pipe_out_colnames_ = pipe_out_colnames
                    X = DataFrame(X, columns=self.pipe_out_colnames_[step_names[-1]])
                    self.transform_out_colnames_ = X.columns

                elif isinstance(self, sklearn.pipeline.FeatureUnion):
                    featureunion_out_colnames = dict()
                    feature_names_list = list()
                    for trans in self.transformer_list:
                        pipeline_name = trans[0]
                        pipeline_feature_names = trans[1].transform_out_colnames_
                        feature_names_list.extend(list(pipeline_feature_names))
                        featureunion_out_colnames[pipeline_name] = trans[1].pipe_out_colnames_

                    self.feature_names_list = feature_names_list
                    # X = DataFrame(X, columns=feature_names_list)
                    self.transform_out_colnames_ = X.columns
                    self.featureunion_out_colnames_ = featureunion_out_colnames

                else:  # Pipeline、FeatureUnion之外的情况
                    if X.shape[1] == len(self.fit_in_colnames_):
                        X = DataFrame(X, columns=self.fit_in_colnames_)
                        self.transform_out_colnames_ = X.columns

                    elif X.shape[1] > len(self.fit_in_colnames_):
                        if hasattr(self, 'get_feature_names'):  # onehot、poly 具备get_feature_names方法，且形如'x0...','x1...'
                            import re
                            pattern = re.compile("x\d+")
                            out_colnames = Series(self.get_feature_names())
                            pats = out_colnames.str.findall(pattern)
                            pats = list(set(sum(list(pats.values), [])))  # 扁平化,去重
                            pats = Series(pats)
                            if len(pats) == 0:
                                raise Exception('%s的get_feature_names方法返回的对象并非形如xn，此处需进一步改造！' % CLASS.__name__)

                            index = pats.apply(lambda x: x.replace('x', '')).astype(int)
                            col_name_head = self.fit_in_colnames_[index]

                            for i in index:
                                pat = pats.iloc[i]
                                repl = col_name_head[i] + '_' + pat
                                out_colnames = out_colnames.str.replace(pat, repl)

                            X = DataFrame(X, columns=out_colnames)
                            self.transform_out_colnames_ = X.columns
                        elif hasattr(self, 'fit_in_colnames_') & hasattr(self, 'categories_'):
                            out_colnames = list()
                            for i in range(len(self.categories_)):
                                out_colnames.extend(self.fit_in_colnames_[i] + '~' + self.categories_[i])
                            X = DataFrame(X, columns=out_colnames)
                            self.transform_out_colnames_ = X.columns
                        else:
                            print(f'    {CLASS.__name__} 没有get_feature_names方法，此处需进一步改造！')
                    elif X.shape[1] < len(self.fit_in_colnames_):  # feature_selection
                        print('    此处暂未用到，待开发')

            col_na_count = X.isnull().sum()
            col_na_count = col_na_count[col_na_count>0]
            if len(col_na_count) > 0:
                s = f'    {type(self).__name__} transform后{len(col_na_count)}个字段存在缺失值：{dict(col_na_count)}'  
                if self.trans_na_error:
                    raise Exception(s)
                else:
                    print(s)          

            if has_y_carrier:
                X.y_carrier = y_carrier

            if has_data_name:
                X.data_name = data_name

            X.id_X = self.id_X
            X.id_X_new = id_X_new
            X.index = self.idx_in

            # if isinstance(self, sklearn.pipeline.Pipeline): 
            #     X.tran_chain = self.steps[-1][1].tran_chain
            # else:
            #     X.tran_chain = self.tran_chain

            if isinstance(self, sklearn.pipeline.Pipeline):
                print('')
                mark = f'{self.__name__} transform '
                # X.tran_chain = self.steps[-1][1].tran_chain
            elif isinstance(self, sklearn.pipeline.FeatureUnion):
                print('')
                mark = f'{self.__name__} transform '
                # X.tran_chain = '[' + ' + '.join(tran_union) + ']'
            else:
                mark = ''
                # X.tran_chain = self.tran_chain
            X.tran_chain = self.tran_chain
            self.print_data_in_out(X, 'out')
            print(f'    transform轨迹：{X.tran_chain}')
            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print("    %s结束时间：" % mark + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return X


    if hasattr(CLASS, 'fit_transform'):  # 舍弃了某些转换器优化过的fit_transform
        def fit_transform(self, X, y=None, **fit_params):
            if hasattr(self, 'print_indent'):
                print = self.print(indent=self.print_indent)
            else:
                print = self.print()
            s = f'my {self.__name__} fit_transform'
            print(''); print(s)

            if not isinstance(X, DataFrame):
                raise Exception('%s 参数X应该为DataFrame，但：%s' % (CLASS.__name__, type(X).__name__))

            if y is not None:
                if not isinstance(y, Series):
                    raise Exception('%s 参数y应该为Series，但：%s' % (CLASS.__name__, type(y).__name__))

            start_time = datetime.datetime.now()
            print("    开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.print_data_in_out(X, 'in')

            ##self.fit_in_colnames_ = X.columns

            # res = super().fit_transform(X, *args, **kwargs)
            print_bak_indent = self.print_indent if self.print_indent else ''
            if y is None:
                self.print_indent = print_bak_indent + '    '
                # fit method of arity 1 (unsupervised transformation)
                self.fit(X, **fit_params)
                res = self.transform(X)
                self.print_indent = print_bak_indent
            else:
                self.print_indent = print_bak_indent + '    '
                # fit method of arity 2 (supervised transformation)
                self.fit(X, y, **fit_params)
                res = self.transform(X)
                self.print_indent = print_bak_indent

            end_time = datetime.datetime.now()
            time_cost = (end_time - start_time).seconds
            print(''); print("    %s 结束时间：" % s + end_time.strftime('%Y-%m-%d %H:%M:%S') + " 耗时(%ss)" % time_cost)
            self.time_cost_ = time_cost
            return res"""
    create_str = create_str.replace('CLASS', CLASS.__name__). \
        replace('#param_value#', param_value). \
        replace('#param_value2#', param_value2). \
        replace('#kwargs#', kwargs)

    # 临时处理
    # create_str_tmp = create_str.replace(', trans_na_error=True, print_indent=None):', '):'). \
    #     replace('self.trans_na_error = trans_na_error', 'self.trans_na_error = True'). \
    #     replace('self.print_indent = print_indent', '')
    create_str_tmp = create_str.replace(', trans_na_error=True', '') \
        .replace('self.trans_na_error = trans_na_error', 'self.trans_na_error = True')
    if CLASS.__name__ in ['Pipeline', 'FeatureUnion']:  # 为了兼容多个版本之间的差异
        # v = int(sklearn.__version__[2:4])
        # if v >= 24:
        s_new = f"""self.print_indent = verbose
        self.verbose = None
        warnings.warn('{CLASS.__name__}_DF：verbose参数值被赋予print_indent后默认参数设置为None')"""
        create_str_tmp2 = create_str_tmp.replace(", print_indent=''", ''). \
            replace('self.print_indent = print_indent', s_new)
        create_str = create_str_tmp2
    if CLASS.__name__ in ['LGBMClassifier']:  # 'XGBClassifier',
        s_new = """self.trans_na_error = True
        self.is_unbalance = is_unbalance"""
        create_str = create_str_tmp
        old = re.findall("importance_type.*?='split'", create_str)[0]  # is_unbalance未在init的参数中，特别处理
        create_str = create_str.replace(old, f"is_unbalance=False, {old}", 1)
        create_str = create_str.replace("self.trans_na_error = True", s_new)
    return create_str


# 导入需要改造的估计器(可根据需求新增)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost.sklearn import XGBClassifier

try:
    from xgboost.sklearn import _SklObjective  # 为兼容版本
except:
    pass
from sklearn.tree import DecisionTreeClassifier

# 兼容版本
try:
    exec(ToDFVersion(NumStrSpliter))
except:
    print('')
    from Mining.selfmodule.toolmodule.dataprep import SimpleImputer, CategoricalEncoder, NumStrSpliter, FeaturePrefilter, \
    Mdlp_dt, NewValueHandler, WoeTransformer, PsiTransformer, OutlierHandler, ValueConverter, ObjectBacktoInt, __all__

# 执行类改造code的字符串，生成各个估计器的DataFrame版本
exec(ToDFVersion(Pipeline))
exec(ToDFVersion(StandardScaler))
exec(ToDFVersion(OneHotEncoder))
exec(ToDFVersion(MinMaxScaler))
exec(ToDFVersion(FeatureUnion))
exec(ToDFVersion(LogisticRegression))
exec(ToDFVersion(DecisionTreeClassifier))
exec(ToDFVersion(RandomForestClassifier))
exec(ToDFVersion(SimpleImputer))
exec(ToDFVersion(CategoricalEncoder))
#exec(ToDFVersion(LGBMRegressor))  #
#exec(ToDFVersion(LGBMClassifier))  #
#exec(ToDFVersion(XGBClassifier))

exec(ToDFVersion(NumStrSpliter))
exec(ToDFVersion(FeaturePrefilter))
exec(ToDFVersion(Mdlp_dt))
exec(ToDFVersion(NewValueHandler))
exec(ToDFVersion(WoeTransformer))
exec(ToDFVersion(PsiTransformer))
exec(ToDFVersion(OutlierHandler))
exec(ToDFVersion(ValueConverter))
exec(ToDFVersion(ObjectBacktoInt))
# exec(ToDFVersion(ColFilter))
# exec(ToDFVersion(NochangeTranformer))


# ----------------------------------------------------------------------------------------------------------------------


