import sys
import os
import copy
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import re
from collections import ChainMap
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import datetime
import time
from functools import reduce
from collections import OrderedDict
import gc

try:
    from sklearn.externals import joblib
except:
    import joblib

# 导入自定义模块
from Mining.selfmodule.toolmodule.privy_outredirect import privy_Autonomy, privy_fileattr_fun
from Mining.selfmodule.toolmodule.datatrans import read_data, month_check, sql_value
from Mining.selfmodule.toolmodule.ignorewarn import filter_warnings
from Mining.selfmodule.toolmodule.dataprep import *
from Mining.selfmodule.binarymodule.modelinfo import month_mark

seconds = 3


# ----------------------------------------------------------------------------------------------------------------------
def train_sample(data, Info, ratio, random_state=None):
    """
    训练集抽样函数
    :param data: 待抽样的DataFrame
    :param Info: 单个模型信息（命名元组）
    :param ratio: 正例:负例 为 1:ratio
    :param random_state: 随机种子
    :return: tuple，按指定比例抽样后的（特征数据集，目标字段）
    """
    print((f'分层抽样 1:{ratio}' if str(Info.col_stratified) != 'nan' else f'随机抽样 1:{ratio}'))
    if_data_name = hasattr(data, 'data_name')
    if if_data_name:
        data_name = data.data_name

    # 限定目标字段取值
    data = data.loc[data[Info.col_target].isin([Info.Pcase, Info.Ncase]), ]

    data_N = data.loc[data[Info.col_target] == Info.Ncase, :]  # 负例样本
    data_P = data.loc[data[Info.col_target] == Info.Pcase, :]  # 正例样本
    N_count = len(data_N)  # 负例样本量
    P_count = len(data_P)  # 正例样本量

    print(f'    正例样本：{P_count}')
    if P_count > Info.Pcase_limit:
        print(f'    超过Pcase_limit（{Info.Pcase_limit}），故随机取{Info.Pcase_limit}个正例样本')
        if str(Info.col_stratified) == 'nan':  # 随机抽样
            data_P_sample = data_P.sample(n=Info.Pcase_limit, random_state=random_state, axis=0)
        else:  # 分层抽样
            Psize = Info.Pcase_limit / P_count
            split = StratifiedShuffleSplit(n_splits=1, test_size=Psize, random_state=random_state)
            for train_index_P, test_index_P in split.split(data_P, data_P[Info.col_stratified]):
                data_P_sample = data_P.iloc[test_index_P]
    else:  # P_count < Info.Pcase_limit 无需操作
        data_P_sample = data_P
    P_count = len(data_P_sample)  # 正例样本量

    print(f'    负例样本：{N_count}')
    Pr_count = int(P_count * ratio)
    if N_count < Pr_count:
        print(f'    小于{Pr_count}（{P_count}*{ratio}），故只取{N_count}')
    else:
        N_count = Pr_count
        print(f'        从负例中随机抽取{N_count}（{P_count}*{ratio}）')

    if len(data_P) == 0:
        raise Exception(f'参数data的{Info.col_target}字段不包括Info.Pcase（{Info.Pcase}）')
    if len(data_N) == 0:
        raise Exception(f'参数data的{Info.col_target}字段不包括Info.Ncase（{Info.Ncase}）')

    Nsample = P_count * ratio  # 待抽负例样本量
    if Nsample > N_count:
        raise Exception(f'1:{ratio} 负例样本的抽样样本量（{Nsample}）大于其总量（{N_count}）')
    if str(Info.col_stratified) == 'nan':  # 随机抽样
        data_N_sample = data_N.sample(n=Nsample, random_state=random_state, axis=0)
    else:  # 分层抽样
        Nsize = Nsample / N_count
        split = StratifiedShuffleSplit(n_splits=1, test_size=Nsize, random_state=random_state)
        for train_index, test_index in split.split(data_N, data_N[Info.col_stratified]):
            data_N_sample = data_N.iloc[test_index]
    res = pd.concat([data_P_sample, data_N_sample])

    if if_data_name:
        res.data_name = data_name
    return res


def CreateModelSet(model_obj, param_grid=None):
    """
    创建模型序列
    :param model_obj: 模型对象
    :param param_grid: 参数网格
    :return:
    """
    class_name = model_obj.__name__
    if param_grid:
        # 构造参数组合（字符串）
        def to_str(x):  # 为了保存字符串参数值的引号
            if type(x) == str:
                quot = "'"
                return quot + x + quot
            else:
                return str(x)

        fn = lambda x, code=', ', fun=str: reduce(lambda x, y: [str(i) + code + fun(j) for i in x for j in y], x)
        param_list = list()
        for i in range(len(param_grid)):
            param_list_i = list()
            param_names = list(param_grid[i].keys())
            values = list(param_grid[i].values())
            for j in range(len(param_grid[i])):
                x = fn([param_names[j:(j + 1)], values[j]], '=', to_str)
                param_list_i.append(x)
            param_list.extend(fn(param_list_i))
        s = "{'class_name' + ' - ' + i : eval('class_name(%s)' % i) for i in param_list}".\
            replace('class_name', class_name).replace('param_list', str(param_list))
    else:
        s = "{'class_name' + ' - 默认参数': class_name()}".replace('class_name', class_name)
    return eval(s)


def model_cycle(data_set, pipelines, models, Info, field_comment, col_pre, skip=None, remain_tran_set=False):
    """
    遍历不同抽样比例、数据预处理流水线、算法 分别进行建模
    :param data_set: 数据集字典，key：数据集名称，value：数据集
    :param pipelines: 数据处理流水线字典，key：流水线名称，value：流水线对象
    :param models: 算法&参数字典，key：算法&参数名称，value：算法类对象
    :param Info: 单个模型信息（命名元组）
    :param field_comment: 模型宽表数据字典，在输出特征重要性时匹配字段中文注释，若设置为None，则仅展示字段英文名称
    :param skip: 元组(流水线名称，算法名称)组成的列表，跳过这些组合不执行， 若设置为None，则全部执行
    :param remain_tran_set: 是否保留流水线转换之后的数据集
    :return:元组(flows, message_pipe, data_trans_set)
                 flows：字典，记录每种 抽样比例 - 数据预处理流水线 - 算法 处理流的过程信息
                 message_pipe：流水线转换过程记录
                 data_trans_set：remain_tran_set为True时：流水线转换之后的数据集；False：空字典
    """
    original = sys.stdout

    data_set = copy.copy(data_set)
    pipelines_init = copy.deepcopy(pipelines)
    models = copy.copy(models)

    flows = OrderedDict()
    data_trans_set = OrderedDict()
    # message_pipe = OrderedDict()
    for ratio in data_set['data_train'].keys():  # --------- 遍历不同的抽样比例
        print(f'- 抽样比例：{ratio} -\n'.replace('-', '-'*42))
        for pipe in pipelines_init.keys():  # --------- 遍历不同的数据转换流水线
            print(f'- 流水线：{pipe} -'.replace('-', '-'*25))
            trans = copy.deepcopy(pipelines_init[pipe])  # trans = copy.deepcopy(pipelines[pipe])
            data_out = OrderedDict()

            train = data_set['data_train'][ratio]
            train_X = train.drop(set(train.columns) & {Info.col_id, Info.col_target}, axis=1)
            train_y = train[Info.col_target]
            if hasattr(train, 'data_name'):
                train_X.data_name = train.data_name
            print(f'train_X: {train_X.shape}')

            r_p = ratio + ' | ' + pipe
            message_pipe = privy_Autonomy()
            sys.stdout = message_pipe
            try:
                print('\n- data_train  fit -'.replace('-', '-'*80))
                trans.fit(train_X, train_y)

                dis_PN = DataFrame()
                for i in data_set.keys():
                    print(f'\n- {i} transform -'.replace('-', '-'*80))
                    if i == 'data_train':
                        d = data_set[i][ratio]
                    else:
                        d = data_set[i]
                    text = {'data_train': '训练集', 'data_timein': '验证集', 'data_timeout': '测试集'}[i]
                    dis = Series({'count': len(d), 'Pcount': len(d[d[Info.col_target] == Info.Pcase])})
                    dis['prop'] = format(dis.Pcount / dis['count'], '.2%')
                    dis_PN[text] = dis

                    data_out[i] = trans.transform(d)
                    if data_out[i].shape[1] == 0:
                        raise Exception(f"{i}特征转换后，shape:{data_out[i].shape},无法继续训练！")
                    data_out[i] = pd.concat([data_out[i], d[Info.col_target]], axis=1)
                    if data_out[i][Info.col_target].isnull().sum():
                        raise Exception(f'{i} 特征转换后与目标字段合并后出现缺失值！')
                dis_PN = dis_PN.T.reset_index()
            except Exception as er:
                raise Exception(er)
            finally:
                sys.stdout = original
            print(f'转换完毕，进入模型(X+Y)：{data_out[i].shape}\n')

            for model in list(models.keys()):  # --------- 遍历不同模型
                # print(model)
                if skip is not None:
                    # 跳过某些流水线与模型的组合
                    br = False
                    for pi, mo in skip:
                        br = True if (pipe == pi) & (mo in model) else br
                    if br:
                        print(f'跳过 {pipe} | {model} 组合')
                        continue

                # 若不跳过：
                key = ratio + ' | ' + pipe + ' | ' + model

                print('训练测试 % s' % model)
                # 训练
                pred = copy.copy(models[model])

                message_model = privy_Autonomy()
                sys.stdout = message_model
                try:
                    print('基于训练数据（data_train）拟合模型：')
                    pred.fit(data_out['data_train'].drop(Info.col_target, axis=1), train_y)

                    # 测试
                    pre_result = {}   # 测试效果
                    pre_result2 = {}  #
                    score_cut_pop = {}  # 概率得分分段统计
                    for j in data_out.keys():
                        print(f'\n基于模型预测概率得分 {j}：')
                        score_j = (pred.predict_proba(data_out[j])[Info.Pcase]) * 100
                        pre_result[j] = pre_result_fun(score_j, data_out[j][Info.col_target], Info.Pcase, Info.Ncase)
                        score_cut = pd.cut(score_j, bins=np.arange(11) * 10, include_lowest=True)
                        score_cut_pop[j] = col_pop(score_cut)
                except Exception as er:
                    raise Exception(er)
                finally:
                    sys.stdout = original

                pre_r = pre_result[j]
                pre_r = pre_r.iloc[:Series((pre_r.分数 == '').values).idxmax()][col_pre]  # 兼容版本
                print(f'{j}测试效果：\n{add_printindent(pre_r)}\n\n')  # 最后一个:若有data_timeout则为它，否则为data_timein

                # 特征重要性
                name = type(pred).__name__
                fit_in_colnames = pred.fit_in_colnames_
                if hasattr(pred, 'feature_importances_'):
                    field_import = DataFrame(pred.feature_importances_, index=fit_in_colnames, columns=['importance'])
                elif name in ['LogisticRegression_DF']:
                    index = ['intercept'] + list(fit_in_colnames)
                    field_import = DataFrame(np.c_[pred.intercept_, pred.coef_][0], index=index, columns=['coef'])
                else:
                    raise Exception(f'{name} 算法的特征重要性获取方式未加入代码中，请补充！')

                if field_comment is not None:
                    col_im = field_import.columns[0]
                    field_import['field_name'] = field_import.index
                    field_import = pd.merge(field_import, field_comment, how='left', on='field_name')

                    comment_null = field_import.comment.isnull()
                    if comment_null.sum():
                        di = field_comment.set_index('field_name').comment
                        di = dict(di[~di.index.duplicated()])
                        field_name_ad = field_import.loc[comment_null, 'field_name'].str.replace('~.*', '')
                        field_import.loc[comment_null, 'comment'] = field_name_ad.map(di)

                    field_import = field_import[['field_name', 'comment', col_im]]
                field_import = field_import.sort_values(by=col_im, ascending=False)
                message = {'pipe': message_pipe, 'model': message_model}

                col_shi = set(trans.fit_in_colnames_) - set(trans.del_colnames_)
                Info.comment_all['是否入模'] = '否'
                Info.comment_all.loc[Info.comment_all.field_name.isin(col_shi), '是否入模'] = '是'

                flows[key] = {
                    'model_name': Info.model_name, 'Pcase': Info.Pcase, 'Ncase': Info.Ncase, 'pipeline': trans,
                    'model': pred, 'pre_result': pre_result, 'score_cut_pop': score_cut_pop,
                    'field_import': field_import, 'message': message, 'flow_name': key, 'dis_PN': dis_PN,
                    'Infoasdict': Info._asdict()
                }
                if len({'data_timein', 'data_timeout'} & set(score_cut_pop.keys())) == 2:
                    flows[key]['psi'] = np.round(PsiTransformer_DF().psi_fun(score_cut_pop['data_timein'].用户量, score_cut_pop['data_timeout'].用户量).iloc[0], 5)
            if remain_tran_set:
                data_trans_set[r_p] = data_out
    return (flows, data_trans_set) # (flows, message_pipe, data_trans_set)


def pre_result_fun(prob, y, Pcase, Ncase, ynull='del',
                   precents=np.r_[[0.1, 0.5], np.arange(1, 10+1)]/10,
                   grades=Series(np.arange(10 + 1) * 10), rate_to_percent=True,
                   score_precision=2):
    """
    分档位预测效果统计
    :param prob: 预测概率（Series）
    :param y: 预测变量（Series）
    :param precents: 按人数划分的人数占比阈值
    :param grades: 按分数划分的分数阈值
    :param rate_to_percent: 是否将比率转化为百分数形式
    :return: 返回记录预测效果的DataFrame
    """
    pd.set_option('display.max_columns', 30)
    y = y.copy()
    prob = prob.copy()

    # 长度验证
    if len(prob) != len(y):
        raise Exception('pre_result_fun: 参数prob(%d)与y(%d)的长度不一致!' % (len(prob), len(y)))

    # 概率字段缺失值检查
    na_prob = prob.isnull().sum()
    if na_prob:
        raise Exception(f"pre_result_fun: prob参数包括{na_prob}个缺失值，请检查！")

    # 目标字段空值处理
    na_y = y.isnull().sum()
    s = f"y参数存在{na_y}个缺失值"
    if na_y:
        if ynull == 'del':
            s_ad = s + ", 剔除缺失记录"
            warnings.warn(s_ad)
            prob = prob.loc[y.notnull()]
            y = y.loc[y.notnull()]
        elif ynull == 'Pcase':
            s_ad = s + f", 将缺失值赋值为Pcase（{sql_value(Pcase)}）"
            warnings.warn(s_ad)
            y.loc[y.isnull()] = Pcase
        elif ynull == 'Ncase':
            s_ad = s + f", 将缺失值赋值为Ncase（{sql_value(Ncase)}）"
            warnings.warn(s_ad)
            y.loc[y.isnull()] = Ncase
        else:
            raise Exception(f"pre_result_fun: ynull的取值({ynull}）有误，应为del、Pcase或Ncase！")
        time.sleep(seconds)

    # 检查目标字段取值
    s = f"pre_result_fun: 取值不一致,参数y取值（{list(y.unique())}），参数Pcase、Ncase的取值（{sql_value(Pcase)}、{sql_value(Ncase)}）"
    if set([Pcase, Ncase]) - set(y):
        raise Exception(s)

    # 目标字段Pcase、Ncase之外取值的处理
    y_more = set(y) - set([Pcase, Ncase])
    if y_more:
        s_ad = "\n" + s + f", 将y中的{y_more}重新赋值为Ncase（{sql_value(Ncase)}）"
        warnings.warn(s_ad)
        y[y.isin(y_more)] = Ncase

    # 数据转换
    prob = Series(prob)  # 针对数组的情况
    y = Series(y)
    y = y.astype('category')

    # 检验概率值是否缺失
    na_count = prob.isnull().sum()
    if na_count > 0:
        raise Exception(f'prob（概率值）存在缺失值：{na_count}个')

    # 按用户占比确定概率值切分点
    pop_cutpoint = prob.quantile(1 - precents)
    pop_cutpoint.index = list(map(lambda x: format(x, '.0%'), precents))
    # pop_cutpoint.name = '概率'

    grades_s = grades.astype(str)
    grades_s.index = [1] * len(grades)
    index = grades_s[:-1].str.cat(grades_s[1:], sep='~')
    grades_cutpoint = grades[:-1]
    grades_cutpoint.index = index
    # grades_cutpoint.name = '概率'
    grades_cutpoint = grades_cutpoint.iloc[::-1]

    def precision_recall_fun(prob_cutpoint, pre_tab, cases):
        pre_tab = pre_tab.copy()
        人数_colnames = pre_tab.columns

        # 非累计
        front = pre_tab.iloc[:-1, ]
        after = pre_tab.iloc[1:, ]
        front.index = after.index
        pre_tab_noncum = pd.concat([pre_tab.iloc[[0],], after - front], axis=0)
        pre_tab_noncum['人数'] = pre_tab_noncum[cases].apply(sum, axis=1)
        pre_tab_noncum['查准率'] = pre_tab_noncum.loc[:, Pcase] / pre_tab_noncum.loc[:, '人数']
        pre_tab_noncum['提升度'] = np.nan  # 占位

        # 累计
        pre_tab_cum = pre_tab.copy()
        pre_tab_cum['累计人数'] = pre_tab_cum[cases].apply(sum, axis=1)
        pre_tab_cum['累计人数占比'] = pre_tab_cum.累计人数 / pre_tab_cum.累计人数.iloc[-1]
        pre_tab_cum['累计查准率'] = pre_tab_cum.loc[:, Pcase] / pre_tab_cum.loc[:, '累计人数']
        pre_tab_cum['累计查全率'] = pre_tab_cum[Pcase] / pre_tab_cum[Pcase].iloc[-1]
        pre_tab_cum['累计提升度'] = np.round(pre_tab_cum.累计查准率 / pre_tab_cum.累计查准率.iloc[-1], 1)
        pre_tab_noncum['提升度'] = np.round(pre_tab_noncum['查准率']/ pre_tab_cum.累计查准率.iloc[-1], 1)

        # 修改列名
        人数_case = {}
        for i in 人数_colnames:
            人数_case[i] = f'人数_{i}'
        pre_tab_noncum = pre_tab_noncum.rename(columns=人数_case)

        累计人数_case = {}
        for i in 人数_colnames:
            累计人数_case[i] = f'累计人数_{i}'
        pre_tab_cum = pre_tab_cum.rename(columns=累计人数_case)

        # 合并
        tab = pd.concat([prob_cutpoint, pre_tab_noncum, pre_tab_cum], axis=1)
        # 百分号
        if rate_to_percent:
            for i in tab.columns[tab.columns.str.contains('率$|占比')]:
                tab[i] = tab[i].map(lambda x: format(x, '.1%'))
        return tab

    def stat_fun(prob_cutpoint, cases):
        prob_cutpoint = prob_cutpoint.copy()
        # prob_cutpoint.iloc[-1] = np.ceil(prob_cutpoint.iloc[-1]*100)/100
        # prob_cutpoint = np.round(prob_cutpoint, 2)
        prob_cutpoint.name = '分数'  # 阈值

        pre_tab = prob_cutpoint.apply(lambda x: y[prob.values >= x].value_counts())
        # pre_tab.loc[prob_cutpoint.index,]
        pre_tab = pre_tab.loc[:, cases]
        pre_tab.columns = cases
        res = precision_recall_fun(prob_cutpoint, pre_tab, cases)
        if score_precision is not None:
            res.分数 = np.round(res.分数, score_precision)
        return res

    tab_pop = stat_fun(pop_cutpoint, [Pcase, Ncase])
    tab_grades = stat_fun(grades_cutpoint, [Pcase, Ncase])
    blank_line = DataFrame([[''] * tab_pop.shape[1]], columns=tab_pop.columns, index=[''])
    res = pd.concat([tab_pop, blank_line, tab_grades])
    res.index = [''] * len(res)
    return res


def lookup_pre_result(model_flows, ev_key):
    """
    查看测试结果('data_timeout'、'data_timein')
    :param model_flows: 训练的多个模型的结果
    :param ev_key: 获取'data_timeout' 或 'data_timein' 对应的测试效果
    :return: 测试效果
    """
    if ev_key is None:
        raise Exception("ev_key参数不可为None！")
    if len(model_flows) == 0:
        raise Exception(f'model_flows为{model_flows}')
    res = {}
    for i in list(model_flows.keys()):
        res[i] = model_flows[i]['pre_result'][ev_key]
    return res


def evaluate_pre_result(model_flows, ev_key, col_pre):
    """
    找出预测效果最佳的模型
    """
    pre_results = lookup_pre_result(model_flows, ev_key)
    model_mark = []
    params = []
    if_weak = []
    indicator = []
    first_precision = []
    for i in pre_results.keys():
        pre_result = pre_results[i].iloc[:8, ]  # .iloc[:8,]是为了在按人数、按分数合并的测试效果中提取按人数的测试效果
        model_mark.append(i)
        params.append(i.split(' - ')[1])

        # 检验测试效果
        weakness = []
        precision = pre_result.查准率.str.strip('%').astype(float)
        precision = precision.fillna(0)
        precision_new = precision[(precision != 100) & (precision != precision.iloc[-1])]  # 忽略头几档的100%，最后几档的相同数值

        len_un = len(pre_result.分数.unique())
        if len_un == 1:
            s = f'分数取值集中(唯一值)'
            weakness.append(s)
        if len(precision_new) == 0:
            s = "线性可分"
            weakness.append(s)
            print(f"{i}: {s}")
            warnings.warn(f"{i}: {s}"); time.sleep(seconds)
        else:
            diff1 = np.array(precision_new.iloc[1:]) - np.array(precision_new.iloc[:-1])
            if any(diff1 > 0):
                weakness.append('查准率非递减')
            if any(diff1 == 0):
                weakness.append('分数区分度弱')

            user_ratio = pre_result.累计人数占比.str.strip('%').astype(float)
            user_ratio = user_ratio[user_ratio != 100]
            diff2 = np.array(user_ratio.iloc[1:]) - np.array(user_ratio.iloc[:-1])
            if any(diff2 == 0):
                weakness.append('分数取值集中')

        if_weak.append(weakness if weakness else None)

        indicator.append(0.4 * precision.iloc[0] +
                         0.3 * precision.iloc[1] +
                         0.2 * precision.iloc[2] +
                         0.1 * precision.iloc[3])

        first_precision.append(precision.iloc[0])
    evaluate = DataFrame({
        'params': params,
        'if_weak': if_weak,
        'indicator': indicator,
        'first_precision': first_precision},
        index=model_mark, columns=['params', 'if_weak', 'indicator', 'first_precision'])
    weak_index = evaluate.loc[evaluate.if_weak.notnull()].index

    # 删除！！！
    if len(weak_index) == len(pre_results):
        s = '删除！！！所有模型的测试效果均未通过，为了测试代码，暂时取第一个模型作为最优模型，此处应删除！'
        print(s)
        warnings.warn(s); time.sleep(seconds)
        evaluate = evaluate.copy()
        evaluate.if_weak.iloc[0] = np.nan
        weak_index = evaluate.loc[evaluate.if_weak.notnull()].index
    # 删除！！！

    if len(weak_index) == len(pre_results):
        s = f"所有模型的测试效果均未通过，请调整过程重新训练:{evaluate[['if_weak']]}\n"
        print(s)
        warnings.warn(s); time.sleep(seconds)
        return None, evaluate
    else:
        if len(weak_index):
            print(f"剔除无效模型 {len(weak_index)} 个：{evaluate.loc[weak_index, 'if_weak']}")
        else:
            print('未出现无效模型')
        valid = evaluate.loc[evaluate.if_weak.isnull(), :]
        valid = valid.sort_values(by=['indicator', 'first_precision'])
        best = valid.iloc[-1, :]
        print(f'最佳模型是：{best.name}\n')
        best_model = model_flows[best.name]
        best_results = best_model['pre_result']
        for k in best_results.keys():
            pre_r = best_results[k]
            pre_r = pre_r.iloc[:Series((pre_r.分数 == '').values).idxmax()][col_pre]  # 兼容版本
            print(f"{k}的测试效果：\n{add_printindent(pre_r)}\n")
        if 'psi' in best_model.keys():
            best_psi = best_model['psi']
            s = f'分数稳定度：{best_psi}\n'
            print(s)
            if best_psi > 0.1:
                warnings.warn(s); time.sleep(seconds)
        import_out = best_model['field_import']
        import_out = import_out.sort_values(by=import_out.columns[-1], ascending=False).head(20)
        print(f'特征重要性：\n{add_printindent(import_out)}\n')
        return best.name, evaluate


def train_test_fun(Info, pipelines, models, skip, retrain_limit=50, if_condition=False):
    """
    二分类模型的训练测试函数
    :param Info: 记录模型信息
    :param pipelines: 数据处理流水线
    :param models: 算法类序列
    :param skip: 需要跳过的数据处理流水线与算法的组合
    :param retrain_limit: 最优模型的流水线中删除的字段过多时（>=retrain_limit）,删除这些字段重新训练模型，精简模型以减少预测阶段的数据转换用时
    :param if_condition: 是否按照Info.condition筛选用户，因为加工近n月基础数据时，已经限定了用户，此函数中可以忽略nfo.condition，即if_condition=False
    :return: 模型训练结果（dict）
    """
    with warnings.catch_warnings(record=True) as w:
        # ------------------------------- <editor-fold desc="打印参数"> -------------------------------------------------
        start_time = datetime.datetime.now()
        print(f"开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        trainproc_ratiolist = Info.trainproc_ratiolist
        if isinstance(trainproc_ratiolist, (int, float)):
            trainproc_ratiolist = [trainproc_ratiolist]
        # field_comment = Info.field_comment
        field_comment = Info.comment_all.loc[Info.comment_all.是否宽表字段 == '是'].copy()
        paras = {'Info.model_name': Info.model_name,
                 'Info.trainproc_ratiolist': trainproc_ratiolist,
                 'field_comment': field_comment.shape,
                 'Info.freq_limit': Info.freq_limit,
                 'Info.unique_limit': Info.unique_limit,
                 'Info.iv_limit': Info.iv_limit,
                 'Info.r_limit': Info.r_limit,
                 'Info.random_state': Info.random_state
                 }
        paras_p = '\n    '.join([str(k) + ': ' + str(v) for k, v in paras.items()])
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # ----------------------------- <editor-fold desc="数据准备"> ---------------------------------------------------
        print(f'模型名称：{Info.model_name}')
        print(f"参数设置：\n    {paras_p}")
        mark = month_mark(Info, 'traintest')
        print(f'mark: {mark}')
        print(f'目标字段：{Info.col_target}; 正负例取值：{Info.Pcase}、{Info.Ncase}')

        # 为避免类型不一致导致错误，在账期字段、目标字段出现的所有地方，将其统一强转类别型
        Info = choradd_namedtuple(Info, {'Pcase': str(Info.Pcase), 'Ncase': str(Info.Ncase)})
        field_comment.loc[field_comment.field_name.isin([Info.col_month, Info.col_target]), 'dtype_classify'] = '类别型'
        col_char = list(field_comment.loc[field_comment.dtype_classify == '类别型', 'field_name'])
        col_num = list(field_comment.loc[field_comment.dtype_classify == '数值型', 'field_name'])

        print(f'field_comment.shape: {field_comment.shape}')
        col_need = field_comment.field_name
        print(f"    len(col_need): {len(col_need)}")
        not_into = field_comment.field_name[field_comment.into_model.str.strip(' ') == '删除']
        if len(not_into):
            print(f"    删除field_comment中into_model取值为‘删除’的{len(not_into)}个字段：{list(not_into)}")
            col_need = [i for i in col_need if i not in not_into]
            print(f"    len(col_need): {len(col_need)}")
        else:
            print('    into_model取值‘删除’字段为空')

        is_del = set(Info.col_del) & set(col_need)
        if is_del:
            print(f'    删除Info.col_del中的{len(is_del)}个字段：{is_del}')
            col_need = [i for i in col_need if i not in is_del]
            print(f"    len(col_need): {len(col_need)}\n")
        else:
            print(f"    Info.col_del为空\n")

        col_no_x = (Info.supply_other | {'user_acct_month'}) & set(col_need)
        if col_no_x:
            print(f'    删除Info.supply_other、col_mark中的{len(col_no_x)}个字段：{col_no_x}')
            col_need = [i for i in col_need if i not in col_no_x]
            print(f"    len(col_need): {len(col_need)}\n")

        # 读取数据
        condition = Info.condition if if_condition else None
        data = read_data(Info.table_train, 'file', col_need=col_need, col_char=col_char, col_num=col_num,
                         condition=condition, if_coltolower=False)
        timeout = ''
        if str(Info.month_test) == 'nan':      # 未设置时间外测试集data_timeout（Info.month_test为缺失值）
            if str(Info.table_test) != 'nan':  # 却设置了时间外测试集数据（Info.table_test非缺失）
                warnings.warn(f'month_test为{Info.month_test}，将忽略table_test，未设置时间外测试集，请确认！'); time.sleep(seconds)
        else:  # 若设置了时间外测试集data_timeout（Info.month_test非缺失值）
            if str(Info.table_test) != 'nan':  # 若时间外测试集并未合并至Info.table_train中
                timeout = read_data(Info.table_test, 'file', col_need=col_need, col_char=col_char, col_num=col_num, if_coltolower=False)
                print('合并训练、测试数据')
                data = pd.concat([data, timeout])
            if 'data_timeout' not in data['data_use'].unique():
                raise Exception(f"month_test为{Info.month_test}, 但数据的data_use字段未包括'data_timeout'，请检查！")

        print(f'data.shape: {data.shape}\n')
        data_dis = data.groupby([Info.col_month, 'data_use', Info.col_target])[Info.col_id].count()
        print(f'数据分布：{add_printindent(data_dis)}\n')

        print('数据集整合：')
        data_set = OrderedDict()
        for data_name in ['data_train', 'data_timein', 'data_timeout']:
            step = data_name.split('_')[1]
            d = data.loc[data['data_use'] == data_name]

            if len(d) > 0:
                print(f'-------------------------- {step} --------------------------')
                # 检查目标字段取值
                y_set = set(d[Info.col_target])
                y_set_expect = {Info.Pcase, Info.Ncase}
                if y_set != y_set_expect:
                    raise Exception(f'{data_name}: {Info.col_target}字段取值有误，实际为{y_set}，应为{y_set_expect}！')

                print('校验账期:', end=' ')
                eval(f'month_check(d, Info.col_month, {d[Info.col_month].unique()[0]})')
                d.data_name = data_name  # 标记名称属性
                d.y_carrier = d[Info.col_target]  # 在新数据集中携带y变量（此操作有点奇怪，但是截至目前最简单的实现方式
                if step == 'train':
                    # 训练集均衡抽样
                    train_sample_set = OrderedDict()
                    target_count = d[Info.col_target].value_counts()
                    if target_count[Info.Pcase] > Info.Pcase_limit:
                        print(f"训练数据data_train中正样本数量{target_count[Info.Pcase]}，大于Pcase_limit（{Info.Pcase_limit}），将进行正样本抽样")
                    ratio_actual = target_count[Info.Ncase] / min(target_count[Info.Pcase], Info.Pcase_limit)
                    ratiolist_figure = [i for i in trainproc_ratiolist if i !='actual']
                    ratio_more = [i for i in ratiolist_figure if i > ratio_actual]
                    if ratio_more:
                        print(f'trainproc_ratiolist：{trainproc_ratiolist}')
                        for i in ratio_more:
                            trainproc_ratiolist.remove(i)
                        print(f'正负例实际比例最高：1:{round(ratio_actual, 2)}，将trainproc_ratiolist纠正为:{trainproc_ratiolist}')
                        if len(trainproc_ratiolist) == 0:
                            print("trainproc_ratiolist中比例全部大于真实比例,令trainproc_ratiolist=['actual']")
                            trainproc_ratiolist = [f'actual({round(ratio_actual, 3)})']
                    for ratio in trainproc_ratiolist:
                        key = 'ratio_1_' + str(ratio)
                        if not isinstance(ratio, str):
                            train_sample_set[key] = train_sample(d, Info, ratio, Info.random_state)
                        elif ratio.count('actual'):
                            train_sample_set[key] = d
                    data_set[data_name] = train_sample_set
                else:
                    data_set[data_name] = d
        del data, timeout
        gc.collect()
        # </editor-fold> -----------------------------------------------------------------------------------------------

        if 'data_timeout' in data_set.keys():
            print("具备data_timeout数据集，以其测试效果评估模型")
            ev_key = 'data_timeout'
        elif 'data_timein' in data_set.keys():
            print("仅具备data_timein数据集（无data_timeout数据集），以其测试效果评估模型")
            ev_key = 'data_timein'
        else:
            s = f'请确定模型评估数据集名称, data_set.keys():{list(data_set.keys())}'
            warnings.warn(s); time.sleep(seconds)
            ev_key = None
        Info = choradd_namedtuple(Info, {'ev_key': ev_key})

        # 测试效果输出字段列表
        col_pre = ['累计人数占比', '累计人数', '累计人数_'+str(Info.Pcase), '累计查准率', '累计查全率', '累计提升度']
        print('\n', end='')
        print('遍历不同的抽样比例、数据预处理流水线、算法 进行训练测试:\n')
        model_flows, _ = model_cycle(data_set, pipelines, models, Info, field_comment, col_pre, skip, False)

        # ---------------------------------- <editor-fold desc="确定最佳模型"> -----------------------------------------
        print('确定最佳模型')
        best_name, evaluate = evaluate_pre_result(model_flows, Info.ev_key, col_pre)
        # </editor-fold> -----------------------------------------------------------------------------------------------

        # ---------------------------- <editor-fold desc="保存模型结果"> ------------------------------------------------
        file1_mark = f'{Info.model_wd_traintest}/train_result{mark}.pkl'
        if best_name is None:
            s = f'未选出最佳模型, 未保存 {file1_mark}，请调试！'
            print(s)
            if os.path.exists(file1_mark):
                # 避免路径下保存了旧的模型训练结果，旧结果的宽表结构可能已被新宽表结构覆盖，
                # 后续按照新的宽表结构加工预测集，基于旧的模型训练结果，二者基于的表结构不同，程序将报错
                old_remove = f"发现并删除旧结果：{file1_mark}"
                print(old_remove)
                os.remove(file1_mark)
        else:
            pipe = model_flows[best_name]['pipeline']
            pipe_del = set(model_flows[best_name]['pipeline'].del_colnames_)
            pipe_del = [i for i in pipe_del if '~' not in i]
            if len(pipe_del) > retrain_limit:
                print('\n')
                print(f"最优模型的流水线中删除了{len(pipe_del)}个字段(>{retrain_limit})，删除这些字段重新训练，精简模型以减少预测阶段耗时" +
                      '\n顺便可以对比两次训练的结果，确保模型稳健性，以防偶然因素导致模型测试效果不稳定')
                rbest, pbest, mbest = [i.strip(' ') for i in best_name.split('|')]
                data_set_best = data_set.copy()
                data_set_best['data_train'] = {k: v for k, v in data_set['data_train'].items() if k == rbest}
                data_set_best['data_train'][rbest] = data_set_best['data_train'][rbest].drop(columns=pipe_del)

                pipe_best = {k: v for k, v in pipelines.items() if k == pbest}
                model_best = {k: v for k, v in models.items() if k == mbest}
                retrain, _ = model_cycle(data_set_best, pipe_best, model_best, Info, field_comment, col_pre, skip, False)
                if len(retrain) != 1:
                    raise Exception('重新训练后所得结果不唯一!')
                best_name2, evaluate2 = evaluate_pre_result(retrain, Info.ev_key, col_pre)

                ind = evaluate.loc[best_name].indicator
                ind2 = evaluate2.loc[best_name].indicator
                ind_diff = abs(ind2 - ind)

                if ind_diff > 10/100:
                    s = '重新训练与首次最优模型的效果差异较大，请复合！'
                    warnings.warn(s); time.sleep(seconds)

                train_result = retrain[best_name2]
            else:
                train_result = model_flows[best_name]
            print(f"保存训练结果至：{file1_mark}")
            joblib.dump(train_result, file1_mark)

        # 若存储空间足够，可以不加条件判断直接保存model_flows，以备不时之需
        save_flows = False
        file2_mark = f'{Info.model_wd_traintest}/train_model_flows{mark}.pkl'
        if best_name is None:  # 未选出最优模型（若无时间重新训练，挑选次优方案）
            print(f"保存模型集合至：{file2_mark}")  # 挑选、更换模型备用
            privy_fileattr_fun(file2_mark, 'unhide')
            joblib.dump(model_flows, file2_mark)
            privy_fileattr_fun(file2_mark)
            save_flows = True
        elif 'pipeline1' not in best_name:
            print()
            name_p1 = [i for i in model_flows.keys() if 'pipeline1' in i]
            s1 = '最优模型的流水线不是pipeline1,若欲在预测时匹配topn原因需要使用pipeline1'
            if len(name_p1) > 1:
                name_p1 = Series(name_p1, index=name_p1).apply(len)
                name_p1 = name_p1[name_p1 == name_p1.min()].index[0]
                print(f"{s1}，从model_flows挑选一个基于pipeline1的保存至{file2_mark}: {name_p1}")
                model_flows_pipe1 = {k: v for k, v in model_flows.items() if k == name_p1}
                privy_fileattr_fun(file2_mark, 'unhide')
                joblib.dump(model_flows_pipe1, file2_mark)
                privy_fileattr_fun(file2_mark)
                save_flows = True
            else:
                s = f"{s1}, 且model_flows中无pipeline1!"
                warnings.warn(s); time.sleep(seconds)

            if os.path.exists(file2_mark) & (not save_flows):
                print(f'删除旧的{file2_mark}')
                os.remove(file2_mark)

        pre_results = {k: v['pre_result'] for k, v in model_flows.items()}
        file_pre_results = f'{Info.model_wd_traintest}/train_pre_results{mark}.pkl'
        print(f"保存模型测试效果集合至：{file_pre_results}")  # 挑选、更换模型备用
        privy_fileattr_fun(file_pre_results, 'unhide')
        joblib.dump(pre_results, file_pre_results)
        privy_fileattr_fun(file_pre_results)

        end_time = datetime.datetime.now()
        print(f"结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time_cost = (end_time - start_time).seconds
        print(f"耗时：{time_cost} s")
        # </editor-fold> -----------------------------------------------------------------------------------------------

        if best_name is None:
            raise Exception(s)

    filter_warnings(w)
    return train_result

