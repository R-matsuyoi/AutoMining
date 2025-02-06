import os
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import sys
import warnings
import datetime
import copy
import re
import time
from Mining.selfmodule.toolmodule.strtotable import infos_to_table

try:
    from sklearn.externals import joblib
except:
    import joblib

# 导入自定义模块
from Mining.selfmodule.toolmodule.privy_outredirect import privy_Autonomy, privy_fileattr_fun
from Mining.selfmodule.toolmodule.datatrans import read_data, month_check, save_data, my_sql_fun
from Mining.selfmodule.toolmodule.ignorewarn import filter_warnings
from Mining.selfmodule.toolmodule.dataprep import *
from Mining.selfmodule.tablemodule.tablefun import get_condition_col, privy_get_trans, opera_pair_fun, month_add, get_month_count
from Mining.selfmodule.binarymodule.modelinfo import month_mark
from Mining.selfmodule.toolmodule.dataprep import MinMaxScaler_DF, MinMaxScaler

seconds = 3

def privy_feature_top(data_predict, pipeline1, woe1, Info, return_woe, n_reason=3, woe_thred=0,
                      col_notcause=None, r_limit_cause=None, valuecount_limit=50):
    """
    为用户匹配topN高分原因
    :param data_predict: 预测集
    :param pipeline1: 数据处理流水线
    :param woe1: 从pipeline1中获取的WoeTransformer_DF
    :param Info: 单个模型信息（命名元组）
    :param return_woe: 是否返回流水线转换后的数据以供后续步骤使用，即woe转换后的数据
    :param n_reason: 规定匹配原因个数
    :param woe_thred: woe阈值，woe小于等于woe_thred的top原因将置为空值(字段、注释、取值、woe等)
    :param col_notcause: 不参与原因匹配的字段列表
    :param r_limit_cause: 相关性系数阈值，超过该阈值字段对中只保留其中一个字段，以免匹配出的原因之间高度重合
    :param valuecount_limit: 用户量<=valuecount_limit的 字段-取值 不参与woe比较，即不会作为原因
    :return: 略
    备注：原因的底线为对应woe大于零
    """
    message = {}
    original = sys.stdout
    data_predict = data_predict.copy()
    pipeline1 = copy.deepcopy(pipeline1)
    data_predict.index = data_predict[Info.col_id].values
    union1 = pipeline1.named_steps['union1']

    # 补齐流水线删除的字段
    col_rid = set(pipeline1.fit_in_colnames_) - set(data_predict.columns)
    if col_rid:
        print(f'    补齐流水线删除的{len(col_rid)}个字段：{set(col_rid)}')
        for i in col_rid:
            data_predict[i] = np.nan
        print(f'    数据量：{data_predict.shape}\n')

    print('\n', end='')
    print(f"    数据分箱（union1） {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    message['union1'] = privy_Autonomy();
    sys.stdout = message['union1']
    try:
        data_cut = union1.transform(data_predict)
    except Exception as er:
        raise Exception(er)
    finally:
        sys.stdout = original
    print(f'    转换前shape：{data_predict.shape}')
    print(f'    转换后shape：{data_cut.shape}\n')

    print(f"    woe转换 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    message['woe'] = privy_Autonomy();
    sys.stdout = message['woe']
    try:
        data_woe = woe1.transform(data_cut)
    except Exception as er:
        raise Exception(er)
    finally:
        sys.stdout = original
    print(f'    转换前shape：{data_cut.shape}')
    print(f'    转换后shape：{data_woe.shape}\n')

    print(f"    确定动因字段 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    col_notcause = set(col_notcause if col_notcause else []) & set(data_woe.columns)
    print(f'    剔除非动因字段（{len(col_notcause)}个）：{col_notcause}')
    col_cause = [i for i in data_woe.columns if i not in col_notcause]
    print(f'    动因字段：{len(col_cause)}个\n')

    col_del_r = []
    if r_limit_cause:
        print('    计算动因字段的相关矩阵，删除高相关字段')
        r_matrix = woe1.r_matrix_[woe1.r_matrix_.level_0.isin(col_cause) & woe1.r_matrix_.level_1.isin(col_cause)]
        col_del_r = woe1._rhigh_del_fun(r_matrix, woe1.col_iv_, thred=r_limit_cause, data_is_rmatrix=True)[0]

    if col_del_r:
        col_cause = [i for i in col_cause if i not in col_del_r]
        print(f'    动因字段：{len(col_cause)}个\n')

    if set(data_woe.index) != set(data_cut.index):
        raise Exception('data_woe、data_cut index不同，请检查！')
    elif (data_woe.index != data_cut.index).sum() > 0:
        print('    data_cut的行序按data_woe的index重新调整')
        data_cut = data_cut.loc[data_woe.index]

    if set(data_woe.columns) - set(data_cut.columns):
        raise Exception('data_cut 缺少字段！')

    if valuecount_limit > 0:
        to = -999
        print(f'    从原因中删除用户量过小的 字段-取值')
        print(f"    阈值 valuecount_limit= {valuecount_limit}")
        if isinstance(valuecount_limit, float) & (valuecount_limit >= 1):
            valuecount_limit = int(valuecount_limit)
            print(f'    valuecount_limit虽为浮点型，但其值超过1，非比例意义，故将其更正为整型取值 valuecount_limit={valuecount_limit}')

        if isinstance(valuecount_limit, int):
            limit = valuecount_limit
            s = f"    将%s个字段的%s个取值（用户量<={limit}）从原因中剔除（对应的woe置为{to}）\n"
        elif isinstance(valuecount_limit, float):
            limit_formula = 'int(valuecount_limit * len(data_predict))'
            limit = eval(limit_formula)
            print(f"    valuecount_limit为浮点型, 阈值进一步更正为{limit}: {limit_formula}")
            s = f"    将%s个字段的%s个取值（用户量<={limit}）从原因中剔除（对应的woe置为{to}）\n"

        col_count = 0
        value_count = 0
        for c in data_woe.columns:
            c_v = data_woe.loc[data_woe[c] > woe_thred, c].value_counts()
            value_to = c_v[c_v <= limit].index
            if len(value_to) > 0:
                data_woe.loc[data_woe[c].isin(value_to), c] = to
                col_count += 1
                value_count += len(value_to)
        if col_count:
            print(s % (col_count, value_count))
        else:
            print('    无用户量过小的 字段-取值，无需剔除\n')

    field_comment = Info.comment_all.loc[Info.comment_all.是否宽表字段 == '是']
    comment_dict = dict(field_comment.set_index('field_name').comment)
    comment_lack = set(data_woe.columns) - set(comment_dict.keys())
    if comment_lack:
        s = f'缺少下列字段注释，请检查：\n{comment_lack}'
        warnings.warn(s); time.sleep(seconds)
        comment_dict_lack = {k: k for k in comment_lack}
        comment_dict = dict(comment_dict, **comment_dict_lack)

    data_colname = DataFrame(index=data_woe.index)
    for c in data_woe.columns:
        data_colname[c] = c

    print(f"    原因排序 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    data_woe_copy = data_woe[col_cause].copy()
    result_out = DataFrame()
    col_woe = data_woe_copy.columns
    woe_less = data_woe_copy.min().min() - 9999 # 比最小的woe更小的值
    for i in range(1, n_reason+1):
        print(f"    top{i} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result_out[f'colname_top{i}'] = data_woe_copy.idxmax(axis=1)
        result_out[f'colcomment_top{i}'] = result_out[f'colname_top{i}'].map(comment_dict)
        for j in col_woe:
            idx = result_out.loc[result_out[f'colname_top{i}'] == j].index
            result_out.loc[result_out.index.isin(idx), f'colvalue_top{i}'] = data_cut.loc[data_cut.index.isin(idx), j]
            result_out.loc[result_out.index.isin(idx), f'colwoe_top{i}'] = data_woe_copy.loc[data_woe_copy.index.isin(idx), j]
            data_woe_copy.loc[data_woe_copy.index.isin(idx), j] = woe_less
    print(f"    排序结束 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 此步骤没有汇入上一个排序循环中，是为了后续也许想基于原始的woe排序结果做分析
    print(f'    将woe<=woe_thred（{woe_thred})的原因置空')
    for i in range(1, n_reason+1):
        col_top = [f'colname_top{i}', f'colcomment_top{i}', f'colvalue_top{i}', f'colwoe_top{i}']
        colname_topi, colcomment_topi, colvalue_topi, colwoe_topi = col_top
        to_null = result_out[colwoe_topi] <= woe_thred
        if any(to_null):
            print(f"    top{i}: 将{to_null.sum()}行woe<=woe_thred（{woe_thred})的{', '.join(col_top)}置空")
            result_out.loc[to_null, col_top] = np.nan
        else:
            woe_min = round(result_out[colwoe_topi].min(), 3)
            print(f"    top{i}: {colwoe_topi}字段最小值{woe_min} > woe_thred（{woe_thred}）,无需处理")
    print('\n')
    return result_out, data_woe if return_woe else None, message



def pipe_need_fun(pipe):
    """
    进入流水线字段 - 流水线中删除字段
    :param pipe: 流水线
    :return: 所需字段列表
    """
    if pipe is None:
        return []
    pipe_in = list(pipe.fit_in_colnames_)  # 进入流水线的字段
    pipe_rid = set(pipe_in) & set(sum([list(v) for k, v in pipe.pipe_del_colnames_.items()], []))  # 流水线剔除的字段
    return [i for i in pipe_in if i not in pipe_rid]


def predict_fun(train_result, Info, n_reason=3, if_condition=False,
                r_limit_cause=0.8, woe_thred=0, valuecount_limit=50):
    """
    二分类模型的预测打分函数
    :param train_result: 模型训练结果（最优）
    :param Info: 单个模型信息（命名元组）
    :param n_reason: 匹配原因的个数，分别用top？标注；若取值为None，则不匹配原因
    :param if_condition: 是否按照Info.condition筛选用户，因为加工近n月基础数据时，已经限定了用户，此函数中可以忽略nfo.condition，即if_condition=False
    :param r_limit_cause: privy_feature_top函数的参数，详情见该函数注释
    :param woe_thred: privy_feature_top函数的参数，详情见该函数注释
    :param valuecount_limit: privy_feature_top函数的参数，详情见该函数注释
    :return: 预测结果(DataFrame)，本地项目组决定：概率得分是否乘以100，保留几位
    备注：若加工预测数据集时已经限定了用户范围，则condition设置为None即可
    data_predict(废弃): 预测集，若不为None，则直接基于data_predict进行数据预处理及预测，否则需在函数中读取预测集数据
    """
    with warnings.catch_warnings(record=True) as w:
        start_time = datetime.datetime.now()
        print(f"开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        message = {}
        original = sys.stdout
        print(f'模型名称：{Info.model_name}')
        print(f'预测账期：{Info.month_predict}')

        mark_traintest = month_mark(Info, 'traintest')
        if train_result is None:
            raise Exception('train_result为None, 无合格模型，请检查训练测试过程！')
        flow_name = train_result['flow_name']
        print(f'提取模型结果: {flow_name}')
        print(f"模型 [宽带探索]~训练~测试 账期：{mark_traintest}")
        pipeline = train_result['pipeline']
        model = train_result['model']
        field_comment = Info.comment_all.loc[Info.comment_all.是否宽表字段 == '是']
        pipe_need = list(field_comment.loc[(field_comment.是否入模 == '是') & (field_comment.field_name != Info.col_target), 'field_name'])

        if 'pipeline1' in train_result['flow_name']:
            print('最优模型的流水线是pipeline1')
            pipeline1 = copy.deepcopy(train_result['pipeline'])
        else:
            print('最优模型的流水线不是pipeline1，为进行“top原因匹配”， 加载train_model_flows，从中获取pipeline1')
            file_flows = f"{Info.model_wd_traintest}/train_model_flows{mark_traintest}.pkl"
            print(f"    加载模型集合：{file_flows}")
            privy_fileattr_fun(file_flows, 'unhide')
            train_model_flows = joblib.load(file_flows)  # 最优模型的相关结果
            privy_fileattr_fun(file_flows)

            # 获取pipeline1
            flows_key = [i for i in train_model_flows.keys() if 'pipeline1' in i]
            if len(flows_key) == 0:
                warnings.warn('train_model_flows中无pipeline1，无法进行“top原因匹配”，请采用其他方式！'); time.sleep(seconds)
                pipeline1 = None
            else:
                print('    从train_model_flows中获取pipeline1')
                flow_p1 = train_model_flows[flows_key[0]]
                pipeline1 = flow_p1['pipeline']
                Info_p1 = to_namedtuple(flow_p1['Infoasdict'])
                field_comment_p1 = Info_p1.comment_all.loc[Info_p1.comment_all.是否宽表字段 == '是']
                pipe1_need = list(field_comment.loc[field_comment.是否入模 == '是', 'field_name'])
                pipe_need = dropdup(pipe_need + pipe1_need)

        print(f'建模时效果评估数据集：{Info.ev_key}\n')
        score_cut_pop_old = train_result['score_cut_pop'][Info.ev_key]

        # 读取数据
        col_con = get_condition_col(Info.condition)  # 筛选条件中的字段列表
        col_score = ['model_name', 'month_train', 'month_test', 'score', 'score_cut', 'precision', 'lift']  # 本函数内添加的分数相关字段名称
        col_out = dropdup([Info.col_month, Info.col_id] + (Info.col_out if str(Info.col_out) != 'nan' else []) +
                          col_score +
                          list(Info.dict_sortscore.keys()))  # 分数表输出字段
        col_need = dropdup([i for i in col_out if i not in col_score] + pipe_need + col_con)  # 需要读取的字段
        condition = Info.condition if if_condition else None
        data_predict = read_data(Info.table_predict, 'file', condition, col_need=col_need, if_coltolower=False)

        # 补齐流水线所需的fit_in_colnames_字段(在流水线中被剔除的fit_in_colnames_字段)
        col_rid = (set(pipeline.fit_in_colnames_) | (set(pipeline1.fit_in_colnames_) if pipeline1 else set())) - \
                  set(data_predict.columns)
        if col_rid:
            print(f'补齐流水线删除的{len(col_rid)}个fit_in_colnames_字段')
            for i in col_rid:
                data_predict[i] = np.nan
            print(f'    数据量：{data_predict.shape}\n')

        print('数据账期校验:')  # 主要用于亚信平台，以防配置数据源的人为失误,可直连数据库的环境中此话为废话，可删除
        month_check(data_predict, Info.col_month, Info.month_predict)
        print('\n', end='')

        # 标记数据集名称属性
        data_predict.data_name = 'data_predict'

        if (n_reason is None) | ('pipeline1' not in train_result['flow_name']):
            print(f"最优模型 - {train_result['flow_name'].split('|')[1].strip(' ')} - 数据转换")
            woe = privy_get_trans(pipeline, 'WoeTransformer_DF', indent='    ')
            print('    向WoeTransformer_DF.psi_.col_ignore补充流水线中删除的字段，不做稳定度检验（未读取，用np.nan填充）')
            woe.psi_.col_ignore = (set(woe.psi_.col_ignore) if woe.psi_.col_ignore else set()) | col_rid

            message['pipeline'] = privy_Autonomy()
            sys.stdout = message['pipeline']
            try:
                predict_X = pipeline.transform(data_predict)
            except Exception as er:
                raise Exception(er)
            finally:
                sys.stdout = original
            print(f'    转换前shape：{data_predict.shape}')
            print(f'    转换后shape：{predict_X.shape}\n')

        result_out = None
        predict_X1 = None
        if n_reason is not None:
            print('top原因匹配')
            if pipeline1 is not None:
                print(f"    {'最优模型 - ' if ('pipeline1' in train_result['flow_name']) else ''}pipeline1 - 数据转换")
                col_notcause = list(field_comment.loc[field_comment.is_cause == '否', 'field_name'])  # 不想作为原因的字段
                print(f'    col_notcause：{col_notcause}')

                woe1 = privy_get_trans(pipeline1, 'WoeTransformer_DF', indent='    ')
                print('    向WoeTransformer_DF.psi_.col_ignore补充流水线中删除的字段，不做稳定度检验（未读取，用np.nan填充）')
                woe1.psi_.col_ignore = (set(woe1.psi_.col_ignore) if woe1.psi_.col_ignore else set()) | col_rid
                try:
                    # 若最优模型恰好是pipeline1，则返回流水线转换后的数据，即woe转换后的数据，一边下一步进行预测打分
                    return_woe = 'pipeline1' in train_result['flow_name']
                    # 匹配topN字段、注释、取值、woe等
                    result_out, predict_X1, message['pipeline1'] = \
                        privy_feature_top(data_predict, pipeline1, woe1, Info, return_woe, n_reason,
                                          woe_thred, col_notcause, r_limit_cause, valuecount_limit)
                except Exception as er:
                    raise Exception(er)
                finally:
                    sys.stdout = original
            else:
                warnings.warn('未获取到pipeline1，无法进行“top原因匹配”，请采用其他方式！'); time.sleep(seconds)

        print('模型预测\n')
        message['predict'] = privy_Autonomy()
        sys.stdout = message['predict']
        try:
            predict_X = predict_X if predict_X1 is None else predict_X1
            data_predict['score'] = np.round(model.predict_proba(predict_X)[train_result['Pcase']] * 100, 3).values
        except Exception as er:
            raise Exception(er)
        finally:
            sys.stdout = original
        data_predict['model_name'] = Info.model_name

        print('匹配各分数档位查准率、提升度（非累计）\n')  # 基于统计测试效果时的分档方式
        pre_r = train_result['pre_result'][Info.ev_key].rename(columns={'查准率': 'precision', '提升度': 'lift'})
        pre_r = pre_r.iloc[:Series((pre_r.分数 == '').values).idxmax()]  # 兼容版本
        pre_r.index = range(len(pre_r))
        pre_r.precision = pre_r.precision.str.strip('%').astype(float)/100  # 转换为数值，以便后续的计算
        cp = Series(sorted(list(pre_r.分数.iloc[:-1]) + [float('-inf'), float('inf')]))  # 分数切分点
        if_dup = cp.duplicated()
        cp[if_dup] = cp[if_dup] + np.arange(1, if_dup.sum() + 1) * 1e-10  # 防止模型分数集中，导致cp有重复
        labels = sorted(range(len(cp) - 1), reverse=True)
        data_predict['score_cut'] = pd.cut(data_predict.score, cp, right=False, labels=labels).astype(int)
        data_predict = pd.merge(data_predict, pre_r[['precision', 'lift']], left_on='score_cut', right_index=True)

        data_predict['month_train'] = Info.month_train
        data_predict['month_test'] = Info.month_test
        score_data = data_predict[col_out].copy()

        # 检验是否产生缺失值
        col_score_na = list(set(col_score) - {'month_test'}) if str(Info.month_test) == 'nan' else col_score
        col_na_count = score_data[col_score_na].isnull().sum()
        col_na_count = col_na_count[col_na_count > 0]
        if len(col_na_count) > 0:
            s = f'存在缺失值：{dict(col_na_count)}'
            raise Exception(s)

        print(f'概率得分描述统计：\n{add_printindent(DataFrame(np.round(data_predict.score.describe(), 3)))}\n')

        score_cut = pd.cut(data_predict.score, bins=np.arange(11) * 10, include_lowest=True)
        score_cut_pop_new = col_pop(score_cut)
        print(f'概率得分分段统计：\n{add_printindent(score_cut_pop_new)}\n')

        psi = np.round(PsiTransformer_DF().psi_fun(score_cut_pop_old.用户量, score_cut_pop_new.用户量).iloc[0], 5)
        s = f'模型稳定度：{psi}\n'
        print(s)
        if psi > 0.1:
            warnings.warn(s); time.sleep(seconds)

        if result_out is not None:
            data_out = pd.merge(score_data, result_out, how="left", left_on=Info.col_id, right_index=True)
        else:
            data_out = score_data
        print(f'结果数据展示：\n{add_printindent(data_out.head(2))}\n')

        # -------------------------------- 保存模型结果 ----------------------------------------------------------------
        predict_result = {
            'model_name': Info.model_name,
            'flow_name': flow_name,
            'month_predict': Info.month_predict,
            'mark_traintest': mark_traintest,
            'pipeline': pipeline,
            'message': message,
            'psi': psi,
            'Infoasdict': Info._asdict()
        }

        save_data(data=data_out, destination=Info.table_score, if_exists='replace')  # 保存分数结果

        result_mark = re.sub('^.*/|\.csv|\.txt', '', Info.table_score)
        result_mark = get_onlyvalue(re.findall('~.*?$', result_mark))  # [~宽表探索]~训练账期[~测试账期]~预测账期[~批次]
        file = f"{Info.model_wd_predict}/predict_result{result_mark}.pkl"
        print(f'保存数据处理流水线至：{file}\n')
        privy_fileattr_fun(file, 'unhide')
        joblib.dump(predict_result, file)
        privy_fileattr_fun(file)

        end_time = datetime.datetime.now()
        print(f"结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time_cost = (end_time - start_time).seconds
        print(f"耗时：{time_cost} s")

    filter_warnings(w)
    return data_out


def privy_score_deal(Info, nround, n_reason=3, woe_thred=0):
    """
    将分批次的打分结果进行整理
    :param score_all: 所有批次打分结果的合并结果
    :param Info: 单个模型信息（命名元组）
    :param n_reason: 匹配原因个数，若无原因字段，则将n_reason设置为None
    :param woe_thred: woe阈值，woe小于等于woe_thred的原因字段将置为空值
    :return:
    """
    score_all = DataFrame()
    if nround == 1:
        score_all = pd.read_csv(Info.table_score)
    else:
        print(f'合并{nround}个批次的分数结果')
        for i in range(nround):
            score_i = pd.read_csv(Info.table_score.replace('.csv', f'~{i}.csv'))
            print(f"    {i}: {score_i.shape}")
            score_all = pd.concat([score_all, score_i], axis=0)
    print(f"分数数据合计shape: {score_all.shape}\n")

    print('进行分数排名与划分档位')
    col_rank = list(Info.dict_sortscore.keys())  # np.nan
    ascending = [False] + [Info.dict_sortscore[i] for i in col_rank]
    score_all = score_all.sort_values(by=['score'] + col_rank, ascending=ascending)
    score_all['score_rank'] = np.arange(1, len(score_all) + 1)
    rank_2 = score_all['score_rank'].max() + 1 - score_all['score_rank']  # score_rank 反转
    score_all['score_level'] = pd.qcut(rank_2, np.arange(100+1)/100, labels=range(1, 100+1)).astype(int)
    print(f"分数档位概览：{add_printindent(score_all['score_level'].value_counts().sort_index(ascending=False))}\n")

    save_data(data=score_all, destination=Info.table_score, if_exists='replace')  # 保存分数结果（CSV）
    return score_all


def conditon_deldayvalue(condition):
    """
    若在日数据到位前打分，日数据条件失效（dayvalue_），在清单输出时补充！！！
    确保剔除日数据条件，可以通过在清单输出时补充以得到预期用户范围的清单！
    :param condition: 原条件
    :return: 剔除日数据条件后的新条件
    """
    if 'dayvalue_' not in condition:
        return condition
    if ' or ' in condition:
        s = "条件中含有or分支，去除部分条件可能导致逻辑错误，请确认！"
        warnings.warn(s); time.sleep(seconds)

    condition = re.sub('  +', ' ', condition)

    # 拆分各条件分支
    con_connect = ' and | and\n|\nand |\nand\n| or | or\n|\nor |\nor\n'
    con_split = re.split(con_connect, condition)

    for c in [i.strip(' |(|)') for i in con_split if 'dayvalue_' in i]:
        condition = condition.replace(c, '')

    p1 = '^ *and | and *$|^ *or | or *$|^ *\n *'
    p2 = '\( *and |\( *or '
    p3 = 'and *\)|or *\)'
    p4 = 'and\n* *and'
    p5 = 'or\n* *or'
    while re.findall('|'.join([p1, p2, p3, p4, p5]), condition):
        condition = re.sub(p1, '', condition)
        condition = re.sub(p2, '(', condition)
        condition = re.sub(p3, ')', condition)
        condition = re.sub(p4, 'and', condition)
        condition = re.sub(p5, 'or', condition)
    return condition


def ch_con_fun(Info):
    """
    替换模型信息中的condition_base、condition，忽略日数据条件
    :param Info: 命名元组，单个模型信息
    :return: 修改后的Info
    """
    print(f'condition中涉及日数据，接下来的预测过程不得不暂时忽略日数据条件')
    print('务必通知清单输出人员在输出清单时补充这些条件')
    print('确认忽略日数据条件，不会造成数据量的异常突增')
    print('检查剔除前后的逻辑确保可通过在清单输出时补充日数据条件以得到预期用户范围的清单\n')
    conbase_new = Info.conbase_new
    con_new = Info.con_new
    # print(f'--- tableXday从{Info.tableXday}修改为np.nan')
    print(f"--- 剔除condition_base中的日数据条件\n原始：\n{Info.condition_base}\n剔除后：\n{conbase_new}\n")
    print(f"--- 剔除condition中的日数据条件\n原始：\n{Info.condition}\n剔除后：\n{con_new}\n")
    Info = choradd_namedtuple(Info, {'tableXday': np.nan, 'condition': con_new, 'condition_base': conbase_new})
    return Info



def dayvalue_stat_fun(infos_pre, s_table_info,if_predict=False):
    """
    检查是否缺少日数据
    :param infos_pre: DataFrame, 个人维护的所有模型信息，添加了预测所需信息
    :return:
    """
    infos_pre = infos_pre.copy()
    for model_name in infos_pre.index:
        print(f"---------------------------------------- {model_name} -----------------------------------------------")
        use_dayvalue = False
        model_wd_traintest = infos_pre.loc[model_name, 'model_wd_traintest']
        mark_traintest = re.sub('^.*traintest', '', model_wd_traintest)  # ~[宽表探索账期~]训练测试[~测试账期]
        file = f"{model_wd_traintest}/train_result{mark_traintest}.pkl"
        train_result = joblib.load(file)  # 最优模型的相关结果
        Info = to_namedtuple(train_result['Infoasdict'])
        into = Info.comment_all.loc[Info.comment_all.是否入模 == '是']
        infos_pre.loc[model_name, 'dayvalue_intomodel'] = into.field_name.str.contains('dayvalue_').sum() > 0
        infos_pre.loc[model_name, 'dayvalue_con'] = 'dayvalue_' in str(infos_pre.loc[model_name, 'condition'])

        if infos_pre.loc[model_name, 'dayvalue_intomodel']:
            print(f'    存在日数据字段进入模型')
            use_dayvalue = True
        else:
            print(f'无日数据字段进入模型')

        if infos_pre.loc[model_name, 'dayvalue_con']:
            print(f'存在日数据字段进入用户筛选条件')
            use_dayvalue = True
        else:
            print(f'无日数据字段进入用户筛选条件')

        if if_predict:  # 只有预测时才需要事先确定日数据的接入情况
            if use_dayvalue:  # 训练时涉及了日数据（入模或条件）
                print('\n数据库基础表信息')
                table_info = infos_to_table(s_table_info, col_eval=['tableXday_desc', 'tableXscore_desc'], col_index=None)
                print('\n')
                table_info = table_info[table_info.s_field_base == Info.s_field_base]
                days = table_info.loc[table_info.tabletype == 'tableXday']
                c_list = []
                print('核验日表的行数')
                for idx in range(len(days)):
                    tableXday_desc = days.tableXday_desc.iloc[idx]
                    tablenameday, col_day, madd, dd = days.tablename.iloc[idx], tableXday_desc['col_day'], tableXday_desc['monthadd'], tableXday_desc['dd']
                    day_next = str(month_add(infos_pre.loc[model_name, 'month_predict'], int(madd))) + str(dd)  # 观察期最后账期次月dd日
                    print(f'    {tablenameday} ({day_next})')
                    mlist_day = get_month_count(tablenameday, col_day,day_next,day_datetye=tableXday_desc['day_datetye'])
                    c_idx = dict(mlist_day['count']).get(str(day_next), 0)
                    print(f'    {c_idx}行\n')
                    c_list.append(c_idx)
                c = min(c_list)
            else:
                # print('    不是预测阶段 或 无tableXday')
                c = -1
            infos_pre.loc[model_name, 'dayvalue_count'] = c

            # 日数据字段入模了 且 暂无日数据，则标记缺少日数据字段，后续将发出错误提示
            infos_pre.loc[model_name, 'dayvalue_lack'] = (infos_pre.dayvalue_intomodel[model_name]) & (c <= 0)

            # 日数据字段用于用户筛选条件 且 暂无日数据，则标记忽略日数据条件，后续将发出检查警告
            infos_pre.loc[model_name, 'dayvalue_delcon'] = infos_pre.loc[model_name, 'dayvalue_con'] & (c <= 0)
            # 若忽略日数据条件，打印模型信息修改信息（但此函数中并不会直接修改原模型信息，只是print，在循环预测中替换，以将过程信息输出到日志中）
            if infos_pre.loc[model_name, 'dayvalue_delcon']:
                conbase_new = conditon_deldayvalue(Info.condition_base)
                con_new = conditon_deldayvalue(Info.condition)
                _ = ch_con_fun(choradd_namedtuple(Info, {'con_new': con_new, 'conbase_new': conbase_new}))
            else:
                # 未赋值为nan，防止后续用新条件替换Info中条件时误操作，nan将关联整表，'???'可提早报错
                conbase_new = con_new = '???'
            infos_pre.loc[model_name, 'conbase_new'] = conbase_new
            infos_pre.loc[model_name, 'con_new'] = con_new

            # infos_pre.loc[(~infos_pre.dayvalue_intomodel) & infos_pre.dayvalue_delcon, 'tableXday'] = np.nan

            lack = infos_pre.dayvalue_lack.sum()
            if lack:
                s = f'下列模型有日数据入模，但此刻还不具备应账期的日数据， 无法完成预测打分：\n{infos_pre.loc[infos_pre.dayvalue_lack]}'
                raise Exception(s)
            else:
                print('可继续执行预测打分过程。')
        else:  # 训练测试时
            day_into = list(infos_pre.loc[infos_pre.dayvalue_intomodel].index)
            if len(day_into) > 0:
                s = f"下列模型有日数据入模，请确保预测打分时对应账期的日数据已接入，否则无法程序无法正确执行:\n{day_into}"
                warnings.warn(s); time.sleep(seconds)

            day_con = list(infos_pre.loc[infos_pre.dayvalue_con].index)
            if len(day_con) > 0:
                s = f"下列模型有日数据作为用户筛选条件，请确保预测打分时对应账期的日数据若未接入，可暂时忽略日数据条件，在清单时补充:\n{day_con}"
                warnings.warn(s); time.sleep(seconds)
    return infos_pre


def allmodel_scores_fun(infos_all_pre, weights=None, csv_filename=None, db_tablename=None, user_limit_sql=None):
    """
    汇总本项目所有模型的分数结果
    :param infos_all_pre: DataFrame，本项目所有模型的模型信息,添加了预测所需信息
    :param weights: 计算每个用户-每个模型清单的综合得分的{'指标字段名': 权重}, 综合得分字段为 model_level
                    weigths为None，则按模型本身输出的分数（分数档位），将不会添加model_level
    :param csv_filename: 汇总结果保存至csv文件名
                         若无需保存至文件，则赋值为None
    :param db_tablename: 汇总结果保存至数据库表名
                         若无需保存至数据库，则赋值为None
    :param user_limit_sql: 最好在清单输出时限制用户范围，为防止清单维护人员遗忘，保险起见，此处可提前限制一次
    :return:
    """
    # 类型限定（根据实际需求调整）
    dtype_dict = {k: str for k in set(infos_all_pre.col_id) | set(infos_all_pre.col_month) | {'month_train', 'month_test', 'area_no', 'area_no_teamzk'}}

    col_weights = list(weights.keys() if weights else [])
    colfrominfo = set(col_weights) & set(infos_all_pre.columns)
    score = DataFrame()
    for i in infos_all_pre.index:
        Info = to_namedtuple(infos_all_pre.loc[i])
        print(f"{Info.model_name}：")
        Info = choradd_namedtuple(Info, rep_dict={'#monpre#': Info.month_predict})
        print('    读取分数结果')
        score_i = pd.read_csv(Info.table_score, dtype=dtype_dict)

        print(f"    shape: {score_i.shape}")
        for j in colfrominfo:
            if j not in score_i.columns:
                value = Info._asdict()[j]
                print(f"    添加{j}字段，该字段取值为{value}")
                score_i[j] = value
                print(f"    shape: {score_i.shape}")

        col_sort = set(Info.dict_sortscore) - set(Info.col_out)
        if col_sort:
            print(f"    删除分数排序字段：{col_sort}")
            score_i = score_i.drop(columns=col_sort)
        if colfrominfo | col_sort:
            print(f"    shape: {score_i.shape}")
        print('\n', end='')
        score = pd.concat([score, score_i], axis=0)
    print(f"所有模型分数合计shape：{score.shape}\n")

    if weights:
        print(f"计算模型清单综合得分的权重 weights: {weights}")
        if None in weights.values():
            weights = {k: (v if v else 1) for k, v in weights.items()}
            print(f"    将None值赋值为1：{weights}")

        if score['precision'].astype(str).str.contains('%').sum() if ('precision' in col_weights) else False:
            print(f"将 precision（%形式） 的字段类型转换为 float")
            score.precision = score.precision.str.strip('%').astype(float)/100

        print(f'对参与计算综合得分的字段进行标准化:{col_weights}')
        minmax = MinMaxScaler_DF()
        data_ad = minmax.fit_transform(score[col_weights])

        print('\n计算模型名单综合评分, 并对分数缩放至0至100')
        minmax2 = MinMaxScaler()  # 不想print过程信息
        score['model_level'] = opera_pair_fun(data_ad * Series(weights), [data_ad.columns], ('add', '相加'))[0].iloc[:, 0]
        score['model_level'] = DataFrame(minmax2.fit_transform(score[['model_level']]) * 100).iloc[:, 0].round(2)
        score['model_level'].describe()

    print('\n调整字段顺序')
    col_out_counts = Series(sum(list(infos_all_pre.col_out), [])).value_counts()
    col_front = dropdup([Info.col_month, Info.col_id] + \
                list(col_out_counts[col_out_counts == len(infos_all_pre)].index) + \
                ['model_name', 'month_train', 'month_test', 'score_rank', 'score_level', 'score', 'score_cut'] + \
                # col_weights + \
                ['model_level'])
    col_reorder = [i for i in col_front if i in score.columns] + [i for i in score.columns if i not in col_front]
    score = score[col_reorder]
    print(f'结果概览：\n{score.head(3)}\n')

    if user_limit_sql:
        # 最好在清单输出时限制用户范围，为防止清单维护人员遗忘，保险起见，此处可提前限制一次
        print(f'输出模型分数前，基于观察账期次月x日用户状态进一步限制用户范围: {user_limit_sql}')
        data_limit = my_sql_fun(user_limit_sql, method='read')
        print(f'限制前：\n    {score.shape}')
        print(f"{add_printindent(score.groupby('model_name')[Info.col_id].count())}\n")
        score = score.loc[score[Info.col_id].isin(data_limit[Info.col_id])]
        print(f'限制后：\n    {score.shape}')
        print(add_printindent(score.groupby('model_name')[Info.col_id].count()))

    if csv_filename:
        save_data(score, csv_filename, to='file', if_exists='replace')

    if db_tablename:
        score_ad = score.drop(columns=set(score.columns) & {'score_cut', 'precision', 'lift'})
        renamedict = {k: k.replace('_teamzk', '') for k in score_ad.columns if '_teamzk' in k}  # 统一地市字段名
        if 'zk_user_no' in score_ad.columns:
            renamedict['zk_user_no'] = 'user_no'
        print(f'修改字段名：{renamedict}')
        score_ad = score_ad.rename(columns=renamedict).head()
        save_data(score_ad, db_tablename, to='gp', if_exists='append')
