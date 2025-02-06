import numpy as np
from collections import ChainMap


from Mining.selfmodule.toolmodule.dataprep import *
from Mining.selfmodule.binarymodule.traintest import CreateModelSet


def create_pipemodel(Info):
    """
    创建数据流水线、算法模型（可以根据实际随意扩充，修改流水线与算法）
    :param Info: 单个模型信息（命名元组）
    :return:
    """
    # 为避免类型不一致导致错误，在账期字段、目标字段出现的所有地方，将其统一强转类别型
    Info = choradd_namedtuple(Info, {'Pcase': str(Info.Pcase), 'Ncase': str(Info.Ncase)})

    # ------------------------------ <editor-fold desc="数据处理流水线"> -----------------------------------------------
    # 为节省时间，同流程复用，不会重复fit
    indent = '    '  # 缩进
    tran_dict = {}

    # 数值型
    tran_dict['select_num'] = NumStrSpliter_DF(select='num', trans_na_error=False, print_indent=indent * 4)
    tran_dict['imputer_num'] = SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value=0, print_indent=indent * 4)
    tran_dict['prefilter_num'] = FeaturePrefilter_DF(freq_limit=Info.freq_limit, print_indent=indent * 4)
    tran_dict['out_num'] = OutlierHandler_DF()
    tran_dict['mdlp_num'] = Mdlp_dt_DF(print_indent=indent * 4)
    tran_dict['scaler_num'] = StandardScaler_DF(print_indent=indent*4)

    # 类别型字段：提取类别型字段、填充缺失值、剔除过度集中字段、取值过多的字段、处理新水平值（应用于新数据时）
    tran_dict['select_notnum'] = NumStrSpliter_DF(select='notnum', trans_na_error=False, print_indent=indent * 4)
    tran_dict['imputer_notnum'] = SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value='unknown', print_indent=indent * 4)
    tran_dict['prefilter_notnum'] = FeaturePrefilter_DF(freq_limit=Info.freq_limit, unique_limit=Info.unique_limit, valuecount_limit=Info.valuecount_limit, print_indent=indent * 4)
    tran_dict['newvalue_notnum'] = NewValueHandler_DF(print_indent=indent * 4)
    tran_dict['encoder_notnum_pipe2'] = CategoricalEncoder_DF(encoding='onehot-dense', valueiv_limit=Info.iv_limit, Pcase=Info.Pcase, Ncase=Info.Ncase, toobject_xj=True, print_indent=indent*4)
    tran_dict['encoder_notnum_pipe3'] = CategoricalEncoder_DF(encoding='onehot-dense', valueiv_limit=Info.iv_limit, Pcase=Info.Pcase, Ncase=Info.Ncase, toobject_xj=True, print_indent=indent*1)

    # woe
    tran_dict['woe_to_pipe1'] = WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=True, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent * 1, warn_mark='pipeline1')
    tran_dict['woe_notto_pipe2'] = WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=False, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent * 1, warn_mark='pipeline2')
    tran_dict['woe_notto_pipe3'] = WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=False, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent * 1, warn_mark='pipeline3')

    # toint
    tran_dict['toint_pipe2'] = ObjectBacktoInt_DF(print_indent=indent*1)
    tran_dict['toint_pipe3'] = ObjectBacktoInt_DF(print_indent=indent*1)

    print('创建数据转换流水线1')
    tran_dict['num_pipe1'] = create_pipeline(['select_num', 'imputer_num', 'prefilter_num', 'mdlp_num'], tran_dict, indent=indent*3)
    tran_dict['notnum_pipe1'] = create_pipeline(['select_notnum', 'imputer_notnum', 'prefilter_notnum', 'newvalue_notnum'], tran_dict, indent=indent*3)
    tran_dict['union1'] = create_featurenion(['num_pipe1', 'notnum_pipe1'], tran_dict, indent=indent*1)
    pipe1 = create_pipeline(['union1', 'woe_to_pipe1'], tran_dict)

    print('创建数据转换流水线2')
    tran_dict['num_pipe2'] = create_pipeline(['select_num', 'imputer_num', 'prefilter_num', 'scaler_num'], tran_dict, indent=indent*3)
    tran_dict['notnum_pipe2'] = create_pipeline(['select_notnum', 'imputer_notnum', 'prefilter_notnum', 'newvalue_notnum', 'encoder_notnum_pipe2'], tran_dict, indent=indent*3)
    tran_dict['union2'] = create_featurenion(['num_pipe2', 'notnum_pipe2'], tran_dict, indent=indent*1)
    pipe2 = create_pipeline(['union2', 'woe_notto_pipe2', 'toint_pipe2'], tran_dict)

    print('创建数据转换流水线3')
    pipe3 = create_pipeline(['union1', 'encoder_notnum_pipe3', 'woe_notto_pipe3', 'toint_pipe3'], tran_dict)

    # 所有的数据预处理流水线
    pipelines = {'pipeline1': pipe1, 'pipeline2': pipe2}
    # </editor-fold> ----------------------------------------------------------------------------------------------

    # ------------------------ <editor-fold desc="创建算法序列"> ----------------------------------------------------
    print('创建算法序列')  # 仿照网格搜索的参数设置
    models = ChainMap(
        CreateModelSet(LogisticRegression_DF, [{'solver': ['liblinear']}]),

        CreateModelSet(RandomForestClassifier_DF,
                       [{'n_estimators': [500, 100],
                         'max_features': [0.06],
                         'max_depth': [10],
                         'min_samples_split': [1000],
                         'min_samples_leaf': [1000]
                         # ,'max_leaf_nodes': [300, 1000],
                         }]),
        CreateModelSet(RandomForestClassifier_DF),

        CreateModelSet(XGBClassifier_DF),

        CreateModelSet(LGBMClassifier_DF, [{
            'boosting_type': ['dart'],
            'objective': ['binary'],
            'num_leaves': [511],
            'learning_rate': [0.1],
            'colsample_bytree': [0.8],
            'subsample': [0.6],
            'subsample_freq': [20]}]),

        CreateModelSet(LGBMClassifier_DF, [{
            'boosting_type': ['gbdt'],
            'colsample_bytree': [0.9916168768550919],
            'importance_type':['split'],
            'is_unbalance': [False],
            'learning_rate': [0.02269504678452342],
            'max_depth': [-1],
            'min_child_samples': [165],
            'min_child_weight': [0.001],
            'min_split_gain': [0.0],
            'n_estimators': [169],
            'n_jobs': [-1],
            'num_leaves': [22],
            'reg_alpha': [0.36607148130872236],
            'reg_lambda': [0.10119960284131346],
            'silent': [True],
            'subsample': [0.9304256141925331],
            'subsample_for_bin': [40000],
            'subsample_freq': [0]}])
    )
    # </editor-fold> -----------------------------------------------------------------------------------------------

    # skip = [('pipeline2', 'LogisticRegression_DF'),
    #         ('pipeline3', 'DecisionTreeClassifier_DF'),
    #         ('pipeline3', 'RandomForestClassifier_DF'),
    #         ('pipeline3', 'LGBMClassifier_DF'),
    #         ('pipeline3', 'XGBClassifier_DF')
    #         ]
    skip = None

    return (pipelines, models, skip)


# 旧方式：
# ------------------------------ <editor-fold desc="数据处理流水线"> ----------------------------------------------
#print('创建数据转换流水线1')  # 数值型字段离散化、所有类别字段（包括数值型字段转化后的类别字段）woe转换
## 数值型字段：提取数值型字段、填充缺失值、剔除过度集中字段、离散化
#num_pipeline_df1 = Pipeline_DF([
#    ('select_num', NumStrSpliter_DF(select='num', trans_na_error=False, print_indent=indent * 4)),
#    ('imputer_num', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value=0, print_indent=indent * 4)),
#    ('prefilter_num', FeaturePrefilter_DF(freq_limit=Info.freq_limit, print_indent=indent * 4)),
#    # ('out', OutlierHandler_DF()),
#    ('Mdlp_dt_DF', Mdlp_dt_DF(print_indent=indent * 4))
#], verbose=indent * 3)
#
## 类别型字段：提取类别型字段、填充缺失值、剔除过度集中字段、取值过多的字段、处理新水平值（应用于新数据时）
#notnum_pipeline_df1 = Pipeline_DF([
#    ('select_notnum', NumStrSpliter_DF(select='notnum', trans_na_error=False, print_indent=indent * 4)),
#    ('imputer_notnum', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value='unknown', print_indent=indent * 4)),
#    ('prefilter_notnum', FeaturePrefilter_DF(freq_limit=Info.freq_limit, unique_limit=Info.unique_limit, print_indent=indent * 4)),
#    ('newvalue', NewValueHandler_DF(print_indent=indent * 4))
#], verbose=indent * 3)
#
## 合并数值、类别型字段
#union1 = FeatureUnion_DF([
#    ('num_pipe', num_pipeline_df1),
#    ('notnum_pipe', notnum_pipeline_df1)
#], verbose=indent * 1)
#
## 最终预处理流水线：数值型、类别型字段分别处理再合并、woe转化计算iv值
#pipeline1 = Pipeline_DF([
#    ('union1', union1),
#    ('woe', WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=True, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent * 1, warn_mark='pipeline1'))
#])
## --------------------------------------------------------------------------------------------------------------
#
#print('创建数据转换流水线2')
## 数值型字段：标准化
#num_pipeline_df2 = Pipeline_DF([
#    ('select_num', NumStrSpliter_DF(select='num', trans_na_error=False, print_indent=indent*4)),
#    ('imputer_num', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value=0, print_indent=indent*4)),
#    ('prefilter', FeaturePrefilter_DF(freq_limit=Info.freq_limit, print_indent=indent*4)),
#    ('scaler', StandardScaler_DF(print_indent=indent*4))
#], verbose=indent*3)
#
## 类别型字段：onehot编码
#notnum_pipeline_df2 = Pipeline_DF([
#    ('select_notnum', NumStrSpliter_DF(select='notnum', trans_na_error=False, print_indent=indent*4)),
#    ('imputer_notnum', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value='unknown', print_indent=indent*4)),
#    ('prefilter', FeaturePrefilter_DF(freq_limit=Info.freq_limit, unique_limit=Info.unique_limit, print_indent=indent*4)),
#    ('newvalue', NewValueHandler_DF(print_indent=indent * 4)),
#    ('encoder', CategoricalEncoder_DF(encoding='onehot-dense', toobject_xj=True, print_indent=indent*4))  # 独热编码
#], verbose=indent*3)
#
## 合并数值、类别型字段
#union2 = FeatureUnion_DF([
#    ('num_pipe', num_pipeline_df2),
#    ('notnum_pipe', notnum_pipeline_df2)
#], verbose=indent*1)
#
#pipeline2 = Pipeline_DF([('union2', union2),
#                         ('woe', WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=False, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent*1, warn_mark='pipeline2')),
#                         ('toint', ObjectBacktoInt_DF(print_indent=indent*1))
#                         ])
#
## --------------------------------------------------------------------------------------------------------------
#print('创建数据转换流水线3')
#num_pipeline_df3 = Pipeline_DF([
#    ('select_num', NumStrSpliter_DF(select='num', trans_na_error=False, print_indent=indent*4)),
#    ('imputer_num', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value=0, print_indent=indent*4)),
#    ('prefilter_num', FeaturePrefilter_DF(freq_limit=Info.freq_limit, print_indent=indent*4)),
#    # ('out', OutlierHandler_DF()),
#    ('Mdlp_dt_DF', Mdlp_dt_DF(print_indent=indent*4))
#], verbose=indent*3)
#
#notnum_pipeline_df3 = Pipeline_DF([
#    ('select_notnum', NumStrSpliter_DF(select='notnum', trans_na_error=False, print_indent=indent*4)),
#    ('imputer_notnum', SimpleImputer_DF(missing_values=np.nan, strategy='constant', fill_value='unknown', print_indent=indent*4)),
#    ('prefilter_notnum', FeaturePrefilter_DF(freq_limit=Info.freq_limit, unique_limit=Info.unique_limit, print_indent=indent*4)),
#    ('newvalue', NewValueHandler_DF(print_indent=indent*4))
#], verbose=indent*3)
#
## 合并数值、类别型字段
#union3 = FeatureUnion_DF([
#    ('num_pipe', num_pipeline_df3),
#    ('notnum_pipe', notnum_pipeline_df3)
#], verbose=indent*1)
#
#pipeline3 = Pipeline_DF([('union3', union3),
#                         ('encoder', CategoricalEncoder_DF(encoding='onehot-dense', toobject_xj=True, print_indent=indent*1)),
#                         ('woe', WoeTransformer_DF(Pcase=Info.Pcase, Ncase=Info.Ncase, to_woe=False, iv_limit=Info.iv_limit, r_limit=Info.r_limit, print_indent=indent*1, warn_mark='pipeline3')),
#                         ('toint', ObjectBacktoInt_DF(print_indent=indent*1))
#                         ])