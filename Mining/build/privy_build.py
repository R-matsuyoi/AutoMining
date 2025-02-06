
# ---------------------------------- <editor-fold desc="Tips"> -----------------------------------------
# 准备：
# 数据库：所有模型所用到的基础数据的近n账期数据
# excel: 整理建模宽表基础数据字典field_base，以长字符串的形式粘贴至 ./selfmodule/tablemodule/basestr.py中
#        整理模型信息汇总,记录各模型之间相异的信息，包括模型信息_静态、模型信息_动态、数据库基础表信息
# 修改./selfmodule/tablemodule/basestr.py：
#      更换“模型信息_静态”以适应本项目情况
# 修改 ./selfmodule/toolmodule/datatrans.py:
#      数据传输，调整适用本地项目的数据库/文件（db = 'gp'，prefix = 'ml.'，用户名密码等）
#      确保read_data()、save_data()按预期执行;
#      确保时间类字段读取为日期型，否则在数据库/文件中转换好格式，在python内被读取为日期型以便计算时长
# 修改 ./selfmodule/toolmodule/privy_outredirect.py:
#       修改sys_tem
# 修改 privy_build - self.py：
#       修改默认工作目录、更换“模型信息_动态”
#
# 训练测试函数中确定最佳模型时，搜索并注释掉 ：删除！！！所有模型的测试效果均未通过，为了测试代码，暂时取第一个模型作为最优模型，此处应删除！
#
# 导入包、模块: 选择隐式、显式中的一种
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ---------------------------------- <editor-fold desc="导入包、模块（隐式）"> -----------------------------------------
# 代码和执行日志在平台无痕：
#  1.将本机代码汇总成长字符串，隐藏上传至平台的“.pkl”文件中
#  2.在平台：从“.pkl”中获取代码写入“testcode.py”，import testcode中所有代码后将“testcode.py”置空
#            模型执行日志隐藏至 “ .pkl”


plat = '本机测试'

from  Mining.build.all_code import *

# --------- 保存模型结果的目录
modelwd_platform = {'本机测试': 'D:\work\pycharm\Project\AutoMining\Mining\data\\test'}[plat]

# --------- 个人工作目录
selfwd_platform = {'本机测试': 'D:\work\pycharm\Project\AutoMining\Mining\data\\test2'}[plat]

# # --------- 默认值替换字段
default_values = {
        'sys_tem': {'本机测试': 'win'}[plat],
        'db': {'本机测试': 'gp'}[plat],
        'prefix': {'本机测试': 'kjdb.'}[plat],
        'dbname': {'本机测试': None}[plat],
        'user': {'本机测试': None}[plat],
        'pwd': {'本机测试': None}[plat],
        'port': {'本机测试': None}[plat],
        'host': {'本机测试': None}[plat]
    }

# --------- 把代码上传至建模平台（首次执行即可，除非更改代码需要重传)
# code_init = privy_upload_code(selfwd_platform, default_values)  # print(code_init)
# --------- 复制函数打印结果，在平台中将代码从.pkl中取出并执行（import部分需要每次执行）
_ = privy_exec_code(plat, selfwd_platform, modelwd_platform, default_values, code_py_empty=False)
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ---------------------------------- <editor-fold desc="导入包、模块（显式）"> -----------------------------------------
print('\n\n~ 导入包、模块 ~'.replace('~', '~'*66))
import os
import traceback
try:
    from sklearn.externals import joblib
except:
    import joblib

import pandas as pd
pd.options.display.max_columns = 30
pd.options.display.max_rows = 500
pd.set_option('display.width', 100000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置工作目录
original_wd = os.getcwd()
print(original_wd)
# os.chdir()  # 确保工作目录为自己的文件夹，以免在他人目录下误操作
print(os.getcwd())

from Mining.selfmodule.toolmodule.privy_outredirect import privy_log_write, privy_log_save
from Mining.selfmodule.binarymodule.modelinfo import *
from Mining.selfmodule.binarymodule.traintest import train_test_fun
from Mining.selfmodule.binarymodule.pipemodel import create_pipemodel
from Mining.selfmodule.binarymodule.predictscore import *
from Mining.selfmodule.tablemodule.tablefun import *
from Mining.selfmodule.binarymodule.privy_stat import *
from Mining.selfmodule.toolmodule.privy_mylistpath import getAllFilesize
from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, choradd_namedtuple
from Mining.selfmodule.binarymodule.privy_report import *
from Mining.selfmodule.toolmodule.predict_job import predict_job_func

from Mining.selfmodule.tablemodule.tablesql import *  # 生成sql
from Mining.selfmodule.toolmodule.strtotable import string_to_table
from Mining.selfmodule.tablemodule.basestr import s_yw_team, s_kd_user

original_out = sys.stdout
original_err = sys.stderr

# privy_log_write函数：
log_type = 'txt'  # 日志类型
log_pkl = None    # 仅针对pkl日志，此处仅为参数占位
# privy_deltestcode函数：仅针对隐式导入包和模块，此处仅为参数展位
selfwd_platform = None



plat = '本机测试'
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ---------------------------------- <editor-fold desc="模型信息_动态（！需手动修改账期）"> ----------------------------
sqltype = 'execute'  # 1.python可连数据库执行sql的环境：execute  2.否则：print，复制语句粘贴到数据库中执行
sqlcomment = '----'    # sql的注释字符

print('\n\n~ 模型信息_动态 ~'.replace('~', '~'*66))
# 1.若需要特别指定宽表探索账期 month_tabexp，则添加进s_info_changed，否则month_tabexp默认等于month_train
# 2.根据实际情况调整各列取值，若不设置则取默认值，详情见privy_modelsummary函数
# 3.非个人维护的模型，在model_name开头添加#：获取个人维护的所有模型信息时将忽略#模型，获取本项目所有模型时则恢复#模型

# 从excel粘贴过来的模型信息_动态
s_info_changed = """model_name	month_train	month_test	Pcase_limit	 traintable_ratio	Pcumsum_limit	 timein_count	 timeout_limit	 trainproc_ratiolist	iv_limit	r_limit	marketlevel
模型示例     	202010	202012	1000	2	2	500	1000	[1, 2, 10]	0.05	0.95	1
模型示例2     	202011	202012	1000	2	2	500	1000	[1, 2, 10]	0.05	0.95	1
流失预警模型_移网	'202203'	'202204'	10000	3	np.nan	50000	100000	[3]	0.05	0.95	1
转融合模型_单C	'202201'	'202202'	10000	3	np.nan	50000	100000	[3]	0.05	0.95	1
副卡加装模型_移网	'202201'	'202202'	10000	3	np.nan	50000	100000	[3]	0.05	0.95	1
套餐高迁模型	'202201'	np.nan	10000	3	np.nan	50000	100000	[3]	0.05	0.95	1
流量月包加装模型	'202201'	'202202'	10000	3	np.nan	50000	100000	[3]	0.05	0.95	1
终端换机模型	'202201'	'202202'	10000	10	np.nan	50000	100000	[10]	0.01	0.95	1
加定向流量包模型	'202201'	'202202'	10000	10	2	50000	100000	[10]	0.05	0.95	1
5G加包模型	'202201'	'202202'	10000	10	np.nan	50000	100000	[3]	0.05	0.95	1
视频彩铃潜客模型	'202201'	'202202'	10000	5	np.nan	50000	100000	[5]	0.05	0.95	1
"""


col_dealvalue = 'ALL'; col_eval = ['trainproc_ratiolist']

# 本人执行的所有模型信息(动态+静态)
infos = privy_modelsummary(infos_to_table(s_info_changed, col_dealvalue, col_eval))

# 本项目的所有模型信息，用于所有模型打分完毕后，交由一人统一整合本项目的所有模型分数
del_model = None  # ['模型示例'] 需要删除的模型名称，无需删除则None；用意：在正式使用代码打分时删除“模型示例”
infos_all = privy_modelsummary(infos_to_table(s_info_changed, col_dealvalue, col_eval, del_pound=False, del_model=del_model))

# privy_basedatafun函数：
nmg_yaxin = {'本机测试': ['']*2,
             }[plat]

# tab_explore_create函数:
auto_pair2 = False  # 是否进行字段两两之间的自动衍生（加减乘除）
diff_limit = None   # 统计近n月基础数据的各账期数据：某字段某两个账期之间取值分布占比差值>=diff_limit,发出警告，None则不检查
table_psi = True    # 是否计算结果宽表字段稳定度
table_r = True      # 是否计算结果宽表字段之间的相关性系数
src = {'本机测试': 'gp'}[plat]  # 规定tab_explore_create()从数据库还是文件中读取输入数据

# 预测阶段设置
n_reason = 3    # 匹配原因字段个数，若取值为None，则不匹配原因
m_p = '202012'  # 预测的数据账期, 若各模型取值不同则分别设置

# 测试代码 删除！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# infos.Pcase_limit = 1000; infos.traintable_ratio=1; infos.timein_count=2000; infos.timeout_limit=3000
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# -------------------------------- <editor-fold desc="生成数据加工sql并执行（！需手动修改账期）"> ----------------------
# python可直连数据库：执行通过python执行sql即可
# python不可直连数据库：复制生成的sql粘贴到数据库中执行
# 为避免多人重复加工数据，一个项目交由一人统一完成本地项目模型数据的“加工”内容：

month_list = ['202203']

print(f'\n\n\n{sqlcomment} 加工：移网基础宽表_用户to套餐组')
for month in month_list:
    print(f'----------------------------- {month} ---------------------------------')
    team_summary = team_summary_sql(month)
    print(team_summary)
    if sqltype == 'execute':
        my_sql_fun(team_summary, method='execute')
t_team_summary = 'kehujingyingbudb.dm_zc_yw_moxing_team_info_m'
# my_sql_fun(f"select * from {t_team_summary} limit 2", method='read')
# my_sql_fun(f"select month_id, count(1) from {t_team_summary} group by month_id", method='read')


print(f'\n\n\n{sqlcomment} 加工：移网特征补充表_套餐组')
for month in month_list:
    print(f'----------------------------- {month} ---------------------------------')
    ywteam_xadd = ywteam_xadd_sql(month)
    print(ywteam_xadd)
    if sqltype == 'execute':
        my_sql_fun(ywteam_xadd, method='execute')
t_ywteam_xadd = 'kehujingyingbudb.dm_zc_yw_moxing_team_add_m '
# my_sql_fun(f"select * from {t_ywteam_xadd} limit 2", method='read')
# my_sql_fun("select acct_month, month_id, count(1) from {t_ywteam_xadd} group by acct_month,month_id", method='read')


print(f'\n\n\n{sqlcomment} 加工：移网目标表_套餐组')
for month in month_list:
    month_data = month_add(month, -1)
    print(f'----------------------------- {month_data} ---------------------------------')
    ywteam_target = ywteam_target_sql(month_data)
    print(ywteam_target)
    # if sqltype == 'print':
    #     my_sql_fun(ywteam_target, method='execute')
t_ywteam_target = 'kehujingyingbudb.dm_zc_yw_moxing_team_target_m'
# my_sql_fun(f"select * from {t_ywteam_target} limit 2", method='read')
# my_sql_fun(f"select acct_month, month_id,count(1) from {t_ywteam_target} group by acct_month,month_id", method='read')


print(f'\n\n\n{sqlcomment} 加工：宽带目标表_用户')
for month in month_list:
    month_data = month_add(month, -1)
    print(f'----------------------------- {month_data} ---------------------------------')
    kduser_target = kduser_target_sql(month_data)
    print(kduser_target)
    if sqltype == 'execute':
        my_sql_fun(kduser_target, method='execute')
t_kduser_target = 'kehujingyingbudb.dm_zc_kd_moxing_target_m'
# my_sql_fun(f"select * from {t_kduser_target} limit 2", method='read')
# my_sql_fun(f"select acct_month, month_id,count(1) from {t_kduser_target} group by acct_month,month_id", method='read')


# ------ 统计: 所有模型目标字段分布，初步检查各模型各账期的目标字段分布有无显著异常
print(f'\n\n\n{sqlcomment} 统计所有模型目标字段分布')
_ = privy_target_stat(sqltype, infos)


print(f'\n\n\n{sqlcomment} 加工：table_XY（field_base中罗列的所有字段）')
# 预测打分时，加工table_XY时，还没有目标字段信息（全null），应在目标字段信息具备后重刷对应账期数据
# 该表只需要保留训练模型涉及的最近账期即可，历史账期数据及时清理，以免浪费存储空间
for s_field_base in infos_all.s_field_base.unique():  # s_field_base = 's_yw_fake'  s_field_base = 's_yw_team'
    print(f'\n\n\n{sqlcomment} ~ {s_field_base} ~'.replace('~', '-'*60))
    tabletype = 'insert'  # 1.创建分区表：create  2.插入数据：insert
    month = '202012'      # 操作账期 tabletype='create' 时，将忽略month参数，随意取值
    table_XY_fun(sqltype, tabletype, infos_all, s_field_base, month, nmg_yaxin)

#=============================================

print(f'\n\n\n{sqlcomment} 统计：检查基础宽表近几个月取值分布,确定“不可用”字段')
tablename_exam_v = "kehujingyingbudb.table_value_exam_v_m"  # 检查结果表名（竖排），正式表，每月插入新数据
tablename_exam_h = 'kehujingyingbudb.table_value_exam_h_m'  # 检查结果表名（横排），临时表，每月覆盖重建
if False:  # tablename_exam_v表：首次需要创建，改成True；之后每月向该表中插入即可
    cre_exam_v = cre_exam_v.replace('@tablename@', tablename_exam_v)
    my_sql_fun(cre_exam_v, method='execute')

tablename_list = ['kehujingyingbudb.dm_zc_yw_moxing_info_m', 'kehujingyingbudb.dm_zc_kd_moxing_info_m']
# tablename_list = ['kehujingyingbudb.ml_feature_info_yw_user_m']  # 本机测试，模型示例

# 数值型字段分段统计的切点, 无则None
dict_cutpoints = {}
dict_cutpoints['kehujingyingbudb.ml_feature_info_yw_user_m'] = {
    "arpu": [float("-inf"), 50, 100, 150, 200, 300, 500, float("inf")],
    "gprs_flow": np.array([0, 100, 500, 1000, 2000, 5000, 10000, float("inf")]) * 1024
}
# dict_cutpoints['kehujingyingbudb.ml_feature_info_yw_user_m'] = None # 无则None

# 类别型字段统计的取值, 无则None
dict_classvalues = {}
dict_classvalues['kehujingyingbudb.ml_feature_info_yw_user_m'] = {
    "sex": ['男', '女'],
    "if_nolimit": ['是'],
    "if_cred_multi": [1, 2, 7]
}
# dict_classvalues['kehujingyingbudb.ml_feature_info_yw_user_m'] = None  # 无则None

month = '202012'                                            # 新增检查账期
month_list = ['202009', '202010', '202011', '202012']       # 进行数据分布对比的账期列表（tablename_exam_v须具备这些账期的检查数据）
for tablename in tablename_list:
    print(f"\n\n\n- {tablename} -".replace('-', '-'*80))
    table_exam_sql(sqltype, month, tablename, get_tablefield(tablename), tablename_exam_v, dict_cutpoint=dict_cutpoints[tablename], dict_classvalue=dict_classvalues[tablename], sqlcomment=sqlcomment)
    table_exam_vtoh(sqltype, month_list, tablename_exam_v, tablename_exam_h, col_month='acct_month',  nmg_yaxin=nmg_yaxin, sqlcomment=sqlcomment)
# </editor-fold> -------------------------------------------------------------------------------------------------------# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# --------------------------- <editor-fold desc="模型宽表探索 & 训练集、测试集加工 & 训练测试过程"> --------------------
print('\n\n~ 模型宽表探索 & 训练集、测试集加工 & 训练测试过程 ~'.replace('~', '~'*50))
log_train, base_train, create_train, base_test, create_test, train_res, train_err = {}, {}, {}, {}, {}, {}, {}
for model_name in infos.index:  # model_name = '模型示例'  model_name = '流失预警模型_移网'
    print('# %s #'.replace('#', '#'*80) % model_name)
    try:
        Info = to_namedtuple(infos.loc[model_name])  # 模型信息
        ch = {'auto_pair2': auto_pair2,
              'diff_limit': diff_limit,
              'table_psi': table_psi,
              'table_r': table_r,
              'col_mark': ['user_acct_month', 'data_use', Info.col_month, Info.col_id]  # 账期、用户、数据集标识
              }
        Info = choradd_namedtuple(Info, ch)

        # <editor-fold desc="准备日志文件">
        lo_paras = privy_log_write(Info, 'traintest', log_type, log_pkl)
        lo = lo_paras[0]
        sys.stdout, sys.stderr = lo, lo
        # </editor-fold>

        print('\n\n# 加工训练账期近n月基础数据\n '.replace('#', '#' * 70))
        base_train[model_name], Info = privy_basedatafun('train', Info, nmg_yaxin=nmg_yaxin)
        # base_train[model_name] = f"{prefix}mid_{Info.short_name}_recent_{'train'}_{Info.month_train}"
        # Info = to_namedtuple(joblib.load(f"{Info.model_wd_traintest}/Info~base_{'train'}.pkl"))

        print('\n\n# 探索模型宽表\n '.replace('#', '#' * 70))
        exp, Info = tab_explore_create(base_train[model_name], Info, 'explore', src)
        # Info = to_namedtuple(joblib.load(Info.model_wd_traintest + '/Info~tabexp.pkl'))

        print('\n\n# 训练集加工\n '.replace('#', '#' * 70))
        create_train[model_name], Info = tab_explore_create(base_train[model_name], Info, 'create', src)
        # Info = to_namedtuple(joblib.load(Info.model_wd_traintest + '/Info~tabcre_train.pkl'))

        if str(Info.month_test) != 'nan':  # 若设置了时间外测试集data_timeout
            print('\n\n# 加工测试账期近n月基础数据\n '.replace('#', '#' * 70))
            base_test[model_name], Info = privy_basedatafun('test', Info, nmg_yaxin=nmg_yaxin)
            # base_test[model_name] = f"{prefix}mid_{Info.short_name}_recent_{'test'}_{Info.month_test}"
            # Info = to_namedtuple(joblib.load(f"{Info.model_wd_traintest}/Info~base_{'test'}.pkl"))

            print('# 测试集加工\n '.replace('#', '#' * 70))
            create_test[model_name] = tab_explore_create(base_test[model_name], Info, 'create', src)

        print('\n\n# 创建数据处理流水线、算法序列\n '.replace('#', '#' * 70))
        pipelines, models, skip = create_pipemodel(Info)
        print('\n\n# 训练测试过程\n '.replace('#', '#' * 70))
        train_res[model_name] = train_test_fun(Info, pipelines, models, skip, retrain_limit=25)
    except Exception as er:
        traceback.print_exc()
        train_err[model_name] = er
        if re.search('memory|unable to allocate',  str(train_err[model_name]).lower()):
            print('内存溢出，结束循环！')
            break
    finally:
        sys.stdout = original_out; sys.stderr = original_err
        if log_type == 'pkl':
            privy_log_save(*lo_paras, Info)
    print('----------------------------------------------------------------------\n\n\n ')


print('\n\n# 训练失败的模型\n '.replace('#', '#'*70))
if train_err:
    print(train_err)
else:
    print('无训练失败的模型')


print('\n\n# 检查个人维护的模型中是否有日数据进入模型\n '.replace('#', '#'*70))
_ = dayvalue_stat_fun(infos, s_table_info)


if log_type == 'pkl':
    print(f'\n加载日志文件：{log_pkl}')
    lt = joblib.load(log_pkl)
    lt.dirdict.keys()
    model_key = 'rhdc'
    lt.dirdict[model_key].keys()
    stepkey = 'traintest~202109~202110'
    lt.dirdict[model_key][stepkey]['log'].keys()
    print(lt.dirdict[model_key][stepkey]['log']['log~20220302122536'])
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# -------------------------------- <editor-fold desc="建模报告（！需手动挑选重要特征）"> -------------------------------
# 手动修改：挑选好解释的重要特征

model_name = '模型示例'
Info = to_namedtuple(infos.loc[model_name])

# ------- 获取模型信息、建模报告信息等
Info, iv_psi, psi_PN, f_init, f_add, data_month, dis_PN, pre_result = privy_get_iv_psi(Info)

# ------- 重要特征的分布情况（挑选好解释的重要特征）
iv_psi.loc[iv_psi.psi < 0.1].sort_values(by='iv', ascending=False)[['iv', 'psi', 'comment']].head(20)
iv_psi.loc[iv_psi.index.str.contains('family')]
iv_psi.loc[iv_psi.comment.str.contains('流量费')]

# -------选择字段，处理二维列联表并画图
col = iv_psi.index[0]  # 挑选好解释的重要特征
tab = privy_tab_modify(col, iv_psi, psi_PN, Info, bin_thred=None, prop_thred=None, rs_thred=0.7)
# 统一不同特征的y轴（正例占比）范围
xj = psi_PN['new'].loc[psi_PN['new'][Info.Pcase] > 5]
prop_lim = (xj[Info.Pcase] / xj.All).describe()[['min', 'max']].to_list()
# 特征分布图
privy_plot_tab(tab, Info, iv_psi, xtick_rotation=0, prop_lim=prop_lim)
# </editor-fold> -------------------------------------------------------------------------------------------------------

#
#
# ----------------------------------- <editor-fold desc="预测集加工 & 预测打分过程">  ----------------------------------
print('\n\n~ 预测集加工 & 预测打分过程 ~'.replace('~', '~'*60))

infos['month_predict'] = m_p
infos_pre1 = add_preinfos(infos)   # 添加预测所需信息

# 若有日数据，检查是否缺少相应账期的日数据（入模、条件字段）
infos_pre2 = dayvalue_stat_fun(infos_pre1,s_table_info, if_predict=True)

log_predict, base_predict, table_predict, pred_res, pred_err = {}, {}, {}, {}, {}
for model_name in infos.index:  # model_name = '模型示例'
    print(f'\n\n# {model_name} #'.replace('#', '#'*60))
    try:
        # <editor-fold desc="准备日志文件">
        Info_pre = to_namedtuple(infos_pre2.loc[model_name])
        lo_paras = privy_log_write(Info_pre, 'predict', log_type, log_pkl)
        lo = lo_paras[0]
        sys.stdout, sys.stderr = lo, lo
        # </editor-fold>

        # <editor-fold desc="加载模型训练结果">
        mark_traintest = re.sub('^.*traintest', '', Info_pre.model_wd_traintest)  # ~[宽表探索账期~]训练测试[~测试账期]
        file = f"{Info_pre.model_wd_traintest}/train_result{mark_traintest}.pkl"
        print(f"加载训练结果：{file}")  # 此处加载，主要为获得训练时的Info
        train_result = joblib.load(file)  # 最优模型的相关结果
        print('从训练结果中获取Info,并加入预测所需信息\n')
        Info = to_namedtuple({**train_result['Infoasdict'], **Info_pre._asdict()})
        if Info.dayvalue_delcon:
            Info = ch_con_fun(Info)
        # </editor-fold>

        print('\n\n# 加工预测账期近n月基础数据\n '.replace('#', '#'*70))
        base_predict[model_name], Info = privy_basedatafun('predict', Info, nmg_yaxin=nmg_yaxin)
        # base_predict[model_name] = f"{prefix}mid_{Info.short_name}_recent_{'predict'}_{Info.month_predict}"

        # <editor-fold desc="分批次加工预测集并预测打分">
        # 适用于内存不足以预测全量数据的环境：拆分预测数据，分批加工预测集、打分， 根据本地实际可修改更灵活智能的拆分方案
        _pices = {
            1: None,
            2: [['0', '1', '2', '3', '4'], ['5', '6', '7', '8', '9']],
            3: [['0', '1', '2', '3'], ['4', '5', '6', '7'], ['8', '9']],
            4: [['0', '1', '2'], ['3', '4', '5'], ['6', '7'], ['8', '9']],
            5: [['0', '1'], ['2', '3'], ['4', '5'], ['6', '7'], ['8', '9']],
            10: [[str(i)] for i in range(10)]}
        nround = 4  # 拆分份数
        print('\n')
        print(f"将数据拆分成 {nround} 份分别预测\n")
        for i, j in enumerate(_pices[nround]):
            # i=0;j=['0', '1', '2']
            print(f'=========================================== {i}份：{j} ==========================================')
            if nround > 1:
                print('修改Info，用于分批预测')  # 用户筛选已经在数据加工环节做过了，此处condition只需设置拆分数据条件
                ch = {'condition': f"right(cast({Info.col_id} as {type_py_sql[str]}), 1) in ({str(j)[1:-1]})",
                      'table_predict': Info.table_predict.replace('.csv', f'~{i}.csv'),
                      'table_score': Info.table_score.replace('.csv', f'~{i}.csv')
                      }
                Info_i = choradd_namedtuple(Info, ch)
            else:
                Info_i = Info
            print(f"condition: {Info_i.condition}\ntable_predict:{Info_i.table_predict}\ntable_score:{Info_i.table_score}")

            # 基于原有进程版本：
            print('\n\n# 预测集加工\n '.replace('#', '#'*70))
            table_predict[model_name] = tab_explore_create(base_predict[model_name], Info_i, 'create', src, if_condition=True)

            print('\n\n# 预测打分过程\n '.replace('#', '#'*70))  # 已经在加工数据环节筛选了用户 Info.condition
            pred_res[model_name] = predict_fun(train_result, Info_i, n_reason)

            # # 创建新进程版本(解决内存溢出报错后内存不释放，影响后续模型的打分)：
            # from multiprocessing import Process, Pipe
            # conn1, conn2 = Pipe(True)
            # args = (f"{model_name}_{i}_{j}", conn2, model_name, base_predict, Info_i._asdict(), train_result, n_reason, src)
            # sub_proc = Process(target=predict_job_func, args=args)
            # sub_proc.start()
            # # sub_proc.kill()  sub_proc.close()
            # sub_proc.join()
            # errecv = conn1.recv()
            # if errecv != 'succeed':
            #     raise Exception(errecv)
            print(f'==========================================================================================\n')
        # </editor-fold>

        print('\n\n# 整理分数数据\n '.replace('#', '#'*70))
        _ = privy_score_deal(Info, nround, n_reason=n_reason, woe_thred=0)

    except Exception as er:
        traceback.print_exc()
        pred_err[model_name] = er
        if re.search('memory|unable to allocate',  str(pred_err[model_name]).lower()):
            print('内存溢出，结束循环！')
            break
    finally:
        sys.stdout = original_out; sys.stderr = original_err
        if log_type == 'pkl':
            privy_log_save(*lo_paras, Info, True)


print('\n\n# 预测失败的模型\n '.replace('#', '#'*70))
if pred_err:
    print(pred_err)
else:
    print('无预测失败的模型')


if log_type == 'pkl':
    print(f'\n加载日志文件：{log_pkl}')
    lt = joblib.load(log_pkl)
    lt.dirdict.keys()
    model_key = 'rhdc'
    lt.dirdict[model_key].keys()
    stepkey = 'predictscore~202201'
    lt.dirdict[model_key][stepkey]['log'].keys()
    print(lt.dirdict[model_key][stepkey]['log']['log~20220302122536'])


print('\n\n# 汇总所有模型的分数\n '.replace('#', '#'*70))
# 本项目所有模型分数整合（若模型由多人维护，则此步骤交由一人统一整合本项目的所有模型，不可遗漏模型）
#   保存csv、导入数据库
#   weights不为None时：计算每个用户在每个模型清单中的综合得分model_level，以决定用户最终在哪个模型清单中输出
infos_all['month_predict'] = m_p
infos_all_pre = add_preinfos(infos_all)   # 添加预测所需信息
wd = './binaryclassify/allmodel'
if not os.path.isdir(wd):
    print(f'创建目录：{wd}')
    os.makedirs(wd)
csv_filename = f'{wd}/allmodel_scores~{m_p}.csv'  # 若无需保存至文件，则赋值为None
db_tablename = 'ml.binaryclassify_score_m'        # 若无需保存至数据库，则赋值为None
weights = {'marketlevel': 10, 'precision': 20, 'lift': 30}  # 无需设置，则赋值为None
allmodel_scores_fun(infos_all_pre, weights, csv_filename, db_tablename, user_limit_sql=None)  # infos_all_pre = infos_all_pre.iloc[[0]]
# </editor-fold>  ------------------------------------------------------------------------------------------------------


#
#
# -------------------------------- <editor-fold desc="模型分数监控"> ---------------------------------------------------
print('\n\n~ 模型分数监控 ~'.replace('~', '~'*66))
db_score = 'ml.table_score_month'
month_perform = '202012'
score_id = None   # None  'user_no'
n_ev = 6
col_exesam = None
privy_score_evaluate1(sqltype, month_perform, infos, db_score, score_id, n_ev, col_exesam, sqlcomment)
score_result, eval_tab = privy_score_evaluate2(infos, n_ev, col_exesam)
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# -------------------- <editor-fold desc="清理保存路径下大文件"> --------------------------------
print(f'\n\n# 查询模型保存路径下文件 \n'.replace('#', '#'*70))
allfile = getAllFilesize('./binaryclassify')
allfile.sort_values(by='size_MB', ascending=False)  # 查看大文件


# ############################### 确保所有模型正确训练完毕不会再返工重新训练：
mode = 'old'  # 'total'  /  'newest'  /   'old'

# -------- 删除训练集
train_data_csv = privy_selectfile(allfile, filecontain='^train_data~', dirupto='traintest~', mode=mode)
privy_delfile(train_data_csv)

# -------- 删除测试集
test_data_csv = privy_selectfile(allfile, filecontain='^test_data~', dirupto='traintest~', mode=mode)
privy_delfile(test_data_csv)

# -------- 删除模型集合
train_model_flows_pkl = privy_selectfile(allfile, filecontain='^train_model_flows~', dirupto='traintest~', mode=mode)
privy_delfile(train_model_flows_pkl)


# ############################### 确保所有模型正确预测完毕不会再返工重新预测：
mode2 = 'newest'
# -------- 删除预测集（可能是分批的）
predict_data_csv = privy_selectfile(allfile, filecontain='^predict_data~', dirupto='predictscore~', mode=mode2)
privy_delfile(predict_data_csv)

# -------- 删除分批保存的分数结果
predict_score_data_csv = privy_selectfile(allfile, filecontain='predict_score_data.*~\d{1,3}\.csv', dirupto='predictscore~', mode=mode2)
privy_delfile(predict_score_data_csv)
# </editor-fold> -------------------------------------------------------------------------------------------------------
