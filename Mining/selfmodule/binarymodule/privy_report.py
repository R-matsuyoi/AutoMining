from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pylab import mpl
from matplotlib.pyplot import savefig
import time

from Mining.selfmodule.binarymodule.modelinfo import *
from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, get_index
from Mining.selfmodule.binarymodule.predictscore import privy_get_trans
from Mining.selfmodule.tablemodule.tablefun import month_list_fun, month_add

seconds = 3


def privy_get_iv_psi(Info, col_ignore=None):
    """
    加载单个模型的训练结果，整理建模报告所需内容（根据实际需求调整内容）
    :param Info: 单个模型的模型信息
    :param col_ignore: 统计字段信息时忽略的字段列表
    :return:
    """
    print('获取模型的路径等')
    model_wd_traintest = Info.model_wd_traintest
    mark_traintest = re.sub('^.*traintest', '', model_wd_traintest)  # ~[宽表探索账期~]训练测试[~测试账期]
    file_train = f"{model_wd_traintest}/train_result{mark_traintest}.pkl"  # 宽表探索账期~训练测试数据账期
    print(f"\n加载训练结果：{file_train}")
    train_result = joblib.load(file_train)  # 最优模型的相关结果
    Info = to_namedtuple(train_result['Infoasdict'])

    # file_tabexp = f'{model_wd_traintest}/tabexp_col_obj~{Info.month_tabexp}.pkl'
    # print(f"加载宽表探索：{file_tabexp}")
    # tabexp_col_obj = joblib.load(file_tabexp)
    comment_all = Info.comment_all  # tabexp_col_obj['comment_all']
    comment_valid = comment_all.loc[comment_all.是否宽表字段 == '是']

    print('\n获取pipeline1')
    if 'pipeline1' in train_result['flow_name']:
        print('最优模型的流水线是pipeline1，从train_result中提取')
        pipeline1 = train_result['pipeline']
    else:
        print('最优模型的流水线不是pipeline1， 加载train_model_flows，从中获取pipeline1')
        file_flows = f"{model_wd_traintest}/train_model_flows{mark_traintest}.pkl"
        print(f"    加载训练结果：{file_flows}")
        train_model_flows = joblib.load(file_flows)
        # 获取pipeline1
        flows_key = [i for i in train_model_flows.keys() if 'pipeline1' in i]
        if len(flows_key) == 0:
            raise Exception('train_model_flows中无pipeline1，无法继续，请采用其他方式！')
            pipeline1 = None
        else:
            print('    从train_model_flows中获取pipeline1')
            pipeline1 = train_model_flows[flows_key[0]]['pipeline']

    print('\n从流水线中获取WoeTransformer_DF')
    woe1 = privy_get_trans(pipeline1, 'WoeTransformer_DF')
    if woe1 is None:
        s = f"请正确获取WoeTransformer_DF"
        raise Exception(s)

    psi_key = f"psi_PN_{Info.ev_key}"
    print(f'\n稳定性: {psi_key}')
    col_iv = woe1.col_iv_[pipeline1.transform_out_colnames_]; col_iv.name = 'iv'
    psi_PN = woe1.psi_.col_psi_PN_[psi_key]
    col_psi = DataFrame(psi_PN.index.to_list(), columns=['col', 'psi', 'value']).set_index('col').psi.drop_duplicates()
    iv_psi = pd.concat([col_iv, col_psi], axis=1)
    iv_psi = pd.merge(iv_psi, comment_all[['field_name', 'comment', 'dtype_classify', 'field_src']],
                      left_index=True, right_on='field_name', how='left').set_index('field_name').sort_values(by='iv', ascending=False)
    na_iv = iv_psi.isnull().sum(axis=1)
    if na_iv.sum():
        s = f"iv_psi存在 {na_iv.sum()} 个缺失值:\n{iv_psi[na_iv > 0]}"
        warnings.warn(s); time.sleep(seconds)

    print('\n《建模报告》：分析目标用户群')
    print(Info.condition)

    print('\n确定报告字段信息')
    col_print = ['field_name', 'comment', '是否入模']
    col_ignore = (set(col_ignore) | {'intercept'}) if col_ignore else {'intercept'}
    f_model = [i for i in train_result['model'].fit_in_colnames_ if i not in col_ignore]
    print(f'入模字段：{len(f_model)}个')

    # 原始字段
    pd.set_option('display.width', 1000)
    f_init = comment_valid.loc[comment_valid.field_src == '原始'].copy()
    f_init['是否入模'] = '否'
    f_init.loc[f_init.field_name.isin(f_model) | f_init.field_name.isin([re.sub('~.*$', '', i) for i in f_model]), '是否入模'] = '是'
    f_init.loc[~f_init.field_name.isin(col_ignore)]
    f_init.index = range(len(f_init))

    # 衍生字段
    f_add = comment_valid.loc[comment_valid.field_src != '原始'].copy()
    f_add.field_src = f_add.field_src.str.replace('自动|手动', '')
    f_add['是否入模'] = '否'
    f_add.loc[f_add.field_name.isin(f_model) | f_add.field_name.isin([re.sub('~.*$', '', i) for i in f_model]), '是否入模'] = '是'
    f_add.loc[~f_add.field_name.isin(col_ignore)]
    f_add.index = range(len(f_add))

    # 核对
    c1 = set(f_model)
    c2 = set(f_init.loc[f_init.是否入模 == '是', 'field_name']) | set(f_add.loc[f_add.是否入模 == '是', 'field_name'])
    into_lack = {re.sub('~.*$', '', i) for i in (c1 - c2)} - c2
    into_more = (c2 - c1) - {re.sub('~.*$', '', i) for i in (c2 - c1)}

    s = f"入模字段总数与原始、衍生字段入模字段不能完全对应，请检查，"
    if into_lack:
        raise Exception(s + f'缺少：{into_lack}')
    if into_more:
        raise Exception(s + f'多出：{into_more}')

    print(f"\n《建模报告》：基础字段：{len(f_init)}个，入模：{(f_init.是否入模 == '是').sum()}个")
    print(f_init[col_print])
    print(f"\n《建模报告》：衍生字段：{len(f_add)}个，入模：{(f_add.是否入模 == '是').sum()}个")
    print(f_add[col_print])

    print('\n《建模报告》：时间窗口设计')
    data_month = '	'.join(['训练'] + month_list_fun(month_add(str(Info.month_train), 1), periods=-4))
    if str(Info.timein_count) != 'nan':
        data_month = data_month + '\n验证'
    if str(Info.month_test) != 'nan':
        m2 = '	'.join(['测试'] + month_list_fun(month_add(str(Info.month_test), 1), periods=-4))
        data_month = '\n'.join([data_month, m2])
    print(data_month)

    print('\n《建模报告》：训练集、验证集、测试集')
    dis_PN = train_result['dis_PN']
    dis_PN = '\n'.join(dis_PN.apply(lambda x: '	'.join(x.astype(str)), axis=1))
    print(dis_PN)
    r = re.sub('.*_', '', train_result['flow_name'].replace(' ', '').split('|')[0])
    print(f"对此模型采取的抽样比例为1：{r}。")

    print('\n《建模报告》：效果评估')
    def fun(v):
        col_pre = ['累计人数占比', '累计人数', '累计人数_' + str(Info.Pcase), '累计查准率', '累计查全率', '累计提升度']
        v = v.iloc[:Series((v.分数 == '').values).idxmax()][col_pre]  # 兼容版本
        r1, r2 = v.累计查准率.str.strip('%').astype(float) / 100, v.累计查全率.str.strip('%').astype(float) / 100
        v['F1'] = round(2 * (r1 * r2) / (r1 + r2), 3)
        return v
    pre_result = {k: fun(v) for k, v in train_result['pre_result'].items()}
    for k, v in pre_result.items():
        print(f"-------------------------------------- {k} ----------------------------------------")
        print(v)

    print('\n《建模报告》：排行前20的重要特征可视化')
    field_import = train_result['field_import']
    col_im = field_import.columns[-1]
    field_import = field_import.loc[~field_import.field_name.isin(col_ignore)]
    top_import = field_import.iloc[:20].sort_values(by=col_im)
    print(top_import)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.3, right=0.7, top=0.9, bottom=0.1)
    plt.barh(range(len(top_import)), top_import[col_im], tick_label=top_import.comment, color='#6699CC')
    plt.title('特征重要性', loc='center', fontsize='20', fontweight='bold', color='#6699CC')
    plt.xticks([])
    plt.show()
    return Info, iv_psi, psi_PN, f_init[col_print], f_init[col_print], data_month, dis_PN, pre_result


def privy_my_sort_index(x, level=None, col_type='数值型', ascending=True):
    x = x.copy()
    if col_type == '类别型':
        return x.sort_values(by='prop')
    elif col_type == '数值型':
        index = get_index(x, level) if level else x.index
        x['s'] = index.map(lambda x: x.strip('(|)|[|]').split(',')[0]).astype(float).values
        return x.sort_values(by='s', ascending=ascending).drop(columns='s')


def privy_combine_bin(tab, idx1, idx2, Info):
    """
    合并组别
    :param tab:
    :param idx1:
    :param idx2:
    :return:
    """
    def get_interval(inter):
        left_v, right_v = inter[1:-1].replace(' ', '').split(',')
        left_s, right_s = inter[0], inter[-1]
        return left_s, left_v, right_v, right_s
    tab = tab.copy()
    idxname = tab.index.name
    vs1 = get_interval(idx1)
    vs2 = get_interval(idx2)
    if float(vs1[1]) > float(vs2[1]):
        vs_min, vs_max = vs2, vs1
    elif float(vs1[1]) < float(vs2[1]):
        vs_min, vs_max = vs1, vs2
    else:
        s = f"privy_combine_bin函数：区间取值异常，{idx1}，{idx2}"
        raise Exception(s)
    if vs_min[2] != vs_max[1]:
        s = f"privy_combine_bin函数：不可合并非相邻区间，{idx1}，{idx2}"
        raise Exception(s)
    else:
        print(f'合并区间：{idx1}，{idx2}')
    index_com = f"{vs_min[0]}{vs_min[1]}, {vs_max[-2]}{vs_max[-1]}"
    tab_del = tab.loc[tab.index.isin([idx1, idx2])]
    tab_com = DataFrame(tab_del.sum()).T
    tab_com.index = [index_com]
    tab_com['prop'] = tab_com[Info.Pcase] / tab_com['All']
    tab_new = tab.loc[~tab.index.isin(tab_del.index)]
    tab_new = pd.concat([tab_new, tab_com])
    tab_new.index.name = idxname
    return privy_my_sort_index(tab_new)


def privy_com_smallbin(tab, bin_thred, Info):
    """
    合并样本量<bin_thred的组别
    :param tab: 输入二位列联表
    :param bin_thred: 组别的样本量阈值
    :return:
    """
    tab = tab.copy()
    print(f'输入：\n{tab}\n')
    while (tab.All.min() < bin_thred) & (len(tab) > 1):
        posi_small = tab.All.argmin()
        idx_small = tab.index[posi_small]
        tab_samll = tab.iloc[[posi_small]]
        idx_near = list({posi_small-1, posi_small+1} & set(range(len(tab))))
        tab_near = tab.iloc[idx_near]
        idx_nearest = (tab_near.prop - tab_samll.prop.values).idxmin()
        tab = privy_combine_bin(tab, idx_small, idx_nearest, Info)
        print(tab, '\n')
    else:
        print(f'不再合并')
    return tab


def privy_com_prop_similar(tab, prop_thred, Info):
    """
    合并prop差距<prop_thred的相邻组别
    :param tab: 输入二位列联表
    :param prop_thred: prop差距阈值
    :return:
    """
    tab = tab.copy()
    tab['prop_sub'] = [np.nan] + list(tab.prop.values[1:] - tab.prop.values[:-1])
    print(f'输入：\n{tab}\n')
    while tab['prop_sub'].abs().min() < prop_thred:
        posi_least = tab['prop_sub'].abs().argmin()
        idx_least = tab.index[posi_least]
        idx_prev = tab.index[posi_least-1]
        tab = privy_combine_bin(tab, idx_prev, idx_least, Info)
        tab['prop_sub'] = [np.nan] + list(tab.prop.values[1:] - tab.prop.values[:-1])
        print(tab, '\n')
    else:
        print(f"不再合并")
    return tab.drop(columns='prop_sub')


def privy_com_turnpoint(tab, rs_thred, Info):
    """
    合并prop弯折过大的组别
    :param tab: 输入二位列联表
    :param rs_thred: 回归的拟合优度阈值
    :return:
    """
    if len(tab) <= 3:
        print(f"tab的行数（{len(tab)}）<=3，不再合并")
        return tab
    tab = tab.copy()
    col_out = tab.columns
    print(f'输入：\n{tab}')
    tab['prop_sub'] = [np.nan] + list(tab.prop.values[1:] - tab.prop.values[:-1])
    tab['intercept'] = 1
    tab['x'] = range(len(tab))
    tab['x2'] = tab['x'] * tab['x']
    model1 = sm.OLS(tab.prop, tab[['intercept', 'x']]).fit()
    model2 = sm.OLS(tab.prop, tab[['intercept', 'x', 'x2']]).fit()
    print(f"model1.rsquared: {model1.rsquared}")
    print(f"model2.rsquared: {model2.rsquared}\n")

    while (model1.rsquared < rs_thred) & (model2.rsquared < rs_thred):
        posi_distant = tab['prop_sub'].abs().argmax()
        idx_distant = tab.index[posi_distant]
        idx_prev = tab.index[posi_distant-1]
        tab = privy_combine_bin(tab, idx_prev, idx_distant, Info)
        tab['prop_sub'] = [np.nan] + list(tab.prop.values[1:] - tab.prop.values[:-1])
        print(tab[col_out])
        tab['intercept'] = 1
        tab['x'] = range(len(tab))
        tab['x2'] = tab['x'] * tab['x']
        model1 = sm.OLS(tab.prop, tab[['intercept', 'x']]).fit()
        model2 = sm.OLS(tab.prop, tab[['intercept', 'x', 'x2']]).fit()
        print(f"model1.rsquared: {model1.rsquared}")
        print(f"model2.rsquared: {model2.rsquared}\n")
    else:
        print(f'不再合并')
    return tab.drop(columns=set(['prop_sub', 'intercept', 'x', 'x2']) & set(tab.columns))


def privy_tab_modify(col, iv_psi, psi_PN, Info, bin_thred=None, prop_thred=None, rs_thred=0.7):
    """
    获取特征字段与目标字段的二维列联表，并合并
    :param col: 特征字段名
    :param iv_psi: 所有特征字段的iv与psi取值
    :param psi_PN: 所有字段的训练集、新数据集中特征字段与目标字段的二维列联表
    :param Info: 模型信息
    :param bin_thred: 每个组别样本量阈值，小于则合并
    :param prop_thred: 相邻两组别正例占比之差的阈值，小于则合并
    :param rs_thred: 合并弯折过大的组别时，回归模型的拟合优度阈值，小于则合并
    :return: 合并后的列联表
    """
    col_type = iv_psi.dtype_classify.loc[col]
    tab = psi_PN.loc[col]['new'][[Info.Pcase, Info.Ncase, 'All']]
    tab.index = tab.index.map(lambda x: x[1])
    tab.index.name = (col, iv_psi.comment[col])
    tab['prop'] = tab[Info.Pcase] / tab['All']
    tab = privy_my_sort_index(tab, col_type=col_type)
    print(f"初始tab:\n{tab}\n")
    count_All = tab.All.sum()
    count_Pcase = tab[Info.Pcase].sum()
    prop_natural = count_Pcase / count_All
    if col_type == '数值型':
        pass
    elif col_type == '类别型':
        print('非数值型字段，暂不处理')
        return tab

    while tab.All.isnull().sum():
        posti_na = tab.All.isnull().argmax()
        idx_na = tab.index[posti_na]
        print(f"{idx_na}为空")
        idx_near = list({posti_na-1, posti_na+1} & set(range(len(tab))))[0]
        idx_near = tab.index[idx_near]
        tab = privy_combine_bin(tab, idx_na, idx_near, Info)

    if bin_thred is None:
        bin_thred = min(500, count_All/100)
        print(f'未设置bin_thred， 将其赋值为{bin_thred}')
    if prop_thred is None:
        prop_thred = min(1/100, prop_natural/2)
        print(f'未设置prop_thred， 将其赋值为{prop_thred}\n')
    print(f'--------------------------- 合并样本量<{bin_thred}的组别 -------------------------------')
    tab = privy_com_smallbin(tab, bin_thred, Info)
    print(f'---------------------- 合并prop差距<{prop_thred}的相邻组别 -----------------------------')
    tab = privy_com_prop_similar(tab, prop_thred, Info)
    print('------------------------------- 合并prop弯折过大的组别 ----------------------------------')
    tab = privy_com_turnpoint(tab, rs_thred, Info)
    return tab


def privy_plot_tab(tab, Info, iv_psi=None, if_save=False, xtick_rotation=0, prop_lim=None):
    """
    可视化
    :param tab: 特征字段与目标字段的二维列联表
    :param iv_psi: 所有特征字段的iv与psi取值
    :param if_save: 是否保存图片
    :param xtick_rotation: x轴标签旋转
    :param prop_lim: 右侧y轴刻度范围
    :return:
    """
    def to_percent(value, position):
        return "%.3f" % (100 * value) + '%';
    # %config InlineBackend.figure_format = 'svg';
    mpl.rcParams['font.sans-serif'] = ['SimHei'];
    bar_width = 0.4;  # 设置柱形图宽度
    fig, ax1 = plt.subplots(figsize=(10, 8));

    for xtick in ax1.get_xticklabels():
        xtick.set_rotation(xtick_rotation)

    count_All = tab.All.sum()
    count_Pcase = tab[Info.Pcase].sum()
    prop_natural = count_Pcase / count_All

    x1 = tab.index;
    y_N = tab[Info.Ncase];
    y_P = tab[Info.Pcase];
    y_prop = tab.prop;

    plt.bar(x1, y_P, bar_width, align="center", color="c", label="正例", alpha=0.5);
    plt.bar(np.arange(len(x1)) + bar_width, y_N, bar_width, color="b", align="center", label="负例", alpha=0.5);
    plt.legend(loc=[0.015, 0.9]);

    ax2 = ax1.twinx();
    ax2.yaxis.set_major_formatter(FuncFormatter(to_percent));
    if prop_lim == 'auto':
        prop_lim = [prop_natural / 10, min(prop_natural * 10, 1)]
    if prop_lim is not None:
        ax2.set_ylim(prop_lim)
    line = ax2.plot(x1, y_prop, c='r');
    for a, b in zip(x1, y_prop):
        ax2.text(a, b, "%.2f" % (100 * b) + '%', ha='center', va='bottom', fontsize=10);
    ax2.legend(line, ("正例占比",), loc=[0.88, 0.95]);
    if iv_psi is not None:
        col_iv_psi = ', '.join(round(iv_psi.loc[iv_psi.index == tab.index.name[0], ['iv', 'psi']].iloc[0],4).astype(str));
        col_iv_psi = f"iv, psi:{col_iv_psi} "
    else:
        col_iv_psi = ''
    title = f"{col_iv_psi}" + '\n\n'.join(tab.index.name);
    ax2.set_title(f"{title}", fontsize=16);

    ax3 = ax1.twinx();
    ax3.set_yticks([])
    hline = ax3.axhline(y=prop_natural, color='r', linestyle='dashed', label='自然率')
    ax3.legend(loc=[0.88, 0.9])

    plt.show();

    if if_save:
        print(if_save);
        wd = f"{Info.model_wd_traintest}/report";
        file = f"{wd}/{tab.index.name[0]}.jpg";
        print(f"将图形保存至{file}");
        savefig(file);
