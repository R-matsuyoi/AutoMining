"""
AutoMining — 电信运营商客户经营大数据分类挖掘 主入口

用法:
    # 运行完整训练+预测流程
    python main.py

    # 仅运行训练测试
    python main.py --mode train

    # 仅运行预测打分
    python main.py --mode predict

    # 仅生成数据加工 SQL（不直连数据库时使用）
    python main.py --mode sql

    # 指定环境变量文件
    python main.py --env .env.production

    # 查看当前配置
    python main.py --show-config
"""

import argparse
import os
import sys
import warnings


def _patch_module_level_config():
    """
    在导入其他模块之前，将 .env 中的配置注入到各模块的全局变量中。
    这样做是为了不修改原有模块的功能逻辑，只在入口处统一覆盖配置。
    """
    from Mining.config import config

    # ---- 注入到 Mining.selfmodule.toolmodule.datatrans ----
    import Mining.selfmodule.toolmodule.datatrans as datatrans

    datatrans.db = config.DB_TYPE
    datatrans.prefix = config.DB_PREFIX
    datatrans.seconds = config.DB_SECONDS

    if config.DB_TYPE == "gp":
        datatrans.dbname = config.DB_NAME
        datatrans.user = config.DB_USER
        datatrans.pwd = config.DB_PWD
        datatrans.port = config.DB_PORT
        datatrans.host = config.DB_HOST
        datatrans.paradict = {
            "dbname": config.DB_NAME,
            "user": config.DB_USER,
            "pwd": config.DB_PWD,
            "port": config.DB_PORT,
            "host": config.DB_HOST,
        }

    # ---- 注入到 Mining.selfmodule.toolmodule.privy_outredirect ----
    import Mining.selfmodule.toolmodule.privy_outredirect as privy_out

    privy_out.sys_tem = config.SYS_TEM

    print(f"[Config] 数据库类型: {datatrans.db}")
    print(f"[Config] 数据库前缀: {datatrans.prefix}")
    print(f"[Config] 系统类型: {privy_out.sys_tem}")
    print(f"[Config] 模型目录: {config.MODELWD_PLATFORM}")
    print(f"[Config] 工作目录: {config.SELFWD_PLATFORM}")


def show_config():
    """打印当前所有配置"""
    from Mining.config import config

    print("\n" + "=" * 60)
    print("  AutoMining 当前配置")
    print("=" * 60)
    print(f"  平台 (PLAT):              {config.PLAT}")
    print(f"  系统类型 (SYS_TEM):       {config.SYS_TEM}")
    print(f"  数据库类型 (DB_TYPE):     {config.DB_TYPE}")
    print(f"  数据库主机 (DB_HOST):     {config.DB_HOST}")
    print(f"  数据库端口 (DB_PORT):     {config.DB_PORT}")
    print(f"  数据库名称 (DB_NAME):     {config.DB_NAME}")
    print(f"  数据库用户 (DB_USER):     {config.DB_USER}")
    print(f"  表名前缀 (DB_PREFIX):     {config.DB_PREFIX}")
    print(f"  模型目录 (MODELWD):       {config.MODELWD_PLATFORM}")
    print(f"  工作目录 (SELFDW):        {config.SELFWD_PLATFORM}")
    print(f"  SQL 类型 (SQL_TYPE):      {config.SQL_TYPE}")
    print(f"  日志类型 (LOG_TYPE):      {config.LOG_TYPE}")
    print(f"  自动衍生 (AUTO_PAIR2):    {config.AUTO_PAIR2}")
    print(f"  字段稳定度 (TABLE_PSI):   {config.TABLE_PSI}")
    print(f"  相关系数 (TABLE_R):       {config.TABLE_R}")
    print(f"  预测账期 (M_P):           {config.M_P}")
    print(f"  原因数 (N_REASON):        {config.N_REASON}")
    print("=" * 60 + "\n")


# =============================================================================
# 各运行模式的具体实现
# =============================================================================

def run_train_test(plat, modelwd_platform, selfwd_platform, default_values):
    """
    运行"数据加工 → 宽表探索 → 训练集加工 → 训练测试"流程。

    对应 privy_build.py 中"模型宽表探索 & 训练集、测试集加工 & 训练测试过程"部分。
    """
    from Mining.config import config

    # ---- 导入所有必需的模块 ----
    print("\n" + "~ " * 33 + " 导入包、模块 " + "~ " * 33)
    import traceback

    try:
        from sklearn.externals import joblib
    except ImportError:
        import joblib

    import pandas as pd

    pd.options.display.max_columns = 30
    pd.options.display.max_rows = 500
    pd.set_option("display.width", 100000)
    pd.set_option("display.unicode.ambiguous_as_wide", True)
    pd.set_option("display.unicode.east_asian_width", True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from Mining.selfmodule.toolmodule.privy_outredirect import privy_log_write, privy_log_save
    from Mining.selfmodule.binarymodule.modelinfo import privy_modelsummary
    from Mining.selfmodule.binarymodule.traintest import train_test_fun
    from Mining.selfmodule.binarymodule.pipemodel import create_pipemodel
    from Mining.selfmodule.tablemodule.tablefun import (
        tab_explore_create,
        privy_basedatafun,
        dayvalue_stat_fun,
    )
    from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, choradd_namedtuple
    from Mining.selfmodule.toolmodule.strtotable import infos_to_table
    from Mining.selfmodule.tablemodule.tablesql import (
        team_summary_sql,
        ywteam_xadd_sql,
        ywteam_target_sql,
        kduser_target_sql,
    )
    from Mining.selfmodule.toolmodule.datatrans import my_sql_fun

    original_out = sys.stdout
    original_err = sys.stderr

    log_type = config.LOG_TYPE
    log_pkl = None
    sqltype = config.SQL_TYPE
    sqlcomment = config.SQL_COMMENT

    # ---- 模型信息_动态 ----
    print("\n" + "~ " * 33 + " 模型信息_动态 " + "~ " * 33)

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

    col_dealvalue = "ALL"
    col_eval = ["trainproc_ratiolist"]

    infos = privy_modelsummary(infos_to_table(s_info_changed, col_dealvalue, col_eval))

    nmg_yaxin = {"本机测试": [""] * 2}.get(plat, [""] * 2)

    auto_pair2 = config.AUTO_PAIR2
    diff_limit = config.DIFF_LIMIT
    table_psi = config.TABLE_PSI
    table_r = config.TABLE_R
    src = config.SRC

    # ---- 生成数据加工 SQL 并执行 ----
    print("\n" + "~ " * 33 + " 生成数据加工 SQL " + "~ " * 33)

    month_list = ["202203"]

    print(f"\n\n\n{sqlcomment} 加工：移网基础宽表_用户to套餐组")
    for month in month_list:
        print(f"----------------------------- {month} ---------------------------------")
        team_summary = team_summary_sql(month)
        print(team_summary)
        if sqltype == "execute":
            my_sql_fun(team_summary, method="execute")

    # ---- 模型宽表探索 & 训练测试 ----
    print("\n" + "~ " * 25 + " 模型宽表探索 & 训练集、测试集加工 & 训练测试过程 " + "~ " * 25)

    log_train, base_train, create_train, base_test, create_test, train_res, train_err = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )

    for model_name in infos.index:
        print("# %s #".replace("#", "#" * 80) % model_name)
        try:
            Info = to_namedtuple(infos.loc[model_name])
            ch = {
                "auto_pair2": auto_pair2,
                "diff_limit": diff_limit,
                "table_psi": table_psi,
                "table_r": table_r,
                "col_mark": ["user_acct_month", "data_use", Info.col_month, Info.col_id],
            }
            Info = choradd_namedtuple(Info, ch)

            lo_paras = privy_log_write(Info, "traintest", log_type, log_pkl)
            lo = lo_paras[0]
            sys.stdout, sys.stderr = lo, lo

            print("\n\n# 加工训练账期近n月基础数据\n ".replace("#", "#" * 70))
            base_train[model_name], Info = privy_basedatafun("train", Info, nmg_yaxin=nmg_yaxin)

            print("\n\n# 探索模型宽表\n ".replace("#", "#" * 70))
            exp, Info = tab_explore_create(base_train[model_name], Info, "explore", src)

            print("\n\n# 训练集加工\n ".replace("#", "#" * 70))
            create_train[model_name], Info = tab_explore_create(base_train[model_name], Info, "create", src)

            if str(Info.month_test) != "nan":
                print("\n\n# 加工测试账期近n月基础数据\n ".replace("#", "#" * 70))
                base_test[model_name], Info = privy_basedatafun("test", Info, nmg_yaxin=nmg_yaxin)
                print("# 测试集加工\n ".replace("#", "#" * 70))
                create_test[model_name] = tab_explore_create(base_test[model_name], Info, "create", src)

            print("\n\n# 创建数据处理流水线、算法序列\n ".replace("#", "#" * 70))
            pipelines, models, skip = create_pipemodel(Info)
            print("\n\n# 训练测试过程\n ".replace("#", "#" * 70))
            train_res[model_name] = train_test_fun(Info, pipelines, models, skip, retrain_limit=25)

        except Exception as er:
            traceback.print_exc()
            train_err[model_name] = er
            import re as _re

            if _re.search("memory|unable to allocate", str(train_err[model_name]).lower()):
                print("内存溢出，结束循环！")
                break
        finally:
            sys.stdout = original_out
            sys.stderr = original_err
            if log_type == "pkl":
                privy_log_save(*lo_paras, Info)
        print("----------------------------------------------------------------------\n\n\n ")

    # 汇报失败模型
    print("\n\n# 训练失败的模型\n ".replace("#", "#" * 70))
    if train_err:
        print(train_err)
    else:
        print("无训练失败的模型")

    return train_res


def run_predict(plat, modelwd_platform, selfwd_platform, default_values):
    """
    运行"预测集加工 → 预测打分"流程。

    对应 privy_build.py 中"预测集加工 & 预测打分过程"部分。
    """
    from Mining.config import config

    import traceback

    try:
        from sklearn.externals import joblib
    except ImportError:
        import joblib

    import pandas as pd

    pd.options.display.max_columns = 30
    pd.options.display.max_rows = 500
    pd.set_option("display.width", 100000)

    from Mining.selfmodule.toolmodule.privy_outredirect import privy_log_write, privy_log_save
    from Mining.selfmodule.binarymodule.modelinfo import privy_modelsummary, add_preinfos, ch_con_fun
    from Mining.selfmodule.binarymodule.predictscore import predict_fun, privy_score_deal
    from Mining.selfmodule.tablemodule.tablefun import tab_explore_create, privy_basedatafun
    from Mining.selfmodule.toolmodule.dataprep import to_namedtuple, choradd_namedtuple
    from Mining.selfmodule.toolmodule.strtotable import infos_to_table
    from Mining.selfmodule.toolmodule.datatrans import type_py_sql

    original_out = sys.stdout
    original_err = sys.stderr

    log_type = config.LOG_TYPE
    log_pkl = None
    n_reason = config.N_REASON
    m_p = config.M_P
    src = config.SRC

    # 重新加载模型信息用于预测
    s_info_changed = """model_name	month_train	month_test	Pcase_limit	 traintable_ratio	Pcumsum_limit	 timein_count	 timeout_limit	 trainproc_ratiolist	iv_limit	r_limit	marketlevel
模型示例     	202010	202012	1000	2	2	500	1000	[1, 2, 10]	0.05	0.95	1
"""

    col_dealvalue = "ALL"
    col_eval = ["trainproc_ratiolist"]

    infos = privy_modelsummary(infos_to_table(s_info_changed, col_dealvalue, col_eval))

    infos["month_predict"] = m_p
    infos_pre = add_preinfos(infos)

    log_predict, base_predict, table_predict, pred_res, pred_err = {}, {}, {}, {}, {}

    for model_name in infos.index:
        print(f"\n\n# {model_name} #".replace("#", "#" * 60))
        try:
            Info_pre = to_namedtuple(infos_pre.loc[model_name])
            lo_paras = privy_log_write(Info_pre, "predict", log_type, log_pkl)
            lo = lo_paras[0]
            sys.stdout, sys.stderr = lo, lo

            # 加载训练结果
            mark_traintest = Info_pre.model_wd_traintest.replace("^.*traintest", "", regex=True)
            file = f"{Info_pre.model_wd_traintest}/train_result{mark_traintest}.pkl"
            print(f"加载训练结果：{file}")
            train_result = joblib.load(file)
            Info = to_namedtuple({**train_result["Infoasdict"], **Info_pre._asdict()})
            if Info.dayvalue_delcon:
                Info = ch_con_fun(Info)

            print("\n\n# 加工预测账期近n月基础数据\n ".replace("#", "#" * 70))
            base_predict[model_name], Info = privy_basedatafun("predict", Info, nmg_yaxin=[""] * 2)

            # 分批次预测
            _pices = {
                1: None,
                2: [["0", "1", "2", "3", "4"], ["5", "6", "7", "8", "9"]],
                3: [["0", "1", "2", "3"], ["4", "5", "6", "7"], ["8", "9"]],
                4: [["0", "1", "2"], ["3", "4", "5"], ["6", "7"], ["8", "9"]],
                5: [["0", "1"], ["2", "3"], ["4", "5"], ["6", "7"], ["8", "9"]],
                10: [[str(i)] for i in range(10)],
            }
            nround = 4

            for i, j in enumerate(_pices[nround]):
                print(
                    f"=========================================== {i}份：{j} =========================================="
                )
                if nround > 1:
                    ch = {
                        "condition": f"right(cast({Info.col_id} as {type_py_sql[str]}), 1) in ({str(j)[1:-1]})",
                        "table_predict": Info.table_predict.replace(".csv", f"~{i}.csv"),
                        "table_score": Info.table_score.replace(".csv", f"~{i}.csv"),
                    }
                    Info_i = choradd_namedtuple(Info, ch)
                else:
                    Info_i = Info

                print("\n\n# 预测集加工\n ".replace("#", "#" * 70))
                table_predict[model_name] = tab_explore_create(
                    base_predict[model_name], Info_i, "create", src, if_condition=True
                )

                print("\n\n# 预测打分过程\n ".replace("#", "#" * 70))
                pred_res[model_name] = predict_fun(train_result, Info_i, n_reason)

                print("=" * 90 + "\n")

            print("\n\n# 整理分数数据\n ".replace("#", "#" * 70))
            _ = privy_score_deal(Info, nround, n_reason=n_reason, woe_thred=0)

        except Exception as er:
            traceback.print_exc()
            pred_err[model_name] = er
            import re as _re

            if _re.search("memory|unable to allocate", str(pred_err[model_name]).lower()):
                print("内存溢出，结束循环！")
                break
        finally:
            sys.stdout = original_out
            sys.stderr = original_err
            if log_type == "pkl":
                privy_log_save(*lo_paras, Info, True)

    print("\n\n# 预测失败的模型\n ".replace("#", "#" * 70))
    if pred_err:
        print(pred_err)
    else:
        print("无预测失败的模型")

    return pred_res


def run_sql_only():
    """仅生成数据加工 SQL（不连接数据库时使用）。"""
    from Mining.config import config

    import pandas as pd

    pd.options.display.max_columns = 30
    pd.options.display.max_rows = 500
    pd.set_option("display.width", 100000)

    from Mining.selfmodule.tablemodule.tablesql import (
        team_summary_sql,
        ywteam_xadd_sql,
        ywteam_target_sql,
        kduser_target_sql,
    )

    month_list = ["202203"]
    sqlcomment = config.SQL_COMMENT

    print(f"\n\n\n{sqlcomment} 加工：移网基础宽表_用户to套餐组")
    for month in month_list:
        print(f"----------------------------- {month} ---------------------------------")
        print(team_summary_sql(month))

    print(f"\n\n\n{sqlcomment} 加工：移网特征补充表_套餐组")
    for month in month_list:
        print(f"----------------------------- {month} ---------------------------------")
        print(ywteam_xadd_sql(month))

    print(f"\n\n\n{sqlcomment} 加工：移网目标表_套餐组")
    for month in month_list:
        print(f"----------------------------- {month} ---------------------------------")
        print(ywteam_target_sql(month))

    print(f"\n\n\n{sqlcomment} 加工：宽带目标表_用户")
    for month in month_list:
        print(f"----------------------------- {month} ---------------------------------")
        print(kduser_target_sql(month))


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AutoMining — 电信运营商客户经营大数据分类挖掘",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                        # 运行完整流程
  python main.py --mode train           # 仅训练测试
  python main.py --mode predict         # 仅预测打分
  python main.py --mode sql             # 仅生成 SQL
  python main.py --show-config          # 查看配置
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["all", "train", "predict", "sql"],
        default="all",
        help="运行模式: all=完整流程, train=训练测试, predict=预测打分, sql=仅生成SQL (默认: all)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="环境变量文件路径 (默认: 自动查找项目根目录下的 .env)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="显示当前配置后退出",
    )

    args = parser.parse_args()

    # 如果指定了 --env，设置环境变量
    if args.env:
        os.environ["AUTOMINING_ENV_FILE"] = args.env

    # 显示配置
    if args.show_config:
        show_config()
        return

    # 注入配置到全局变量
    print("=" * 60)
    print("  AutoMining 启动中...")
    print("=" * 60)

    _patch_module_level_config()

    from Mining.config import config

    plat = config.PLAT
    modelwd_platform = config.MODELWD_PLATFORM
    selfwd_platform = config.SELFWD_PLATFORM
    default_values = config.get_default_values()

    print(f"\n[启动] 平台: {plat}")
    print(f"[启动] 模式: {args.mode}")
    print(f"[启动] 模型目录: {modelwd_platform}")

    try:
        if args.mode == "all":
            print("\n>>> 阶段 1: 训练测试")
            run_train_test(plat, modelwd_platform, selfwd_platform, default_values)
            print("\n>>> 阶段 2: 预测打分")
            run_predict(plat, modelwd_platform, selfwd_platform, default_values)
        elif args.mode == "train":
            run_train_test(plat, modelwd_platform, selfwd_platform, default_values)
        elif args.mode == "predict":
            run_predict(plat, modelwd_platform, selfwd_platform, default_values)
        elif args.mode == "sql":
            _patch_module_level_config()
            run_sql_only()

        print("\n" + "=" * 60)
        print("  AutoMining 执行完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n[错误] 执行失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

