import os
import sys
import traceback

try:
    from Mining.selfmodule.tablemodule.tablefun import tab_explore_create
    from Mining.selfmodule.binarymodule.predictscore import predict_fun
    from Mining.selfmodule.toolmodule.dataprep import to_namedtuple
    pass  # 用于隐式导入时，将所有代码写入testcode时，将剔除from Mining.selfmodule...行，此处为try后的内容占个位，否则报错
except:
    pass


def predict_job_func(job_name: str, conn, model_name, base_predict, Info_i_asdict, train_result, n_reason, src):
    """"""

    # print(conn.recv())  # Data@parentprocess
    print(f"job_name: {job_name}")
    print(f"pid: {os.getpid()}")
    Info_i = to_namedtuple(Info_i_asdict)

    table_predict, pred_res = {}, {}

    try:
        print('\n\n# 预测集加工\n '.replace('#', '#' * 70))
        table_predict[model_name] = tab_explore_create(base_predict[model_name], Info_i, 'create', src, if_condition=True)

        print('\n\n# 预测打分过程\n '.replace('#', '#' * 70))  # 已经在加工数据环节筛选了用户 Info.condition
        pred_res[model_name] = predict_fun(train_result, Info_i, n_reason)

        conn.send("succeed")
    except Exception as er:
        traceback.print_exc()
        conn.send(er)
    finally:
        sys.exit()


