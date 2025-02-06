# 电信运营商客户经营需求驱动的大数据分类挖掘程序

## 一、项目概述

本代码实现了一个完整的机器学习建模与预测流程，涵盖数据加工、模型训练、测试评估、预测打分及结果监控等环节。项目针对多业务场景（如流失预警、套餐迁移、流量包推荐等）构建了多个模型，支持跨数据平台运行，并通过模块化设计实现高复用性。


## 二、核心功能解析

### 1. 环境配置与依赖管理
多平台适配：通过plat变量动态配置工作目录、数据库连接参数（如Hive/PostgreSQL）、文件存储路径等，支持本地测试与生产环境无缝切换。

依赖注入：default_values定义数据库连接、日志类型等全局参数，确保代码在不同环境中灵活执行。

代码无痕上传：通过privy_upload_code将代码隐藏上传至平台的.pkl文件，保障代码安全性。

``` python
#示例配置
default_values = {
    'sys_tem': {'本机测试': 'win', '亚信jupyter': 'linux'}[plat],
    'db': {'本机测试': 'gp', '亚信jupyter': 'hive'}[plat],
    'prefix': {'本机测试': 'ml.', '亚信jupyter': 'kjdb.'}[plat]
}
```

### 2. 数据加工与特征工程

SQL动态生成：
``` python
team_summary = team_summary_sql(month)
my_sql_fun(team_summary, method='execute')
# 执行或打印SQL 根据模型需求自动生成数据加工SQL，包括用户宽表、套餐组特征表、目标表等。
```

数据分布检查：通过table_exam_sql统计字段分布稳定性，识别异常字段（如PSI>0.1时告警）。

特征衍生：支持字段两两自动衍生（如加减乘除），并通过iv_limit和r_limit过滤低IV值或高相关性特征。

### 3. 模型训练与测试

``` python
for model_name in infos.index:
    Info = to_namedtuple(infos.loc[model_name])
    pipelines, models = create_pipemodel(Info)  # 构建流水线
    train_res[model_name] = train_test_fun(Info, pipelines, models, skip)
```

流水线构建：通过create_pipemodel创建包含特征处理、算法模型的流水线，支持自定义预处理（如标准化、分箱）。

多模型训练：遍历infos中的模型配置，依次训练并评估模型效果，支持时间内外测试集验证。

模型保存与日志：训练结果保存为.pkl文件，日志记录详细过程（如内存溢出自动中断）。


### 4. 预测与结果整合

分批次预测：通过_pices将数据拆分为多份，解决内存不足问题，支持分布式计算。

分数合并：使用allmodel_scores_fun汇总所有模型分数，按权重计算用户综合得分。

结果存储：预测分数可保存至CSV或数据库（如ml.binaryclassify_score_m表）。

``` python
#预测配置示例
_pices = {4: [['0','1','2'], ['3','4','5'], ['6','7'], ['8','9']]}
for i, j in enumerate(_pices[4]):
    Info_i = choradd_namedtuple(Info, {'condition': f"right({Info.col_id},1) in {j}"})
    pred_res[model_name] = predict_fun(train_result, Info_i, n_reason)
```

### 5. 监控与维护

分数评估：通过privy_score_evaluate2统计模型KS、AUC等指标，监控模型衰减。

文件清理：

``` python
train_data_csv = privy_selectfile(allfile, filecontain='^train_data~', mode='old')
privy_delfile(train_data_csv)  
# 删除旧训练数据 测试集及临时文件，释放存储空间
```

## 三、技术优化

模块化设计：将数据加工、模型训练、预测等环节封装为独立函数（如tab_explore_create、train_test_fun），提升代码复用性。

内存管理：通过分批次预测、日志监控内存溢出，避免大规模数据处理时的崩溃风险。

自动化SQL生成：动态拼接SQL语句，减少人工编写错误，适配多数据源（Hive/PostgreSQL）。

## 四、使用注意

配置检查：运行前需确认plat和default_values与实际环境匹配，尤其是数据库连接参数。

模型迭代：定期更新infos中的模型配置（如训练账期、特征阈值），并通过privy_score_evaluate1监控模型表现。

日志分析：利用privy_log_save生成的日志文件（如.pkl）定位训练或预测异常。

## 五、总结

本项目通过标准化流程实现了机器学习模型的全生命周期管理，涵盖数据准备、模型训练、预测部署及监控维护。代码设计兼顾灵活性与效率，适用于多业务场景的快速迭代，为企业在用户画像、精准营销等领域的应用提供了可靠的技术支持。
