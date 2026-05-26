---
description: >-
  AutoMining 电信运营商客户经营大数据分类挖掘项目。帮助 AI 编码代理快速理解项目结构、
  约定和常见注意事项。
globs: "**/*.py"
---

# AutoMining — AI 代理指南

## 项目概述

电信运营商客户经营需求驱动的大数据分类挖掘程序。实现完整的机器学习建模与预测流程：数据加工、模型训练、测试评估、预测打分及结果监控。支持多业务场景（流失预警、套餐迁移、流量包推荐等）和多数据平台（Hive/PostgreSQL）运行。

## 快速命令

```bash
# 安装依赖
pip install -e .                # 安装核心依赖
pip install -e ".[hive]"       # 含 Hive/Spark 支持

# 主入口 — 查看配置
python main.py --show-config

# 主入口 — 运行完整流程
python main.py

# 主入口 — 仅训练测试
python main.py --mode train

# 主入口 — 仅预测打分
python main.py --mode predict

# 主入口 — 仅生成 SQL（不连数据库时）
python main.py --mode sql

# 指定环境变量文件
python main.py --env .env.production

# 原有方式（仍可用）
python Mining/build/all_code.py
python Mining/build/privy_build.py
```

> **注意**: 运行前需确认 `.env` 文件中的配置与实际环境匹配，尤其是数据库连接参数。详见 [README.md](./README.md)。

## 项目架构

```
├── main.py                 # 主入口（CLI 驱动，支持多种运行模式）
├── .env                    # 环境配置文件（数据库、路径、运行参数）
├── pyproject.toml          # 项目元数据与依赖
├── Mining/
│   ├── config.py           # 配置加载模块（从 .env 读取，提供统一接口）
│   ├── build/              # 工作流驱动脚本（原有入口）
│   │   ├── all_code.py     # 汇总代码并上传到平台
│   │   └── privy_build.py  # 预测/打分主函数
│   ├── selfmodule/
│   │   ├── binarymodule/   # 二分类模型模块
│   │   │   ├── modelinfo.py    # 模型信息、参数配置、字段校验
│   │   │   ├── pipemodel.py    # 流水线构建（create_pipemodel）
│   │   │   ├── traintest.py    # 训练与测试（model_cycle, CreateModelSet）
│   │   │   ├── predictscore.py # 预测打分
│   │   │   ├── privy_report.py # 报表生成
│   │   │   └── privy_stat.py   # 统计工具
│   │   ├── tablemodule/    # 表结构与 SQL 工具
│   │   │   ├── basestr.py      # 数据字典/字段定义（需手动维护）
│   │   │   ├── tablefun.py     # 表操作函数
│   │   │   └── tablesql.py     # SQL 动态生成
│   │   └── toolmodule/     # 通用工具
│   │       ├── dataprep.py     # DataFrame 转换器、流水线工具（Pipeline_DF, WoeTransformer_DF 等）
│   │       ├── datatrans.py    # 数据库交互（gp/hive）、全局配置
│   │       ├── predict_job.py  # 并行预测作业入口
│   │       ├── strtotable.py   # 字符串转表工具
│   │       ├── ignorewarn.py   # 警告抑制
│   │       └── privy_*.py      # 平台私有工具
```

## 关键约定

### 编码风格
- **命名**: 函数/变量使用 `snake_case`；自定义 Transformer 类使用 `CamelCase_DF` 后缀。
- **导入**: 支持绝对导入（`from Mining.selfmodule...`）和 try/except 兼容导入。
- **注释**: 使用中文注释和中文 docstring。
- **错误处理**: 使用 `raise Exception(...)` + `warnings.warn(...)` 严格校验风格。

### 自定义 DataFrame Transformer
项目中大量使用以 `_DF` 后缀结尾的自定义 scikit-learn transformer（如 `Pipeline_DF`、`FeatureUnion_DF`、`WoeTransformer_DF`、`PsiTransformer_DF`），它们将 sklearn API 与 pandas DataFrame 结合。见 `Mining/selfmodule/toolmodule/dataprep.py`。

### 命名元组配置
模型参数使用 `to_namedtuple` / `choradd_namedtuple` 将字典转换为命名元组（`Info` 对象），支持属性访问。见 `dataprep.py`。

### 模型持久化
使用 `joblib` 保存/加载模型集合（`.pkl` 文件）。

## 关键配置

所有可配置项集中在项目根目录的 `.env` 文件中，由 `Mining/config.py` 统一加载：


配置访问方式：
```python
from Mining.config import config
print(config.DB_TYPE)        # 'gp'
print(config.get_paradict())  # {'dbname': ..., 'user': ..., ...}
```

`main.py` 启动时会自动调用 `_patch_module_level_config()` 将 `.env` 中的值注入到 `datatrans.py` 和 `privy_outredirect.py` 的模块级变量中，**不修改原有模块代码**。

原有构建脚本（`all_code.py` / `privy_build.py`）中仍保留硬编码配置，可独立运行（兼容原有方式）。

全局配置模块级变量（`datatrans.py`）：
- `db` — 数据库类型（`'gp'` 或 `'hive'`）
- `paradict` — 数据库连接参数
- `prefix` — 表名前缀

## 常见注意事项

1. **环境配置**: 运行前必须配置 `.env` 文件（数据库连接、工作目录等），否则会因数据库连接失败报错。可用 `python main.py --show-config` 验证配置。
2. **虚拟环境**: 项目已配置 `.venv` 虚拟环境，激活后使用 `pip install -e .` 安装依赖（含 pandas, numpy, sklearn, lightgbm, xgboost 等）。
3. **数据库依赖**: 许多功能依赖外部 PostgreSQL 或 Hive 数据库，本地运行需配置好连接参数。
4. **路径硬编码**: 脚本中包含 Windows 风格路径（`D:\...`），跨平台运行需调整 `.env` 中的 `MODELWD_PLATFORM` 和 `SELFWD_PLATFORM`。
5. **数据字典依赖**: `basestr.py` 中的表结构定义必须与实际数据库一致。
6. **严格列名约定**: 代码假定特定列名（`col_target`、`col_month`、`col_id` 等），数据列名不匹配会报错。多处执行 `to_lower` 大小写转换。
7. **平台上传机制**: `all_code.py` 包含代码上传逻辑，本地运行时可能触发不必要的上传操作。

## 参考文档

- [项目说明文档](./README.md) — 完整的功能解析、技术优化和使用注意
