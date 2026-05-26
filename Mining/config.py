"""
AutoMining 配置管理模块
从 .env 文件加载所有配置，提供统一的配置访问接口。

用法:
    from Mining.config import config
    print(config.DB_TYPE)
    print(config.get_paradict())
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any


def _find_env_file() -> Optional[Path]:
    """向上查找 .env 文件"""
    current = Path(__file__).resolve().parent.parent  # Mining 目录的父目录
    for _ in range(5):
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        if current.parent == current:
            break
        current = current.parent
    return None


def _load_dotenv(env_path: Optional[Path] = None) -> Dict[str, str]:
    """手动解析 .env 文件（无需 python-dotenv 依赖）"""
    if env_path is None:
        env_path = _find_env_file()

    result: Dict[str, str] = {}

    if env_path is None or not env_path.exists():
        return result

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            # 解析 KEY=VALUE
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 移除引号
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                result[key] = value

    return result


def _to_bool(value: str) -> bool:
    """字符串转布尔值"""
    return value.strip().lower() in ("true", "yes", "1", "on")


def _to_optional(value: str) -> Optional[str]:
    """字符串转可选值（None / 字符串）"""
    if value is None or value.strip().lower() in ("none", "null", ""):
        return None
    return value


class Config:
    """全局配置单例"""

    _instance: Optional["Config"] = None
    _env: Dict[str, str] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._env = _load_dotenv()
        return cls._instance

    # ---- 平台与系统 ----
    @property
    def PLAT(self) -> str:
        return self._env.get("PLAT", "本机测试")

    @property
    def SYS_TEM(self) -> str:
        return self._env.get("SYS_TEM", "win")

    # ---- 数据库配置 ----
    @property
    def DB_TYPE(self) -> str:
        return self._env.get("DB_TYPE", "gp")

    @property
    def DB_NAME(self) -> Optional[str]:
        return _to_optional(self._env.get("DB_NAME"))

    @property
    def DB_USER(self) -> Optional[str]:
        return _to_optional(self._env.get("DB_USER"))

    @property
    def DB_PWD(self) -> Optional[str]:
        return _to_optional(self._env.get("DB_PWD"))

    @property
    def DB_PORT(self) -> Optional[str]:
        return _to_optional(self._env.get("DB_PORT"))

    @property
    def DB_HOST(self) -> Optional[str]:
        return _to_optional(self._env.get("DB_HOST"))

    @property
    def DB_PREFIX(self) -> str:
        return self._env.get("DB_PREFIX", "ml.")

    @property
    def DB_SECONDS(self) -> int:
        return int(self._env.get("DB_SECONDS", "5"))

    # ---- 工作目录 ----
    @property
    def MODELWD_PLATFORM(self) -> str:
        return self._env.get("MODELWD_PLATFORM", "")

    @property
    def SELFWD_PLATFORM(self) -> str:
        return self._env.get("SELFWD_PLATFORM", "")

    # ---- SQL 执行模式 ----
    @property
    def SQL_TYPE(self) -> str:
        return self._env.get("SQL_TYPE", "execute")

    @property
    def SQL_COMMENT(self) -> str:
        return self._env.get("SQL_COMMENT", "----")

    @property
    def SRC(self) -> str:
        return self._env.get("SRC", "gp")

    # ---- 模型训练参数 ----
    @property
    def AUTO_PAIR2(self) -> bool:
        return _to_bool(self._env.get("AUTO_PAIR2", "false"))

    @property
    def DIFF_LIMIT(self) -> Optional[str]:
        return _to_optional(self._env.get("DIFF_LIMIT"))

    @property
    def TABLE_PSI(self) -> bool:
        return _to_bool(self._env.get("TABLE_PSI", "true"))

    @property
    def TABLE_R(self) -> bool:
        return _to_bool(self._env.get("TABLE_R", "true"))

    # ---- 预测阶段参数 ----
    @property
    def N_REASON(self) -> Optional[int]:
        val = self._env.get("N_REASON", "3")
        return None if _to_optional(val) is None else int(val)

    @property
    def M_P(self) -> str:
        return self._env.get("M_P", "202012")

    # ---- 日志配置 ----
    @property
    def LOG_TYPE(self) -> str:
        return self._env.get("LOG_TYPE", "txt")

    # ---- 默认值替换字段 ----
    @property
    def DEFAULT_DBNAME(self) -> Optional[str]:
        return _to_optional(self._env.get("DEFAULT_DBNAME"))

    @property
    def DEFAULT_USER(self) -> Optional[str]:
        return _to_optional(self._env.get("DEFAULT_USER"))

    @property
    def DEFAULT_PWD(self) -> Optional[str]:
        return _to_optional(self._env.get("DEFAULT_PWD"))

    @property
    def DEFAULT_PORT(self) -> Optional[str]:
        return _to_optional(self._env.get("DEFAULT_PORT"))

    @property
    def DEFAULT_HOST(self) -> Optional[str]:
        return _to_optional(self._env.get("DEFAULT_HOST"))

    # ---- 便捷方法 ----

    def get_paradict(self) -> Dict[str, Optional[str]]:
        """获取数据库连接参数字典 (兼容原 datatrans.py 中的 paradict)"""
        return {
            "dbname": self.DB_NAME,
            "user": self.DB_USER,
            "pwd": self.DB_PWD,
            "port": self.DB_PORT,
            "host": self.DB_HOST,
        }

    def get_default_values(self) -> Dict[str, Any]:
        """获取 privy_replace_default / privy_upload_code 所需的 default_values 字典"""
        return {
            "sys_tem": self.SYS_TEM,
            "db": self.DB_TYPE,
            "prefix": self.DB_PREFIX,
            "dbname": self.DEFAULT_DBNAME,
            "user": self.DEFAULT_USER,
            "pwd": self.DEFAULT_PWD,
            "port": self.DEFAULT_PORT,
            "host": self.DEFAULT_HOST,
        }


# 全局配置单例
config = Config()
