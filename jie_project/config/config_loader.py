import yaml
from typing import Any


class ConfigMeta(type):
    """
    元类核心逻辑：
    1. 通过类属性直接访问配置 (Config.valve.port)
    2. 首次访问时合并加载所有配置文件
    3. 自动处理嵌套字典的链式访问
    """
    _merged_data = None  # 类级配置存储

    def __getattr__(cls, name: str) -> Any:
        # 拦截未定义属性的访问
        if cls._merged_data is None:
            # 惰性加载机制：第一次访问时初始化
            cls._load_configs()

        value = cls._merged_data.get(name)
        if value is None:
            raise AttributeError(f"'{cls.__name__}' 对象没有属性 '{name}'")

        # 嵌套字典转 ConfigDict 以便链式访问
        return ConfigDict(value) if isinstance(value, dict) else value

    def _load_configs(cls):
        """加载并合并所有配置文件"""
        if not cls.config_paths:
            raise ValueError("必须通过 ConfigLoader.config_paths 指定配置文件路径")

        def deep_merge(base: dict, update: dict):
            """增强型深度合并"""
            for key, value in update.items():
                # 跳过空字典的覆盖
                if isinstance(value, dict) and not value:
                    continue

                if isinstance(value, dict):
                    node = base.setdefault(key, {})
                    deep_merge(node, value)
                else:
                    base[key] = value
            return base

        merged = {}
        for path in cls.config_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    current_config = yaml.safe_load(f) or {}
                    deep_merge(merged, current_config)
            except FileNotFoundError:
                raise RuntimeError(f"配置文件不存在: {path}")
            except yaml.YAMLError as e:
                raise RuntimeError(f"YAML解析失败 [{path}]: {e}")

        if not merged:
            raise ValueError("所有配置文件内容均为空")
        cls._merged_data = merged


class ConfigDict:
    """字典包装器，支持点操作符访问嵌套字典"""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        value = self._data.get(name)
        if value is None:
            raise AttributeError(f"配置项 '{name}' 不存在")
        return ConfigDict(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        """递归转换为原生字典"""
        return {
            key: value.to_dict() if isinstance(value, ConfigDict) else value
            for key, value in self._data.items()
        }

    def get(self, key, default=None):
        value = self._data.get(key)
        if value is None:
            return default
        return value

    def __repr__(self):
        return f"ConfigDict({self._data})"


class ConfigLoader(metaclass=ConfigMeta):
    """
    用法示例：
    """
    config_paths = []  # 通过这里指定配置文件路径


if __name__ == '__main__':
    ConfigLoader.config_paths = [
        r"/home/ykj/build/jie_project/config/default_config.yaml",
    ]
    print(ConfigLoader.model_path)
