"""
案例6：缓存管理
"""

import time
from typing import Dict, Any


class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}

    def get(self, key: str) -> Any:
        """获取缓存"""
        if key in self.cache:
            item = self.cache[key]
            if item["expires_at"] > time.time():
                return item["data"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: Any, ttl: int = 1800):
        """设置缓存"""
        self.cache[key] = {
            "data": data,
            "expires_at": time.time() + ttl
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
