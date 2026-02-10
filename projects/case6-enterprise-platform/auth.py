"""
案例6：认证模块
"""

from fastapi import Header, HTTPException
from typing import Optional


class User:
    """用户模型"""

    def __init__(self, id: str, username: str, email: str):
        self.id = id
        self.username = username
        self.email = email


async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> User:
    """获取当前用户（简化实现）"""

    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证信息")

    # 简化：实际应该验证JWT token
    if authorization.startswith("Bearer "):
        token = authorization[7:]

        # 模拟用户
        return User(
            id="user_123",
            username="demo_user",
            email="demo@example.com"
        )

    raise HTTPException(status_code=401, detail="无效的认证信息")
