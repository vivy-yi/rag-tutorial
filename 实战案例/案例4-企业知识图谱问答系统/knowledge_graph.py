"""
案例4：知识图谱构建
"""

from typing import Dict, List


def build_sample_graph() -> Dict:
    """构建示例知识图谱"""

    graph = {
        "entities": [
            {"id": "e1", "name": "张三", "type": "Person", "properties": {"部门": "技术部", "职位": "工程师"}},
            {"id": "e2", "name": "李四", "type": "Person", "properties": {"部门": "产品部", "职位": "经理"}},
            {"id": "e3", "name": "项目A", "type": "Project", "properties": {"状态": "进行中", "预算": "100万"}},
            {"id": "e4", "name": "项目B", "type": "Project", "properties": {"状态": "已完成", "预算": "50万"}},
            {"id": "e5", "name": "技术部", "type": "Department", "properties": {"负责人": "李四"}},
        ],
        "relationships": [
            {"source": "e1", "target": "e3", "type": "负责"},
            {"source": "e2", "target": "e4", "type": "负责"},
            {"source": "e1", "target": "e5", "type": "属于"},
        ]
    }

    return graph
