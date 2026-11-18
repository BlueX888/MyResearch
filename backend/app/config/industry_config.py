# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
行业配置 - 定义各行业的搜索关键词
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IndustryConfig:
    """行业配置"""
    id: str
    name: str
    description: str
    news_keywords: List[str]
    bidding_keywords: List[str]
    research_keywords: List[str]


# 预定义的行业配置
INDUSTRY_CONFIGS: Dict[str, IndustryConfig] = {
    "aviation": IndustryConfig(
        id="aviation",
        name="航空工业",
        description="航空装备、主机厂（成飞/沈飞）、航发与产业链",
        news_keywords=[
            "航空工业 十五五",
            "成飞 沈飞",
            "六代机",
            "舰载机",
            "航空发动机 国产化",
            "军贸 出口",
            "航空复合材料",
            "低空经济",
        ],
        bidding_keywords=[
            "航空装备",
            "飞机制造",
            "航空发动机",
            "机载系统",
            "航空复合材料",
            "航空维修",
        ],
        research_keywords=["成飞", "沈飞", "六代机", "航空发动机", "舰载机", "军贸出口"],
    ),
}

# 默认行业
DEFAULT_INDUSTRY_ID = "aviation"


def get_industry_config(industry_id: Optional[str] = None) -> IndustryConfig:
    """
    获取行业配置

    Args:
        industry_id: 行业ID，如果为空则返回默认行业

    Returns:
        行业配置
    """
    if not industry_id:
        industry_id = DEFAULT_INDUSTRY_ID

    config = INDUSTRY_CONFIGS.get(industry_id)
    if not config:
        logger.warning(f"[industry_config] 未找到行业配置: {industry_id}, 使用默认行业")
        config = INDUSTRY_CONFIGS[DEFAULT_INDUSTRY_ID]

    logger.info(f"[industry_config] 获取行业配置: {config.name} ({config.id})")
    return config


def get_all_industries() -> List[Dict]:
    """
    获取所有行业列表

    Returns:
        行业列表
    """
    return [
        {
            "id": config.id,
            "name": config.name,
            "description": config.description,
        }
        for config in INDUSTRY_CONFIGS.values()
    ]
