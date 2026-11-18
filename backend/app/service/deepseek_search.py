"""DeepSeek Search API adapter shared by chat, research and news collection."""
import os
import time
from typing import Dict, List

import requests

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"


def search_web(query: str, count: int = 10, api_key: str = None) -> List[Dict]:
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未配置")
    query = query.strip()
    if not query or len(query) > 128:
        raise ValueError("DeepSeek 搜索词长度必须为 1–128 个字符")
    if not 1 <= count <= 50:
        raise ValueError("DeepSeek 搜索结果数量必须为 1–50")
    for attempt in range(4):
        response = requests.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": query, "count": count, "need_content": True},
            timeout=30,
        )
        if response.status_code != 429 or attempt == 3:
            break
        time.sleep(1.5 * (attempt + 1))
    response.raise_for_status()
    data = response.json()
    items = data.get("results")
    if not isinstance(items, list):
        raise ValueError("DeepSeek 搜索响应缺少 results 列表")
    results = []
    for item in items:
        if not item.get("url") or not (item.get("snippet") or item.get("summary") or item.get("content")):
            continue
        title = item.get("title") or "N/A"
        site = item.get("site_name") or ""
        date = item.get("date") or ""
        results.append({
            "url": item["url"], "title": title, "name": title,
            "snippet": item.get("snippet") or "",
            "summary": item.get("summary") or item.get("snippet") or item.get("content") or "",
            "content": item.get("content") or "",
            "site_name": site, "siteName": site,
            "siteIcon": item.get("site_icon") or "",
            "date": date, "datePublished": date, "source": "web",
            "authority": item.get("authority"), "rerank_score": item.get("rerank_score"),
        })
    return results
