"""Public search service backed by the official DeepSeek Search API."""
from typing import Any, Dict, List
from dotenv import load_dotenv
from .deepseek_search import search_web

load_dotenv()


class WebSearchService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def search(self, query: str, gl: str = "cn", hl: str = "zh-cn",
               autocorrect: bool = True, page: int = 1,
               search_type: str = "search") -> Dict[str, Any]:
        # Preserve the public request schema; DeepSeek has no pagination or verticals.
        try:
            if page != 1 or search_type != "search":
                raise ValueError("DeepSeek 搜索仅支持 page=1、search_type=search")
            results = search_web(query, 10, self.api_key)
            return {"organic": [
                {"title": r["title"], "link": r["url"], "snippet": r["snippet"] or r["summary"],
                 "position": i, "date": r["date"]}
                for i, r in enumerate(results, 1)
            ]}
        except Exception as e:
            return {"error": True, "message": f"DeepSeek search failed: {e}"}

    def extract_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"type": "organic", **item} for item in search_results.get("organic", [])]
