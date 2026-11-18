# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
知识库检索服务 - 基于 Milvus

功能：
1. retrieve_content - 从指定集合检索内容
2. retrieve_from_knowledge_base - 从知识库检索内容
"""

from contextvars import ContextVar
from typing import List, Dict, Any, Optional
from service.milvus_service import get_milvus_service, kb_collection_name

# 本次研究请求可检索的知识库集合，由 research_router 按登录用户设置。
# 默认空集：没设置就一个库都不查，宁可查不到也不越权。
# shortcut: 集合按库名建，两个用户建同名库会共用一个集合；多用户上线前集合名要带上用户或库 ID。
research_collections: ContextVar[frozenset] = ContextVar("research_collections", default=frozenset())
from service.embedding_service import generate_embedding


def retrieve_content(
    indexNames: str,
    question: str,
    top_k: int = 5,
    kb_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    检索相关内容

    Args:
        indexNames: 集合名称（知识库索引）
        question: 查询问题
        top_k: 返回结果数量
        kb_id: 知识库ID（可选过滤）

    Returns:
        检索结果列表
    """
    try:
        # 1. 生成查询向量
        query_vectors = generate_embedding([question])
        if not query_vectors or len(query_vectors) == 0:
            print("生成查询向量失败")
            return []

        query_vector = query_vectors[0]

        # 2. 执行向量搜索
        milvus = get_milvus_service()
        results = milvus.search(
            collection_name=indexNames,
            query_vector=query_vector,
            top_k=top_k,
            kb_id=kb_id,
        )

        # 3. 格式化结果
        extracted_data = []
        for i, result in enumerate(results, start=1):
            message = {
                "id": i,
                "chunk_index": result.get("chunk_index"),
                "document_id": result.get("doc_id", "N/A"),
                "document_name": result.get("filename", "N/A"),
                "content_with_weight": result.get("content", ""),
                "score": result.get("score", 0),
            }
            extracted_data.append(message)

        return extracted_data

    except Exception as e:
        print(f"检索错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def retrieve_from_knowledge_base(
    kb_name: str,
    question: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    从知识库检索内容

    Args:
        kb_name: 知识库名称
        question: 查询问题
        top_k: 返回结果数量

    Returns:
        检索结果列表
    """
    # 将知识库名称转换为集合名称
    collection_name = kb_collection_name(kb_name)
    return retrieve_content(collection_name, question, top_k)


def retrieve_research_chunks(question: str, kb_name: Optional[str], top_k: int) -> List[Dict[str, Any]]:
    """Use the selected research KB, or merge hits from the caller's own KB collections."""
    allowed = research_collections.get()
    names = allowed & {kb_collection_name(kb_name)} if kb_name else allowed
    hits = []
    for name in sorted(names):
        hits.extend(retrieve_content(name, question, top_k))
    return sorted(hits, key=lambda r: r.get("score", 0), reverse=True)[:top_k]
