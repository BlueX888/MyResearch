"""研究接口鉴权与知识库隔离的离线自检（不连数据库、Milvus、模型）。

    cd backend/app && ../.venv/bin/python scripts/test_research_auth.py
"""
import asyncio
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.database import get_db
import router  # noqa: F401  包里把 research_router 重新导出成了路由对象，这里取模块本身
rr = sys.modules["router.research_router"]
from router.auth_router import get_current_user_required
from service import retrieval_service as rs
from service.milvus_service import kb_collection_name


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class FakeDb:
    """按模型返回预设行；归属过滤由被测代码表达，这里只控制"查得到 / 查不到"。"""

    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        return FakeQuery(self.rows.get(model, []))


def make_client(user=None, rows=None):
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[get_db] = lambda: FakeDb(rows or {})
    if user:
        app.dependency_overrides[get_current_user_required] = lambda: user
    return TestClient(app)


USER = mock.Mock(id=uuid.uuid4())
SESSION = str(uuid.uuid4())


class AuthTest(unittest.TestCase):
    def test_every_route_rejects_anonymous(self):
        client = make_client()
        for route in rr.router.routes:
            path = route.path.replace("{session_id}", SESSION)
            for method in route.methods:
                resp = client.request(method, path, json={"query": "x"}, params={"query": "x"})
                self.assertEqual(resp.status_code, 401, f"{method} {path}")

    def test_foreign_session_is_not_found(self):
        client = make_client(USER, {rr.ChatSession: []})  # 会话不属于该用户 → 查不到
        for method, path in [("POST", f"/research/cancel/{SESSION}"), ("GET", f"/research/checkpoint/{SESSION}"),
                             ("GET", f"/research/checkpoint/{SESSION}/full"), ("DELETE", f"/research/checkpoint/{SESSION}"),
                             ("POST", f"/research/resume/{SESSION}")]:
            self.assertEqual(client.request(method, path).status_code, 404, f"{method} {path}")
        resp = client.post("/research/stream", json={"query": "x", "session_id": SESSION})
        self.assertEqual(resp.status_code, 404)

    def test_stream_scopes_retrieval_to_own_kbs(self):
        seen = {}

        class FakeV2:
            async def research(self, **kwargs):
                # 与真实链路一致：检索在 to_thread 里执行
                seen["scope"] = await asyncio.to_thread(rs.research_collections.get)
                seen["user_id"] = kwargs["user_id"]
                yield "data: {}\n\n"

        kb = mock.Mock()
        kb.name = "成飞知识库"
        client = make_client(USER, {rr.KnowledgeBase: [kb]})
        with mock.patch.object(rr, "get_research_service_v2", FakeV2), \
                mock.patch.object(rr, "get_research_service", lambda: {"research_service": None}):
            client.app.dependency_overrides[rr.get_research_service] = lambda: {"research_service": None}
            resp = client.post("/research/stream", json={"query": "x", "version": "v2"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["scope"], frozenset({kb_collection_name("成飞知识库")}))
        self.assertEqual(seen["user_id"], str(USER.id))


class ScopeTest(unittest.TestCase):
    def search(self, scope, kb_name=None):
        calls = []

        def fake(name, question, top_k):
            calls.append(name)
            return [{"score": 1, "collection": name}]

        def run():
            if scope is not None:
                rs.research_collections.set(frozenset(scope))
            return rs.retrieve_research_chunks("q", kb_name, 5)

        import contextvars
        with mock.patch.object(rs, "retrieve_content", fake):
            hits = contextvars.copy_context().run(run)
        return calls, hits

    def test_no_scope_searches_nothing(self):
        self.assertEqual(self.search(None), ([], []))

    def test_only_own_collections_are_searched(self):
        calls, hits = self.search({"kb_a", "kb_b"})
        self.assertEqual(calls, ["kb_a", "kb_b"])
        self.assertEqual(len(hits), 2)

    def test_named_kb_outside_scope_is_refused(self):
        mine, other = kb_collection_name("mine"), kb_collection_name("other")
        self.assertEqual(self.search({mine}, "other"), ([], []))
        self.assertEqual(self.search({mine, other}, "other")[0], [other])


if __name__ == "__main__":
    unittest.main()
