"""Offline contract tests: python app/scripts/test_deepseek_search.py."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import requests

spec = importlib.util.spec_from_file_location('deepseek_search', Path(__file__).parents[1] / 'service/deepseek_search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


class SearchContractTests(unittest.TestCase):
    @patch.object(search.requests, 'post')
    def test_request_and_source_mapping(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {'results': [{
            'title': '原文标题', 'url': 'https://example.com/report', 'snippet': '摘要',
            'summary': '详细摘要', 'content': '原文', 'site_name': None, 'site_icon': None,
            'date': '2026-11-17T00:00:00Z', 'authority': 3, 'rerank_score': 2.5,
        }]}
        result = search.search_web(' 航空 ', 5, 'test-key')[0]
        self.assertEqual(post.call_args.args, (search.SEARCH_URL,))
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer test-key')
        self.assertEqual(post.call_args.kwargs['json'], {'query': '航空', 'count': 5, 'need_content': True})
        self.assertEqual(result['title'], result['name'])
        self.assertEqual(result['date'], result['datePublished'])
        self.assertEqual(result['content'], '原文')
        self.assertEqual(result['siteName'], '')

    @patch.object(search.requests, 'post')
    def test_invalid_query_never_sent(self, post):
        for query in ['', '  ', '字' * 129]:
            with self.assertRaises(ValueError): search.search_web(query, 5, 'test-key')
        post.assert_not_called()

    @patch.object(search.time, 'sleep')
    @patch.object(search.requests, 'post')
    def test_rate_limit_retries_then_success(self, post, sleep):
        ok = Mock(status_code=200)
        ok.json.return_value = {'results': []}
        post.side_effect = [Mock(status_code=429), ok]
        self.assertEqual(search.search_web('航空', 5, 'test-key'), [])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    @patch.object(search.requests, 'post')
    def test_errors_are_not_empty_success(self, post):
        post.return_value = Mock(status_code=401)
        post.return_value.raise_for_status.side_effect = requests.HTTPError('401')
        with self.assertRaises(requests.HTTPError): search.search_web('航空', 5, 'test-key')
        post.return_value.raise_for_status.side_effect = None
        post.return_value.json.return_value = {'unexpected': []}
        with self.assertRaises(ValueError): search.search_web('航空', 5, 'test-key')

if __name__ == '__main__': unittest.main()
