"""Regression tests for scripts/upstream_doc_monitor.py fixes."""


from __future__ import annotations


import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "upstream_doc_monitor.py"
_spec = importlib.util.spec_from_file_location("upstream_doc_monitor", _SCRIPT)
udm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(udm)




class HttpProbeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/lower-403":
            b = b"ok"
            self.send_response(403)
            self.send_header("x-ratelimit-remaining", "0")
        elif self.path == "/etag":
            b = b"<html>v2</html>"
            self.send_response(200)
            self.send_header("etag", 'W/"v2"')
            self.send_header("last-modified", "Mon, 01 Jan 2026 00:00:00 GMT")
        elif self.path == "/not-modified":
            b = b""
            self.send_response(304)
            self.send_header("etag", 'W/"v2"')
        else:
            b = b"<html>v1</html>"
            self.send_response(200)
            self.send_header("ETag", 'W/"v1"')
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)




class HttpProbeServer:
    def __init__(self):
        self._srv = HTTPServer(("127.0.0.1", 0), HttpProbeHandler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._srv.server_address[1]}"
    def close(self):
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)




class TestUrlAndKey(unittest.TestCase):
    def test_blob_fragment_and_query_stripped(self):
        for url in (
            "https://github.com/hf/tf/blob/main/README.md#L1-L5",
            "https://github.com/hf/tf/blob/main/README.md?plain=1",
            "https://github.com/hf/tf/blob/main/README.md",
        ):
            e = udm._parse_source_url("t", "proj", url)
            self.assertEqual(e["type"], "repo_file")
            self.assertEqual(e["branch"], "main")
            self.assertEqual(e["path"], "README.md")
            self.assertEqual(e["key"], "proj::hf/tf/main/README.md")


    def test_web_page_keeps_query_strips_fragment(self):
        e = udm._parse_source_url("t", "proj", "https://example.com/doc?version=2#section")
        self.assertEqual(e["type"], "web_page")
        self.assertEqual(e["url"], "https://example.com/doc?version=2")


    def test_github_home_and_tree_rejected(self):
        for url in ("https://github.com/hf/tf", "https://github.com/hf/tf/tree/main/docs"):
            with self.assertRaises(udm.FatalError):
                udm._parse_source_url("t", "proj", url)


    def test_key_unique_across_repos_and_branches(self):
        a = udm._parse_source_url("t", "ci", "https://github.com/orgA/repoX/blob/main/doc.md")
        b = udm._parse_source_url("t", "ci", "https://github.com/orgB/repoY/blob/main/doc.md")
        c = udm._parse_source_url("t", "ci", "https://github.com/orgA/repoX/blob/dev/doc.md")
        self.assertNotEqual(a["key"], b["key"])
        self.assertNotEqual(a["key"], c["key"])


    def test_title_and_doc_links_use_branch(self):
        a = udm._parse_source_url("t", "ci", "https://github.com/orgA/repoX/blob/main/doc.md")
        self.assertEqual(udm._title_target(a), "orgA/repoX doc.md (main)")
        doc_url, history = udm._doc_links(a)
        self.assertEqual(doc_url, "https://github.com/orgA/repoX/blob/main/doc.md")
        self.assertEqual(history, "https://github.com/orgA/repoX/commits/main/doc.md")




class TestHttpHeaders(unittest.TestCase):
    def setUp(self):
        self._srv = HttpProbeServer()
    def tearDown(self):
        self._srv.close()


    def test_lowercase_rate_limit_header_detected(self):
        with self.assertRaises(udm.RateLimitError):
            udm.http_with_retry(self._srv.base_url + "/lower-403")


    def test_web_page_etag_lowercase_stored(self):
        res = udm.fetch_web_page({"url": self._srv.base_url + "/etag"}, {})
        self.assertEqual(res["outcome"], "ok")
        self.assertEqual(res["etag"], 'W/"v2"')
        self.assertEqual(res["last_modified"], "Mon, 01 Jan 2026 00:00:00 GMT")


    def test_web_page_304_fast_path(self):
        res = udm.fetch_web_page({"url": self._srv.base_url + "/not-modified"}, {"etag": 'W/"v2"', "sha": "abc123"})
        self.assertEqual(res["outcome"], "unchanged")
        self.assertEqual(res["sha"], "abc123")




class TestSyncFailureStateMachine(unittest.TestCase):
    def _run(self, state, create_result):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = tmp / "cfg.yaml"
            cfg.write_text("owners:\n  - lltiaor\nmaintainer: lltiaor\nprojects:\n  - project: demo\n    owner: lltiaor\n    url: https://github.com/orgA/repoX/blob/main/README.md\n", encoding="utf-8")
            st = tmp / "state.json"
            st.write_text(json.dumps({"schema_version": 1, "entries": state}), encoding="utf-8")
            out = tmp / "out"
            olds = [udm.fetch_repo_file_sha, udm.fetch_upstream_version, udm.fetch_open_ticket_map, udm.IssueSync.create, udm.IssueSync.get_state, os.environ.copy()]
            try:
                udm.fetch_repo_file_sha = lambda e, t: "NEW_SHA"
                udm.fetch_upstream_version = lambda r, t: "v9.9.9"
                # 模拟真实 fetch_open_ticket_map 的 baseline 回退行为：
                # 本轮新建工单写入 baseline.issue_url 后，result 应判 pending。
                udm.fetch_open_ticket_map = lambda r, t, e, b: {
                    entry["key"]: "https://github.com/cosdt-ci-test/workflows/issues/1"
                    for entry in e if b.get(entry["key"], {}).get("issue_url")
                }
                udm.IssueSync.create = lambda s, t, b: create_result
                udm.IssueSync.get_state = lambda s, u: None
                os.environ.update({"GH_TOKEN":"fake","GITHUB_REPOSITORY":"cosdt-ci-test/workflows","GITHUB_RUN_ID":"999","GITHUB_EVENT_NAME":"schedule"})
                rc = udm.run(["--config", str(cfg), "--state", str(st), "--output-dir", str(out), "--repo", "cosdt-ci-test/workflows"])
                ns = json.loads(st.read_text(encoding="utf-8"))["entries"]
                res = json.loads((out / "result.json").read_text(encoding="utf-8"))
                return rc, ns, res
            finally:
                udm.fetch_repo_file_sha = olds[0]; udm.fetch_upstream_version = olds[1]; udm.fetch_open_ticket_map = olds[2]; udm.IssueSync.create = olds[3]; udm.IssueSync.get_state = olds[4]
                os.environ.clear(); os.environ.update(olds[5])


    def test_create_failure_keeps_old_sha_and_redetects(self):
        k = "demo::orgA/repoX/main/README.md"
        s = {k: {"sha": "OLD_SHA", "date": "2026-01-01T00:00:00Z", "last_event": "ok"}}
        rc, ns, res = self._run(s, None)
        self.assertEqual(rc, 0)
        self.assertEqual(ns[k]["sha"], "OLD_SHA")
        self.assertNotEqual(res[0]["status"], "unchanged")
        rc2, ns2, _ = self._run(ns, None)
        self.assertEqual(rc2, 0)
        self.assertEqual(ns2[k]["sha"], "OLD_SHA")


    def test_create_success_advances_baseline(self):
        k = "demo::orgA/repoX/main/README.md"
        s = {k: {"sha": "OLD_SHA", "date": "2026-01-01T00:00:00Z", "last_event": "ok"}}
        rc, ns, res = self._run(s, "https://api.github.com/repos/foo/bar/issues/1")
        self.assertEqual(rc, 0)
        self.assertEqual(ns[k]["sha"], "NEW_SHA")
        self.assertEqual(res[0]["status"], "pending")




if __name__ == "__main__":
    unittest.main()
