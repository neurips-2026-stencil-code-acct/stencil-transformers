"""Trusted deterministic fixtures only: no model, target code, or network calls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("coding_agent", Path(__file__).with_name("coding_agent.py"))
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


def budgets():
    # Small test-fixture limits, not proposed benchmark budgets.
    return {
        "max_model_calls": 4, "max_total_tokens": 1000, "max_wall_seconds": 10.0,
        "max_command_seconds": 2.0, "max_command_output_bytes": 1000,
        "max_tool_output_bytes": 10000, "max_http_response_bytes": 10000,
        "max_repository_bytes": 10000, "max_repository_files": 20,
        "max_new_tokens_per_call": 20, "cleanup_grace_seconds": 1.0,
    }


class FakeModel:
    responses = []
    prompt_tokens = 10
    completed = 0

    def __init__(self, endpoint, limits, deadline):
        self.http_requests = 0
        self.index = 0

    def prepare(self, messages):
        return json.dumps(messages), self.prompt_tokens

    def complete(self, prompt, settings, output_tokens):
        type(self).completed += 1
        content = self.responses[self.index]
        self.index += 1
        return {"content": content, "tokens_evaluated": self.prompt_tokens,
                "tokens_predicted": 3, "truncated": False}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "main.py").write_bytes(b"value = 1\n")
        self.settings = {"inference": {"context_tokens": 50000, "cache_prompt": False,
                         "speculative_decoding": False, "remote_model_code": False,
                         "temperature": 0, "seed": 0, "top_p": 1, "top_k": 0,
                         "min_p": 0, "repeat_penalty": 1}}
        FakeModel.responses, FakeModel.completed, FakeModel.prompt_tokens = [], 0, 10

    def tearDown(self):
        self.temporary.cleanup()

    def controller(self, limits=None, experience=None, name="run"):
        return agent.CodingAgent(source=self.source, output=self.root / name, experience=experience,
                    task="Trusted fixture: edit a value without running it.",
                    condition="accumulated" if experience else "none",
                    commands={"fixed-tests": [sys.executable, "-m", "unittest"]},
                    budgets=limits or budgets(), lock=self.settings,
                    endpoint="http://127.0.0.1:18081", model_factory=FakeModel)

    def run_mocked(self, controller):
        with patch.object(agent, "isolation_preflight", return_value={"mode": "landlock", "fallback": "none"}):
            return controller.run()

    def test_all_budget_fields_are_required_and_finite(self):
        for field in agent.BUDGET_FIELDS:
            value = budgets()
            del value[field]
            with self.assertRaises(agent.ContractError):
                agent.validate_budgets(value)
        for invalid in (True, 0, -1, float("inf"), float("nan")):
            value = budgets()
            value["max_wall_seconds"] = invalid
            with self.assertRaises(agent.ContractError):
                agent.validate_budgets(value)

    def test_endpoint_only_literal_loopback_without_redirect_or_proxy(self):
        for endpoint in ("http://example.com:80", "http://localhost:80", "https://127.0.0.1:80",
                         "http://127.0.0.1:80/v1", "http://user@127.0.0.1:80", "http://127.0.0.1"):
            with self.assertRaises(agent.ContractError):
                agent.LocalModel(endpoint, budgets(), 1)
        agent.LocalModel("http://127.0.0.1:18081", budgets(), 1)

    def test_path_traversal_and_hardlinks_are_refused(self):
        for name in ("../outside", "/outside", "a/../outside", "C:/outside", "a\\outside", ".git/config"):
            with self.assertRaises(agent.ContractError):
                agent.safe_path(self.source, name)
        os.link(self.source / "main.py", self.source / "linked.py")
        with self.assertRaises(agent.ContractError):
            agent.snapshot(self.source, budgets())

    def test_duplicate_json_keys_and_extra_action_fields_are_rejected(self):
        for value in ('{"action":"finish","summary":"x","summary":"y"}',
                      '{"action":"run_test","command":"fixed-tests","argv":["anything"]}',
                      '{"action":"read_file","scope":"other","path":"main.py"}'):
            with self.assertRaises(ValueError):
                agent.parse_action(value)

    def test_windows_fails_before_any_model_or_target_execution(self):
        controller = self.controller()
        with patch.object(agent.sys, "platform", "win32"), patch.object(agent.subprocess, "Popen") as execute:
            result = controller.run()
        self.assertEqual(result["status"], "controller_or_infrastructure_failure")
        self.assertIn("Windows is unsupported", result["error"])
        self.assertEqual(FakeModel.completed, 0)
        execute.assert_not_called()

    def test_context_overflow_fails_before_generation(self):
        self.settings["inference"]["context_tokens"] = 29  # 10 prompt + 20 requested > 29.
        result = self.run_mocked(self.controller())
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIn("Context overflow", result["error"])
        self.assertEqual(FakeModel.completed, 0)

    def test_token_cap_is_checked_before_generation(self):
        limits = budgets()
        limits["max_total_tokens"] = 29
        result = self.run_mocked(self.controller(limits))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(FakeModel.completed, 0)

    def test_malformed_output_and_corrective_turn_are_both_counted(self):
        FakeModel.responses = ["not json", json.dumps({"action": "finish", "summary": "fixture only"})]
        result = self.run_mocked(self.controller())
        self.assertEqual(result["status"], "agent_finished_unverified")
        self.assertEqual(result["cost"]["model_calls"], 2)
        self.assertEqual(result["cost"]["malformed_responses"], 1)
        self.assertEqual(result["cost"]["prompt_tokens"], 20)
        self.assertEqual(result["cost"]["completion_tokens"], 6)

    def test_edit_is_fresh_source_is_unchanged_and_patch_is_saved(self):
        FakeModel.responses = [json.dumps({"action": "replace_text", "path": "main.py", "old": "1", "new": "2"}),
                               json.dumps({"action": "finish", "summary": "Not executed or verified"})]
        controller = self.controller()
        result = self.run_mocked(controller)
        self.assertEqual(result["status"], "agent_finished_unverified")
        self.assertEqual((self.source / "main.py").read_bytes(), b"value = 1\n")
        self.assertEqual((controller.repository / "main.py").read_bytes(), b"value = 2\n")
        self.assertIn("+value = 2", (controller.output / "patch.diff").read_text())
        self.assertEqual(result["changes"][0]["path"], "main.py")
        with self.assertRaises(agent.ContractError):
            self.controller(name="run")  # Cannot reuse a previous run directory.

    def test_unknown_command_never_launches_a_subprocess(self):
        FakeModel.responses = [json.dumps({"action": "run_test", "command": "invented-command"}),
                               json.dumps({"action": "finish", "summary": "No command executed"})]
        with patch.object(agent.subprocess, "Popen") as execute:
            result = self.run_mocked(self.controller())
        execute.assert_not_called()
        self.assertEqual(result["cost"]["action_errors"], 1)
        self.assertEqual(result["cost"]["command_calls"], 0)

    def test_frozen_experience_change_is_detected(self):
        experience = self.root / "experience"
        experience.mkdir()
        (experience / "lesson.txt").write_text("verified source lesson", encoding="utf-8")
        controller = self.controller(experience=experience)
        controller.package = agent.snapshot(experience, budgets())
        (experience / "lesson.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(agent.ContractError, "Frozen experience changed"):
            controller.check_package()

    def test_tool_output_cap_is_counted_and_stops_run(self):
        FakeModel.responses = [json.dumps({"action": "read_file", "scope": "repository", "path": "main.py"})]
        limits = budgets()
        limits["max_tool_output_bytes"] = 1
        result = self.run_mocked(self.controller(limits))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertGreater(result["cost"]["tool_output_bytes"], 1)

    def test_late_final_completion_is_budget_exhausted_with_usage_recorded(self):
        controller = self.controller()

        def late_complete(*args):
            controller.deadline = agent.time.monotonic() - 1
            return {"content": json.dumps({"action": "finish", "summary": "Late finish"}),
                    "tokens_evaluated": 10, "tokens_predicted": 3, "truncated": False}

        with patch.object(controller.model, "complete", side_effect=late_complete):
            result = self.run_mocked(controller)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["cost"]["model_calls"], 1)
        self.assertEqual(result["cost"]["completion_tokens"], 3)
        self.assertNotIn("agent_summary", result)

    def test_tool_overrun_cannot_be_reported_as_success(self):
        controller = self.controller()
        FakeModel.responses = [json.dumps({"action": "finish", "summary": "Late tool"})]

        def late_tool(action):
            controller.deadline = agent.time.monotonic() - 1
            return {"finished": True}

        with patch.object(controller, "execute", side_effect=late_tool):
            result = self.run_mocked(controller)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertNotIn("agent_summary", result)

    def test_http_body_uses_absolute_deadline_and_no_network_fixture(self):
        now = [0.0]
        model = agent.LocalModel("http://127.0.0.1:18081", budgets(), 1.0)

        class Response:
            status = 200

            def read1(self, count):
                now[0] = 1.1  # Even a successful late read must fail.
                return b"{}"

        class Connection:
            sock = unittest.mock.Mock()

            def connect(self):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Response()

            def close(self):
                pass

        with patch.object(agent.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(agent.http.client, "HTTPConnection", return_value=Connection()), \
             patch.object(agent.threading, "Timer") as timer:
            with self.assertRaisesRegex(agent.BudgetError, "wall-time"):
                model.post("/completion", {})
        timer.assert_called_once()
        self.assertEqual(timer.call_args.args[0], 1.0)
        timer.return_value.start.assert_called_once()
        timer.return_value.cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
