"""Tests for RunSummaryExporter."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from framework.builder.exporter import RunSummaryExporter
from framework.schemas.decision import (
    Decision,
    DecisionEvaluation,
    DecisionType,
    Option,
    Outcome,
)
from framework.schemas.run import Problem, Run, RunMetrics, RunStatus
from framework.storage.backend import FileStorage


@pytest.fixture
def storage(tmp_path):
    return FileStorage(tmp_path / "test_storage")


@pytest.fixture
def sample_run():
    now = datetime.now()
    option = Option(
        id="opt_1",
        description="Call search API",
        action_type="tool_call",
        pros=["fast"],
        cons=["may miss results"],
        confidence=0.8,
    )
    outcome = Outcome(
        success=True,
        result="found 3 results",
        summary="Found 3 contacts matching query",
        tokens_used=150,
        latency_ms=320,
    )
    decision = Decision(
        id="dec_1",
        node_id="search-node",
        intent="Find matching contacts",
        decision_type=DecisionType.TOOL_SELECTION,
        options=[option],
        chosen_option_id="opt_1",
        reasoning="Search API is fastest option",
        outcome=outcome,
        evaluation=DecisionEvaluation(
            goal_aligned=True,
            outcome_quality=0.9,
        ),
    )
    failed_option = Option(
        id="opt_2",
        description="Send email",
        action_type="tool_call",
    )
    failed_decision = Decision(
        id="dec_2",
        node_id="email-node",
        intent="Send notification email",
        decision_type=DecisionType.TOOL_SELECTION,
        options=[failed_option],
        chosen_option_id="opt_2",
        reasoning="User requested notification",
        outcome=Outcome(
            success=False,
            error="SMTP connection refused",
            summary="Failed to send email",
            tokens_used=50,
            latency_ms=1200,
        ),
    )

    run = Run(
        id="run_test_001",
        goal_id="goal_support",
        goal_description="Process customer support tickets",
        started_at=now,
        status=RunStatus.COMPLETED,
        completed_at=now + timedelta(seconds=5),
        decisions=[decision, failed_decision],
        problems=[
            Problem(
                id="prob_0",
                severity="critical",
                description="Email delivery failed",
                root_cause="SMTP server down",
                suggested_fix="Add retry logic or fallback provider",
                decision_id="dec_2",
            ),
            Problem(
                id="prob_1",
                severity="warning",
                description="Search latency above threshold",
            ),
        ],
        metrics=RunMetrics(
            total_decisions=2,
            successful_decisions=1,
            failed_decisions=1,
            total_tokens=200,
            total_latency_ms=1520,
            nodes_executed=["search-node", "email-node"],
            edges_traversed=["e1"],
        ),
        narrative="Run completed. 1 of 2 decisions succeeded. Email delivery failed.",
        input_data={"ticket_id": "T-123"},
        output_data={"status": "partial", "contacts_found": 3},
    )
    return run


@pytest.fixture
def exporter_with_run(storage, sample_run):
    storage.save_run(sample_run)
    return RunSummaryExporter(storage), sample_run


class TestExportJSON:
    def test_returns_all_top_level_keys(self, exporter_with_run):
        exporter, run = exporter_with_run
        data = exporter.export_json(run.id)
        assert set(data.keys()) == {
            "metadata", "metrics", "decisions", "problems",
            "narrative", "input", "output",
        }

    def test_metadata_fields(self, exporter_with_run):
        exporter, run = exporter_with_run
        meta = exporter.export_json(run.id)["metadata"]
        assert meta["run_id"] == "run_test_001"
        assert meta["goal_id"] == "goal_support"
        assert meta["status"] == "completed"
        assert meta["duration_ms"] > 0

    def test_metrics_fields(self, exporter_with_run):
        exporter, run = exporter_with_run
        metrics = exporter.export_json(run.id)["metrics"]
        assert metrics["total_decisions"] == 2
        assert metrics["successful_decisions"] == 1
        assert metrics["failed_decisions"] == 1
        assert metrics["success_rate"] == 0.5
        assert metrics["total_tokens"] == 200

    def test_decisions_list(self, exporter_with_run):
        exporter, run = exporter_with_run
        decisions = exporter.export_json(run.id)["decisions"]
        assert len(decisions) == 2
        assert decisions[0]["intent"] == "Find matching contacts"
        assert decisions[0]["successful"] is True
        assert decisions[1]["successful"] is False
        assert decisions[1]["outcome"]["error"] == "SMTP connection refused"

    def test_problems_list(self, exporter_with_run):
        exporter, run = exporter_with_run
        problems = exporter.export_json(run.id)["problems"]
        assert len(problems) == 2
        assert problems[0]["severity"] == "critical"
        assert problems[0]["suggested_fix"] is not None

    def test_raises_on_missing_run(self, storage):
        exporter = RunSummaryExporter(storage)
        with pytest.raises(ValueError, match="Run not found"):
            exporter.export_json("nonexistent")


class TestExportMarkdown:
    def test_contains_header(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "# Run Report: run_test_001" in md
        assert "COMPLETED" in md

    def test_contains_metrics_table(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "## Metrics" in md
        assert "| Decisions | 2 |" in md
        assert "| Success Rate | 50% |" in md

    def test_contains_decisions(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "## Decisions" in md
        assert "Find matching contacts" in md
        assert "Send notification email" in md

    def test_contains_problems(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "## Problems" in md
        assert "### Critical" in md
        assert "Email delivery failed" in md
        assert "Root cause: SMTP server down" in md

    def test_contains_output(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "## Output" in md
        assert '"contacts_found": 3' in md

    def test_contains_narrative(self, exporter_with_run):
        exporter, run = exporter_with_run
        md = exporter.export_markdown(run.id)
        assert "## Summary" in md
        assert "Email delivery failed" in md

    def test_raises_on_missing_run(self, storage):
        exporter = RunSummaryExporter(storage)
        with pytest.raises(ValueError, match="Run not found"):
            exporter.export_markdown("nonexistent")


class TestExportToFile:
    def test_write_markdown_file(self, exporter_with_run, tmp_path):
        exporter, run = exporter_with_run
        out = tmp_path / "report.md"
        exporter.export_to_file(run.id, str(out), fmt="markdown")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# Run Report" in content

    def test_write_json_file(self, exporter_with_run, tmp_path):
        exporter, run = exporter_with_run
        out = tmp_path / "report.json"
        exporter.export_to_file(run.id, str(out), fmt="json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["metadata"]["run_id"] == "run_test_001"

    def test_creates_parent_dirs(self, exporter_with_run, tmp_path):
        exporter, run = exporter_with_run
        out = tmp_path / "nested" / "dir" / "report.md"
        exporter.export_to_file(run.id, str(out))
        assert out.exists()
