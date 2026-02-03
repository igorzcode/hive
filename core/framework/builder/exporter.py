"""
Run Summary Exporter - Generates structured reports from completed agent runs.

Exports run data as JSON (for programmatic use) or Markdown (for humans).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from framework.schemas.decision import Decision
from framework.schemas.run import Run, RunStatus
from framework.storage.backend import FileStorage


class RunSummaryExporter:
    """Exports completed run data as structured JSON or human-readable Markdown."""

    def __init__(self, storage: FileStorage):
        self.storage = storage

    def export_json(self, run_id: str) -> dict[str, Any]:
        """Export a run as a structured JSON-serializable dict.

        Returns a dict with: metadata, metrics, decisions, problems, and output.
        Raises ValueError if the run is not found.
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        return {
            "metadata": self._build_metadata(run),
            "metrics": self._build_metrics(run),
            "decisions": [self._build_decision_entry(d) for d in run.decisions],
            "problems": [self._build_problem_entry(p) for p in run.problems],
            "narrative": run.narrative,
            "input": run.input_data,
            "output": run.output_data,
        }

    def export_markdown(self, run_id: str) -> str:
        """Export a run as a formatted Markdown report.

        Raises ValueError if the run is not found.
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        sections = [
            self._md_header(run),
            self._md_narrative(run),
            self._md_metrics(run),
            self._md_decisions(run),
            self._md_problems(run),
            self._md_output(run),
        ]
        return "\n".join(sections)

    def export_to_file(
        self, run_id: str, output_path: str, fmt: str = "markdown"
    ) -> None:
        """Write a run report to a file.

        Args:
            run_id: The run to export.
            output_path: Destination file path.
            fmt: "json" or "markdown".
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            data = self.export_json(run_id)
            path.write_text(json.dumps(data, indent=2, default=str))
        else:
            text = self.export_markdown(run_id)
            path.write_text(text, encoding="utf-8")

    # ── JSON helpers ─────────────────────────────────────────

    def _build_metadata(self, run: Run) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "goal_id": run.goal_id,
            "goal_description": run.goal_description,
            "status": run.status.value,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "duration_ms": run.duration_ms,
        }

    def _build_metrics(self, run: Run) -> dict[str, Any]:
        m = run.metrics
        return {
            "total_decisions": m.total_decisions,
            "successful_decisions": m.successful_decisions,
            "failed_decisions": m.failed_decisions,
            "success_rate": round(m.success_rate, 4),
            "total_tokens": m.total_tokens,
            "total_latency_ms": m.total_latency_ms,
            "nodes_executed": m.nodes_executed,
            "edges_traversed": m.edges_traversed,
        }

    def _build_decision_entry(self, d: Decision) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": d.id,
            "node_id": d.node_id,
            "intent": d.intent,
            "type": d.decision_type.value,
            "chosen": d.chosen_option.description if d.chosen_option else None,
            "reasoning": d.reasoning,
            "successful": d.was_successful,
        }
        if d.outcome:
            entry["outcome"] = {
                "success": d.outcome.success,
                "summary": d.outcome.summary,
                "error": d.outcome.error,
                "tokens_used": d.outcome.tokens_used,
                "latency_ms": d.outcome.latency_ms,
            }
        if d.evaluation:
            entry["evaluation"] = {
                "goal_aligned": d.evaluation.goal_aligned,
                "outcome_quality": d.evaluation.outcome_quality,
                "better_option_existed": d.evaluation.better_option_existed,
            }
        return entry

    def _build_problem_entry(self, p) -> dict[str, Any]:
        return {
            "id": p.id,
            "severity": p.severity,
            "description": p.description,
            "root_cause": p.root_cause,
            "suggested_fix": p.suggested_fix,
            "decision_id": p.decision_id,
        }

    # ── Markdown helpers ─────────────────────────────────────

    def _md_header(self, run: Run) -> str:
        badge = "COMPLETED" if run.status == RunStatus.COMPLETED else run.status.value.upper()
        duration = self._format_duration(run.duration_ms)
        lines = [
            f"# Run Report: {run.id}",
            "",
            f"**Status:** {badge}  ",
            f"**Goal:** {run.goal_description or run.goal_id}  ",
            f"**Started:** {run.started_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Duration:** {duration}",
            "",
        ]
        return "\n".join(lines)

    def _md_narrative(self, run: Run) -> str:
        if not run.narrative:
            return ""
        return f"## Summary\n\n{run.narrative}\n"

    def _md_metrics(self, run: Run) -> str:
        m = run.metrics
        lines = [
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Decisions | {m.total_decisions} |",
            f"| Successful | {m.successful_decisions} |",
            f"| Failed | {m.failed_decisions} |",
            f"| Success Rate | {m.success_rate:.0%} |",
            f"| Tokens Used | {m.total_tokens} |",
            f"| Latency | {m.total_latency_ms}ms |",
            f"| Nodes Executed | {len(m.nodes_executed)} |",
            "",
        ]
        return "\n".join(lines)

    def _md_decisions(self, run: Run) -> str:
        if not run.decisions:
            return "## Decisions\n\nNo decisions recorded.\n"
        lines = ["## Decisions", ""]
        for i, d in enumerate(run.decisions, 1):
            icon = "+" if d.was_successful else "-"
            chosen = d.chosen_option.description if d.chosen_option else "unknown"
            line = f"{i}. **[{d.node_id}]** {d.intent} -> {chosen}"
            if d.outcome and d.outcome.error:
                line += f"  \n   Error: {d.outcome.error}"
            lines.append(line)
        lines.append("")
        return "\n".join(lines)

    def _md_problems(self, run: Run) -> str:
        if not run.problems:
            return ""
        critical = [p for p in run.problems if p.severity == "critical"]
        warnings = [p for p in run.problems if p.severity == "warning"]
        minor = [p for p in run.problems if p.severity == "minor"]

        lines = ["## Problems", ""]
        for label, group in [("Critical", critical), ("Warnings", warnings), ("Minor", minor)]:
            if group:
                lines.append(f"### {label}")
                lines.append("")
                for p in group:
                    lines.append(f"- {p.description}")
                    if p.root_cause:
                        lines.append(f"  - Root cause: {p.root_cause}")
                    if p.suggested_fix:
                        lines.append(f"  - Fix: {p.suggested_fix}")
                lines.append("")
        return "\n".join(lines)

    def _md_output(self, run: Run) -> str:
        if not run.output_data:
            return ""
        lines = [
            "## Output",
            "",
            "```json",
            json.dumps(run.output_data, indent=2, default=str),
            "```",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_duration(ms: int) -> str:
        if ms == 0:
            return "N/A"
        seconds = ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"
