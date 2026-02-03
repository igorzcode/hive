"""
Tests for the dry-run execution planner.

Tests cover:
- ExecutionPlan generation from graphs
- Node and edge plan extraction
- Path tracing algorithms
- Complexity metrics calculation
- Cycle detection
- Human-readable and JSON output formatting
- CLI integration
"""

import json

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.goal import Constraint, Goal, SuccessCriterion
from framework.graph.node import NodeSpec
from framework.runner.planner import (
    ComplexityMetrics,
    EdgePlan,
    ExecutionPath,
    ExecutionPlan,
    NodePlan,
    plan_execution,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_linear_graph() -> tuple[GraphSpec, Goal]:
    """Create a simple linear graph: A -> B -> C (terminal)."""
    nodes = [
        NodeSpec(
            id="node-a",
            name="Node A",
            description="First node",
            node_type="llm_generate",
            input_keys=["input"],
            output_keys=["a_output"],
        ),
        NodeSpec(
            id="node-b",
            name="Node B",
            description="Second node",
            node_type="llm_tool_use",
            input_keys=["a_output"],
            output_keys=["b_output"],
            tools=["search_tool", "write_tool"],
        ),
        NodeSpec(
            id="node-c",
            name="Node C",
            description="Final node",
            node_type="llm_generate",
            input_keys=["b_output"],
            output_keys=["final_output"],
        ),
    ]

    edges = [
        EdgeSpec(
            id="a-to-b",
            source="node-a",
            target="node-b",
            condition=EdgeCondition.ON_SUCCESS,
        ),
        EdgeSpec(
            id="b-to-c",
            source="node-b",
            target="node-c",
            condition=EdgeCondition.ON_SUCCESS,
        ),
    ]

    graph = GraphSpec(
        id="simple-linear-agent",
        goal_id="test-goal",
        entry_node="node-a",
        terminal_nodes=["node-c"],
        nodes=nodes,
        edges=edges,
        description="A simple linear agent for testing",
    )

    goal = Goal(
        id="test-goal",
        name="Test Goal",
        description="A test goal for the linear agent",
        success_criteria=[
            SuccessCriterion(
                id="sc1",
                description="Complete all steps",
                metric="steps_completed",
                target="3",
            )
        ],
        constraints=[
            Constraint(
                id="c1",
                description="No errors allowed",
                constraint_type="hard",
                category="quality",
            )
        ],
    )

    return graph, goal


@pytest.fixture
def branching_graph() -> tuple[GraphSpec, Goal]:
    """Create a graph with branching: A -> (B or C) -> D."""
    nodes = [
        NodeSpec(
            id="start",
            name="Start Node",
            description="Entry point",
            node_type="llm_generate",
            input_keys=["query"],
            output_keys=["result", "confidence"],
        ),
        NodeSpec(
            id="high-confidence",
            name="High Confidence Path",
            description="Process high confidence results",
            node_type="llm_generate",
            input_keys=["result"],
            output_keys=["processed"],
        ),
        NodeSpec(
            id="low-confidence",
            name="Low Confidence Path",
            description="Process low confidence results with extra validation",
            node_type="llm_tool_use",
            input_keys=["result"],
            output_keys=["processed"],
            tools=["validate_tool"],
        ),
        NodeSpec(
            id="finish",
            name="Finish Node",
            description="Terminal node",
            node_type="llm_generate",
            input_keys=["processed"],
            output_keys=["final"],
        ),
    ]

    edges = [
        EdgeSpec(
            id="start-to-high",
            source="start",
            target="high-confidence",
            condition=EdgeCondition.CONDITIONAL,
            condition_expr="confidence > 0.8",
            priority=1,
        ),
        EdgeSpec(
            id="start-to-low",
            source="start",
            target="low-confidence",
            condition=EdgeCondition.CONDITIONAL,
            condition_expr="confidence <= 0.8",
            priority=0,
        ),
        EdgeSpec(
            id="high-to-finish",
            source="high-confidence",
            target="finish",
            condition=EdgeCondition.ON_SUCCESS,
        ),
        EdgeSpec(
            id="low-to-finish",
            source="low-confidence",
            target="finish",
            condition=EdgeCondition.ON_SUCCESS,
        ),
    ]

    graph = GraphSpec(
        id="branching-agent",
        goal_id="branch-goal",
        entry_node="start",
        terminal_nodes=["finish"],
        nodes=nodes,
        edges=edges,
        description="An agent with conditional branching",
    )

    goal = Goal(
        id="branch-goal",
        name="Branching Goal",
        description="Test branching execution paths",
    )

    return graph, goal


@pytest.fixture
def graph_with_cycle() -> tuple[GraphSpec, Goal]:
    """Create a graph with a cycle: A -> B -> C -> A (cycle back)."""
    nodes = [
        NodeSpec(
            id="loop-start",
            name="Loop Start",
            description="Start of loop",
            node_type="llm_generate",
            input_keys=["input"],
            output_keys=["output"],
        ),
        NodeSpec(
            id="loop-middle",
            name="Loop Middle",
            description="Middle of loop",
            node_type="llm_generate",
            input_keys=["output"],
            output_keys=["result"],
        ),
        NodeSpec(
            id="loop-check",
            name="Loop Check",
            description="Check if done",
            node_type="router",
            input_keys=["result"],
            output_keys=[],
            routes={"done": "loop-end", "continue": "loop-start"},
        ),
        NodeSpec(
            id="loop-end",
            name="Loop End",
            description="Exit loop",
            node_type="llm_generate",
            input_keys=["result"],
            output_keys=["final"],
        ),
    ]

    edges = [
        EdgeSpec(
            id="start-to-middle",
            source="loop-start",
            target="loop-middle",
            condition=EdgeCondition.ON_SUCCESS,
        ),
        EdgeSpec(
            id="middle-to-check",
            source="loop-middle",
            target="loop-check",
            condition=EdgeCondition.ON_SUCCESS,
        ),
        EdgeSpec(
            id="check-to-start",
            source="loop-check",
            target="loop-start",
            condition=EdgeCondition.CONDITIONAL,
            condition_expr="route == 'continue'",
        ),
        EdgeSpec(
            id="check-to-end",
            source="loop-check",
            target="loop-end",
            condition=EdgeCondition.CONDITIONAL,
            condition_expr="route == 'done'",
        ),
    ]

    graph = GraphSpec(
        id="looping-agent",
        goal_id="loop-goal",
        entry_node="loop-start",
        terminal_nodes=["loop-end"],
        nodes=nodes,
        edges=edges,
        description="An agent with a loop/cycle",
    )

    goal = Goal(
        id="loop-goal",
        name="Loop Goal",
        description="Test loop detection",
    )

    return graph, goal


@pytest.fixture
def graph_with_pause() -> tuple[GraphSpec, Goal]:
    """Create a graph with a pause node for HITL."""
    nodes = [
        NodeSpec(
            id="prepare",
            name="Prepare",
            description="Prepare data",
            node_type="llm_generate",
            input_keys=["input"],
            output_keys=["prepared"],
        ),
        NodeSpec(
            id="human-review",
            name="Human Review",
            description="Wait for human approval",
            node_type="human_input",
            input_keys=["prepared"],
            output_keys=["approved", "feedback"],
        ),
        NodeSpec(
            id="execute",
            name="Execute",
            description="Execute after approval",
            node_type="llm_tool_use",
            input_keys=["approved"],
            output_keys=["result"],
            tools=["execute_action"],
        ),
    ]

    edges = [
        EdgeSpec(
            id="prepare-to-review",
            source="prepare",
            target="human-review",
            condition=EdgeCondition.ON_SUCCESS,
        ),
        EdgeSpec(
            id="review-to-execute",
            source="human-review",
            target="execute",
            condition=EdgeCondition.ON_SUCCESS,
        ),
    ]

    graph = GraphSpec(
        id="hitl-agent",
        goal_id="hitl-goal",
        entry_node="prepare",
        terminal_nodes=["execute"],
        pause_nodes=["human-review"],
        nodes=nodes,
        edges=edges,
        description="An agent with human-in-the-loop",
    )

    goal = Goal(
        id="hitl-goal",
        name="HITL Goal",
        description="Test pause node handling",
    )

    return graph, goal


# =============================================================================
# Test plan_execution() function
# =============================================================================


class TestPlanExecution:
    """Tests for the plan_execution function."""

    def test_simple_linear_plan(self, simple_linear_graph):
        """Test planning a simple linear graph."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        assert plan.agent_name == "simple-linear-agent"
        assert plan.goal_name == "Test Goal"
        assert plan.entry_node == "node-a"
        assert plan.terminal_nodes == ["node-c"]
        assert plan.is_valid is True
        assert len(plan.validation_errors) == 0

    def test_node_extraction(self, simple_linear_graph):
        """Test that all nodes are extracted correctly."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        assert len(plan.nodes) == 3

        # Find node A
        node_a = next(n for n in plan.nodes if n.id == "node-a")
        assert node_a.name == "Node A"
        assert node_a.is_entry is True
        assert node_a.is_terminal is False
        assert node_a.node_type == "llm_generate"
        assert node_a.input_keys == ["input"]
        assert node_a.output_keys == ["a_output"]

        # Find node C (terminal)
        node_c = next(n for n in plan.nodes if n.id == "node-c")
        assert node_c.is_terminal is True
        assert node_c.is_entry is False

    def test_edge_extraction(self, simple_linear_graph):
        """Test that all edges are extracted correctly."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        assert len(plan.edges) == 2

        # Check edge a-to-b
        edge_ab = next(e for e in plan.edges if e.id == "a-to-b")
        assert edge_ab.source == "node-a"
        assert edge_ab.target == "node-b"
        assert edge_ab.condition == "on_success"

    def test_outgoing_edges_attached_to_nodes(self, simple_linear_graph):
        """Test that outgoing edges are attached to their source nodes."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        node_a = next(n for n in plan.nodes if n.id == "node-a")
        assert len(node_a.outgoing_edges) == 1
        assert node_a.outgoing_edges[0].target == "node-b"

        node_c = next(n for n in plan.nodes if n.id == "node-c")
        assert len(node_c.outgoing_edges) == 0  # Terminal node

    def test_tool_extraction(self, simple_linear_graph):
        """Test that tools are extracted from nodes."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        node_b = next(n for n in plan.nodes if n.id == "node-b")
        assert "search_tool" in node_b.tools
        assert "write_tool" in node_b.tools

        # Check metrics include unique tools
        assert "search_tool" in plan.metrics.unique_tools
        assert "write_tool" in plan.metrics.unique_tools


class TestComplexityMetrics:
    """Tests for complexity metrics calculation."""

    def test_linear_graph_metrics(self, simple_linear_graph):
        """Test metrics for a simple linear graph."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)
        metrics = plan.metrics

        assert metrics.total_nodes == 3
        assert metrics.llm_nodes == 3  # All are LLM nodes
        assert metrics.tool_nodes == 1  # Only node-b has tools
        assert metrics.router_nodes == 0
        assert metrics.function_nodes == 0
        assert metrics.total_edges == 2
        assert metrics.has_cycles is False
        assert metrics.entry_points == 1
        assert metrics.terminal_points == 1
        assert metrics.pause_points == 0

    def test_branching_graph_metrics(self, branching_graph):
        """Test metrics for a branching graph."""
        graph, goal = branching_graph
        plan = plan_execution(graph, goal)
        metrics = plan.metrics

        assert metrics.total_nodes == 4
        assert metrics.total_edges == 4
        assert metrics.has_cycles is False

    def test_cycle_detection(self, graph_with_cycle):
        """Test that cycles are detected in the graph."""
        graph, goal = graph_with_cycle
        plan = plan_execution(graph, goal)

        assert plan.metrics.has_cycles is True

    def test_pause_point_counting(self, graph_with_pause):
        """Test that pause points are counted correctly."""
        graph, goal = graph_with_pause
        plan = plan_execution(graph, goal)

        assert plan.metrics.pause_points == 1
        assert "human-review" in plan.pause_nodes


class TestExecutionPaths:
    """Tests for execution path tracing."""

    def test_linear_primary_path(self, simple_linear_graph):
        """Test primary path for linear graph."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        assert plan.primary_path.is_primary is True
        assert plan.primary_path.nodes == ["node-a", "node-b", "node-c"]
        assert plan.primary_path.ends_at_terminal is True

    def test_branching_alternate_paths(self, branching_graph):
        """Test that branching creates alternate paths."""
        graph, goal = branching_graph
        plan = plan_execution(graph, goal)

        # Should have at least 2 paths (high and low confidence)
        total_paths = 1 + len(plan.alternate_paths)
        assert total_paths >= 2

        # All paths should end at the terminal node
        all_paths = [plan.primary_path] + plan.alternate_paths
        for path in all_paths:
            assert "finish" in path.nodes

    def test_pause_node_path(self, graph_with_pause):
        """Test that paths recognize pause nodes."""
        graph, goal = graph_with_pause
        plan = plan_execution(graph, goal)

        # Find paths that end at the pause node
        all_paths = [plan.primary_path] + plan.alternate_paths

        # At least one path should contain the pause node
        paths_with_pause = [p for p in all_paths if "human-review" in p.nodes]
        assert len(paths_with_pause) > 0


class TestValidation:
    """Tests for graph validation in plans."""

    def test_valid_graph_passes(self, simple_linear_graph):
        """Test that valid graphs pass validation."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        assert plan.is_valid is True
        assert len(plan.validation_errors) == 0

    def test_missing_entry_node_detected(self):
        """Test that missing entry node is detected."""
        nodes = [
            NodeSpec(
                id="orphan",
                name="Orphan Node",
                description="Not connected",
                node_type="llm_generate",
            )
        ]

        graph = GraphSpec(
            id="invalid-agent",
            goal_id="test",
            entry_node="nonexistent",  # Entry node doesn't exist
            terminal_nodes=[],
            nodes=nodes,
            edges=[],
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        assert plan.is_valid is False
        assert any("Entry node" in e for e in plan.validation_errors)

    def test_invalid_edge_reference_detected(self):
        """Test that invalid edge references are detected."""
        nodes = [
            NodeSpec(
                id="start",
                name="Start",
                description="Start node",
                node_type="llm_generate",
            )
        ]

        edges = [
            EdgeSpec(
                id="bad-edge",
                source="start",
                target="nonexistent",  # Target doesn't exist
                condition=EdgeCondition.ON_SUCCESS,
            )
        ]

        graph = GraphSpec(
            id="invalid-agent",
            goal_id="test",
            entry_node="start",
            terminal_nodes=["start"],
            nodes=nodes,
            edges=edges,
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        assert plan.is_valid is False
        assert any("nonexistent" in e for e in plan.validation_errors)


class TestOutputFormatting:
    """Tests for output formatting."""

    def test_human_readable_format(self, simple_linear_graph):
        """Test human-readable output format."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        output = plan.format_readable()

        # Check that key sections are present
        assert "EXECUTION PLAN" in output
        assert "simple-linear-agent" in output
        assert "COMPLEXITY METRICS" in output
        assert "NODE DETAILS" in output
        assert "EXECUTION PATHS" in output
        assert "node-a" in output
        assert "node-b" in output
        assert "node-c" in output
        assert "[ENTRY]" in output
        assert "[TERMINAL]" in output

    def test_human_readable_shows_tools(self, simple_linear_graph):
        """Test that tools are shown in human-readable output."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        output = plan.format_readable()

        assert "search_tool" in output
        assert "write_tool" in output

    def test_human_readable_shows_edges(self, branching_graph):
        """Test that conditional edges are shown."""
        graph, goal = branching_graph
        plan = plan_execution(graph, goal)

        output = plan.format_readable()

        # Should show conditional expressions
        assert "conditional" in output.lower()
        assert "confidence" in output


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = GraphSpec(
            id="empty",
            goal_id="test",
            entry_node="nonexistent",
            terminal_nodes=[],
            nodes=[],
            edges=[],
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        assert plan.is_valid is False
        assert len(plan.nodes) == 0
        assert len(plan.edges) == 0

    def test_single_node_graph(self):
        """Test graph with single node."""
        nodes = [
            NodeSpec(
                id="only-node",
                name="Only Node",
                description="Single node",
                node_type="llm_generate",
            )
        ]

        graph = GraphSpec(
            id="single-node",
            goal_id="test",
            entry_node="only-node",
            terminal_nodes=["only-node"],
            nodes=nodes,
            edges=[],
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        assert plan.is_valid is True
        assert len(plan.nodes) == 1
        assert plan.nodes[0].is_entry is True
        assert plan.nodes[0].is_terminal is True
        assert plan.primary_path.nodes == ["only-node"]

    def test_node_with_all_types(self):
        """Test graph with all node types."""
        nodes = [
            NodeSpec(
                id="llm-gen",
                name="LLM Generate",
                description="Generate text",
                node_type="llm_generate",
            ),
            NodeSpec(
                id="llm-tool",
                name="LLM Tool Use",
                description="Use tools",
                node_type="llm_tool_use",
                tools=["my_tool"],
            ),
            NodeSpec(
                id="router",
                name="Router",
                description="Route",
                node_type="router",
                routes={"a": "llm-gen", "b": "llm-tool"},
            ),
            NodeSpec(
                id="human",
                name="Human Input",
                description="Wait for human",
                node_type="human_input",
            ),
        ]

        edges = [
            EdgeSpec(id="e1", source="llm-gen", target="router", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="e2", source="router", target="llm-tool", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e3", source="llm-tool", target="human", condition=EdgeCondition.ON_SUCCESS),
        ]

        graph = GraphSpec(
            id="all-types",
            goal_id="test",
            entry_node="llm-gen",
            terminal_nodes=["human"],
            nodes=nodes,
            edges=edges,
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        # Check all node types are counted
        assert plan.metrics.llm_nodes == 2  # llm_generate + llm_tool_use
        assert plan.metrics.tool_nodes == 1  # llm_tool_use
        assert plan.metrics.router_nodes == 1
        assert plan.metrics.human_input_nodes == 1


class TestAsyncEntryPoints:
    """Tests for async entry points handling."""

    def test_async_entry_points_extracted(self):
        """Test that async entry points are extracted."""
        from framework.graph.edge import AsyncEntryPointSpec

        nodes = [
            NodeSpec(
                id="webhook-handler",
                name="Webhook Handler",
                description="Handle webhooks",
                node_type="llm_generate",
            ),
            NodeSpec(
                id="api-handler",
                name="API Handler",
                description="Handle API calls",
                node_type="llm_generate",
            ),
        ]

        async_entry_points = [
            AsyncEntryPointSpec(
                id="webhook",
                name="Webhook Entry",
                entry_node="webhook-handler",
                trigger_type="webhook",
                isolation_level="isolated",
            ),
            AsyncEntryPointSpec(
                id="api",
                name="API Entry",
                entry_node="api-handler",
                trigger_type="api",
                isolation_level="shared",
            ),
        ]

        graph = GraphSpec(
            id="multi-entry",
            goal_id="test",
            entry_node="webhook-handler",
            terminal_nodes=["webhook-handler", "api-handler"],
            nodes=nodes,
            edges=[],
            async_entry_points=async_entry_points,
        )

        goal = Goal(id="test", name="Test", description="Test")
        plan = plan_execution(graph, goal)

        assert len(plan.async_entry_points) == 2
        assert plan.metrics.entry_points == 3  # 1 main + 2 async

        webhook_ep = next(ep for ep in plan.async_entry_points if ep["id"] == "webhook")
        assert webhook_ep["trigger_type"] == "webhook"
        assert webhook_ep["isolation_level"] == "isolated"


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full dry-run workflow."""

    def test_full_plan_to_json(self, simple_linear_graph):
        """Test that a plan can be converted to JSON."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        # This simulates what the CLI does
        output_data = {
            "agent_name": plan.agent_name,
            "is_valid": plan.is_valid,
            "metrics": {
                "total_nodes": plan.metrics.total_nodes,
                "has_cycles": plan.metrics.has_cycles,
            },
            "primary_path": {
                "nodes": plan.primary_path.nodes,
            },
        }

        # Should be JSON serializable
        json_str = json.dumps(output_data, indent=2)
        parsed = json.loads(json_str)

        assert parsed["agent_name"] == "simple-linear-agent"
        assert parsed["is_valid"] is True

    def test_format_readable_is_valid_string(self, simple_linear_graph):
        """Test that format_readable returns a valid string."""
        graph, goal = simple_linear_graph
        plan = plan_execution(graph, goal)

        output = plan.format_readable()

        assert isinstance(output, str)
        assert len(output) > 100  # Should have substantial content
        assert "\n" in output  # Should be multi-line
