"""
Execution Planner - Traces agent graphs without executing them.

The planner analyzes an agent's graph structure and produces an execution plan
that shows the possible paths through the agent without making any LLM calls
or executing any side effects.

This is useful for:
- Understanding what an agent will do before running it
- Debugging graph structure issues
- Documentation and code review
- Estimating complexity and cost
"""

from dataclasses import dataclass, field
from typing import Any

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.node import NodeSpec


@dataclass
class NodePlan:
    """Plan for a single node in the execution path."""

    id: str
    name: str
    description: str
    node_type: str
    input_keys: list[str]
    output_keys: list[str]
    tools: list[str]
    is_entry: bool = False
    is_terminal: bool = False
    is_pause: bool = False
    outgoing_edges: list["EdgePlan"] = field(default_factory=list)
    system_prompt_preview: str | None = None
    model: str | None = None


@dataclass
class EdgePlan:
    """Plan for an edge between nodes."""

    id: str
    source: str
    target: str
    condition: str
    condition_expr: str | None = None
    description: str = ""
    priority: int = 0
    input_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionPath:
    """A possible execution path through the graph."""

    nodes: list[str]
    description: str
    is_primary: bool = False
    ends_at_terminal: bool = False
    ends_at_pause: bool = False
    condition_chain: list[str] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """Metrics estimating execution complexity."""

    total_nodes: int
    llm_nodes: int
    tool_nodes: int
    router_nodes: int
    function_nodes: int
    human_input_nodes: int
    total_edges: int
    max_path_length: int
    has_cycles: bool
    unique_tools: list[str]
    pause_points: int
    entry_points: int
    terminal_points: int


@dataclass
class ExecutionPlan:
    """Complete execution plan for an agent."""

    agent_name: str
    agent_description: str
    goal_name: str
    goal_description: str

    # Graph structure
    entry_node: str
    terminal_nodes: list[str]
    pause_nodes: list[str]
    async_entry_points: list[dict[str, Any]]

    # Node and edge details
    nodes: list[NodePlan]
    edges: list[EdgePlan]

    # Execution paths
    primary_path: ExecutionPath
    alternate_paths: list[ExecutionPath]

    # Complexity metrics
    metrics: ComplexityMetrics

    # Validation
    is_valid: bool
    validation_errors: list[str]
    validation_warnings: list[str]

    def format_readable(self) -> str:
        """Format the execution plan as a human-readable string."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"EXECUTION PLAN: {self.agent_name}")
        lines.append("=" * 70)
        lines.append("")

        # Goal
        lines.append(f"Goal: {self.goal_name}")
        lines.append(f"  {self.goal_description}")
        lines.append("")

        # Validation status
        if self.is_valid:
            lines.append("[OK] Agent is valid")
        else:
            lines.append("[ERROR] Agent has validation errors:")
            for error in self.validation_errors:
                lines.append(f"  - {error}")

        if self.validation_warnings:
            lines.append("\nWarnings:")
            for warning in self.validation_warnings:
                lines.append(f"  - {warning}")
        lines.append("")

        # Complexity metrics
        lines.append("-" * 70)
        lines.append("COMPLEXITY METRICS")
        lines.append("-" * 70)
        lines.append(f"  Total Nodes:        {self.metrics.total_nodes}")
        lines.append(f"  LLM Nodes:          {self.metrics.llm_nodes}")
        lines.append(f"  Tool Nodes:         {self.metrics.tool_nodes}")
        lines.append(f"  Router Nodes:       {self.metrics.router_nodes}")
        lines.append(f"  Function Nodes:     {self.metrics.function_nodes}")
        lines.append(f"  Human Input Nodes:  {self.metrics.human_input_nodes}")
        lines.append(f"  Total Edges:        {self.metrics.total_edges}")
        lines.append(f"  Max Path Length:    {self.metrics.max_path_length}")
        lines.append(f"  Has Cycles:         {'Yes' if self.metrics.has_cycles else 'No'}")
        lines.append(f"  Entry Points:       {self.metrics.entry_points}")
        lines.append(f"  Terminal Points:    {self.metrics.terminal_points}")
        lines.append(f"  Pause Points:       {self.metrics.pause_points}")
        lines.append("")

        # Unique tools
        if self.metrics.unique_tools:
            lines.append(f"  Required Tools ({len(self.metrics.unique_tools)}):")
            for tool in sorted(self.metrics.unique_tools):
                lines.append(f"    - {tool}")
            lines.append("")

        # Node details
        lines.append("-" * 70)
        lines.append("NODE DETAILS")
        lines.append("-" * 70)

        for node in self.nodes:
            marker = ""
            if node.is_entry:
                marker = " [ENTRY]"
            elif node.is_terminal:
                marker = " [TERMINAL]"
            elif node.is_pause:
                marker = " [PAUSE]"

            lines.append(f"\n  {node.id}: {node.name}{marker}")
            lines.append(f"    Type: {node.node_type}")
            lines.append(f"    Description: {node.description[:80]}{'...' if len(node.description) > 80 else ''}")

            if node.input_keys:
                lines.append(f"    Inputs:  {', '.join(node.input_keys)}")
            if node.output_keys:
                lines.append(f"    Outputs: {', '.join(node.output_keys)}")
            if node.tools:
                lines.append(f"    Tools:   {', '.join(node.tools)}")
            if node.model:
                lines.append(f"    Model:   {node.model}")

            if node.outgoing_edges:
                lines.append("    Edges:")
                for edge in node.outgoing_edges:
                    cond_str = edge.condition
                    if edge.condition_expr:
                        cond_str += f" ({edge.condition_expr})"
                    lines.append(f"      -> {edge.target} [{cond_str}]")

        lines.append("")

        # Execution paths
        lines.append("-" * 70)
        lines.append("EXECUTION PATHS")
        lines.append("-" * 70)

        lines.append(f"\n  Primary Path:")
        lines.append(f"    {' -> '.join(self.primary_path.nodes)}")
        if self.primary_path.ends_at_terminal:
            lines.append("    (ends at terminal node)")
        elif self.primary_path.ends_at_pause:
            lines.append("    (ends at pause node)")

        if self.alternate_paths:
            lines.append(f"\n  Alternate Paths ({len(self.alternate_paths)}):")
            for i, path in enumerate(self.alternate_paths[:5], 1):  # Show max 5
                lines.append(f"    {i}. {' -> '.join(path.nodes)}")
                if path.condition_chain:
                    lines.append(f"       Conditions: {' AND '.join(path.condition_chain)}")

            if len(self.alternate_paths) > 5:
                lines.append(f"    ... and {len(self.alternate_paths) - 5} more paths")

        lines.append("")

        # Async entry points
        if self.async_entry_points:
            lines.append("-" * 70)
            lines.append("ASYNC ENTRY POINTS")
            lines.append("-" * 70)
            for ep in self.async_entry_points:
                lines.append(f"\n  {ep['id']}: {ep.get('name', ep['id'])}")
                lines.append(f"    Entry Node: {ep['entry_node']}")
                lines.append(f"    Trigger: {ep.get('trigger_type', 'manual')}")
                lines.append(f"    Isolation: {ep.get('isolation_level', 'shared')}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def plan_execution(
    graph: GraphSpec,
    goal: Any,
    tool_registry: Any | None = None,
) -> ExecutionPlan:
    """
    Trace the graph and produce an execution plan without running anything.

    This function analyzes the graph structure and produces a complete plan
    showing what the agent would do when executed, including:
    - All nodes and their configurations
    - All edges and their conditions
    - Possible execution paths
    - Complexity metrics
    - Validation status

    Args:
        graph: The GraphSpec to analyze
        goal: The Goal object driving the agent
        tool_registry: Optional tool registry to check tool availability

    Returns:
        ExecutionPlan with full analysis
    """
    # Validate the graph
    validation_errors = graph.validate()
    validation_warnings = []

    # Build node plans
    nodes: list[NodePlan] = []
    node_map: dict[str, NodePlan] = {}

    for node_spec in graph.nodes:
        node_plan = NodePlan(
            id=node_spec.id,
            name=node_spec.name,
            description=node_spec.description,
            node_type=node_spec.node_type,
            input_keys=node_spec.input_keys,
            output_keys=node_spec.output_keys,
            tools=node_spec.tools or [],
            is_entry=(node_spec.id == graph.entry_node),
            is_terminal=(node_spec.id in graph.terminal_nodes),
            is_pause=(node_spec.id in graph.pause_nodes),
            system_prompt_preview=_truncate(node_spec.system_prompt, 100) if node_spec.system_prompt else None,
            model=node_spec.model,
        )
        nodes.append(node_plan)
        node_map[node_spec.id] = node_plan

    # Build edge plans and attach to nodes
    edges: list[EdgePlan] = []

    for edge_spec in graph.edges:
        edge_plan = EdgePlan(
            id=edge_spec.id,
            source=edge_spec.source,
            target=edge_spec.target,
            condition=edge_spec.condition.value,
            condition_expr=edge_spec.condition_expr,
            description=edge_spec.description,
            priority=edge_spec.priority,
            input_mapping=edge_spec.input_mapping,
        )
        edges.append(edge_plan)

        # Attach to source node
        if edge_spec.source in node_map:
            node_map[edge_spec.source].outgoing_edges.append(edge_plan)

    # Sort outgoing edges by priority (highest first)
    for node in nodes:
        node.outgoing_edges.sort(key=lambda e: -e.priority)

    # Calculate complexity metrics
    metrics = _calculate_metrics(graph, nodes, edges)

    # Find execution paths
    primary_path, alternate_paths = _find_execution_paths(graph, node_map)

    # Check tool availability
    if tool_registry is not None:
        for node in nodes:
            for tool_name in node.tools:
                if not tool_registry.has_tool(tool_name):
                    validation_warnings.append(
                        f"Node '{node.id}' requires tool '{tool_name}' which is not registered"
                    )

    # Build async entry points info
    async_entry_points = [
        {
            "id": ep.id,
            "name": ep.name,
            "entry_node": ep.entry_node,
            "trigger_type": ep.trigger_type,
            "isolation_level": ep.isolation_level,
            "max_concurrent": ep.max_concurrent,
        }
        for ep in graph.async_entry_points
    ]

    return ExecutionPlan(
        agent_name=graph.id,
        agent_description=graph.description,
        goal_name=goal.name if goal else "Unknown",
        goal_description=goal.description if goal else "",
        entry_node=graph.entry_node,
        terminal_nodes=graph.terminal_nodes,
        pause_nodes=graph.pause_nodes,
        async_entry_points=async_entry_points,
        nodes=nodes,
        edges=edges,
        primary_path=primary_path,
        alternate_paths=alternate_paths,
        metrics=metrics,
        is_valid=len(validation_errors) == 0,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )


def _truncate(text: str | None, max_length: int) -> str | None:
    """Truncate text to max_length, adding ellipsis if needed."""
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _calculate_metrics(
    graph: GraphSpec,
    nodes: list[NodePlan],
    edges: list[EdgePlan],
) -> ComplexityMetrics:
    """Calculate complexity metrics for the graph."""
    # Count node types
    llm_nodes = sum(1 for n in nodes if n.node_type in ("llm_generate", "llm_tool_use"))
    tool_nodes = sum(1 for n in nodes if n.node_type == "llm_tool_use")
    router_nodes = sum(1 for n in nodes if n.node_type == "router")
    function_nodes = sum(1 for n in nodes if n.node_type == "function")
    human_input_nodes = sum(1 for n in nodes if n.node_type == "human_input")

    # Collect unique tools
    unique_tools: set[str] = set()
    for node in nodes:
        unique_tools.update(node.tools)

    # Calculate max path length and check for cycles
    max_path_length, has_cycles = _analyze_paths(graph)

    # Count entry points (main + async)
    entry_points = 1 + len(graph.async_entry_points)

    return ComplexityMetrics(
        total_nodes=len(nodes),
        llm_nodes=llm_nodes,
        tool_nodes=tool_nodes,
        router_nodes=router_nodes,
        function_nodes=function_nodes,
        human_input_nodes=human_input_nodes,
        total_edges=len(edges),
        max_path_length=max_path_length,
        has_cycles=has_cycles,
        unique_tools=sorted(unique_tools),
        pause_points=len(graph.pause_nodes),
        entry_points=entry_points,
        terminal_points=len(graph.terminal_nodes),
    )


def _analyze_paths(graph: GraphSpec) -> tuple[int, bool]:
    """
    Analyze paths in the graph to find max length and detect cycles.

    Returns:
        Tuple of (max_path_length, has_cycles)
    """
    if not graph.entry_node:
        return 0, False

    # DFS to find max path length and detect cycles
    max_length = 0
    has_cycles = False

    def dfs(node_id: str, path: set[str], length: int) -> int:
        nonlocal has_cycles

        if node_id in path:
            has_cycles = True
            return length

        if node_id in graph.terminal_nodes or node_id in graph.pause_nodes:
            return length

        path.add(node_id)
        max_child_length = length

        for edge in graph.get_outgoing_edges(node_id):
            child_length = dfs(edge.target, path.copy(), length + 1)
            max_child_length = max(max_child_length, child_length)

        return max_child_length

    max_length = dfs(graph.entry_node, set(), 1)

    return max_length, has_cycles


def _find_execution_paths(
    graph: GraphSpec,
    node_map: dict[str, NodePlan],
) -> tuple[ExecutionPath, list[ExecutionPath]]:
    """
    Find all possible execution paths through the graph.

    Returns:
        Tuple of (primary_path, alternate_paths)
    """
    if not graph.entry_node:
        return ExecutionPath(nodes=[], description="Empty graph", is_primary=True), []

    all_paths: list[ExecutionPath] = []

    def trace_path(
        node_id: str,
        current_path: list[str],
        conditions: list[str],
        visited: set[str],
    ) -> None:
        """Recursively trace paths through the graph."""
        if node_id in visited:
            # Cycle detected, stop this path
            path = ExecutionPath(
                nodes=current_path + [f"{node_id} (cycle)"],
                description=f"Path with cycle at {node_id}",
                condition_chain=conditions.copy(),
            )
            all_paths.append(path)
            return

        current_path = current_path + [node_id]
        visited = visited | {node_id}

        # Check if we've reached an end
        if node_id in graph.terminal_nodes:
            path = ExecutionPath(
                nodes=current_path,
                description=f"Path ending at terminal node {node_id}",
                ends_at_terminal=True,
                condition_chain=conditions.copy(),
            )
            all_paths.append(path)
            return

        if node_id in graph.pause_nodes:
            path = ExecutionPath(
                nodes=current_path,
                description=f"Path ending at pause node {node_id}",
                ends_at_pause=True,
                condition_chain=conditions.copy(),
            )
            all_paths.append(path)
            return

        # Get outgoing edges
        edges = graph.get_outgoing_edges(node_id)

        if not edges:
            # Dead end (no outgoing edges)
            path = ExecutionPath(
                nodes=current_path,
                description=f"Path ending at {node_id} (no outgoing edges)",
                condition_chain=conditions.copy(),
            )
            all_paths.append(path)
            return

        # Follow each edge
        for edge in edges:
            condition_desc = _describe_condition(edge)
            new_conditions = conditions + [condition_desc] if condition_desc else conditions
            trace_path(edge.target, current_path, new_conditions, visited)

    # Start tracing from entry node
    trace_path(graph.entry_node, [], [], set())

    # Determine primary path (first path that ends at terminal, or longest path)
    primary_path = None
    alternate_paths = []

    for path in all_paths:
        if primary_path is None:
            primary_path = path
            primary_path.is_primary = True
        elif path.ends_at_terminal and not primary_path.ends_at_terminal:
            # Prefer paths that end at terminal
            alternate_paths.append(primary_path)
            primary_path = path
            primary_path.is_primary = True
        elif path.ends_at_terminal and len(path.nodes) < len(primary_path.nodes):
            # Among terminal paths, prefer shorter ones (happy path)
            alternate_paths.append(primary_path)
            primary_path = path
            primary_path.is_primary = True
        else:
            alternate_paths.append(path)

    if primary_path is None:
        primary_path = ExecutionPath(
            nodes=[graph.entry_node],
            description="Single node graph",
            is_primary=True,
        )

    return primary_path, alternate_paths


def _describe_condition(edge: EdgePlan) -> str | None:
    """Generate a human-readable description of an edge condition."""
    if edge.condition == "always":
        return None
    elif edge.condition == "on_success":
        return f"{edge.source} succeeds"
    elif edge.condition == "on_failure":
        return f"{edge.source} fails"
    elif edge.condition == "conditional":
        expr = edge.condition_expr or "unknown"
        return f"when {expr}"
    elif edge.condition == "llm_decide":
        return f"LLM decides based on {edge.description or 'context'}"
    else:
        return edge.condition
