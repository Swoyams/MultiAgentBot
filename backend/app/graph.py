from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents import (
    budget_agent_node,
    coding_agent_node,
    critic_validator_agent_node,
    dispatcher_node,
    formatter_node,
    memory_agent_node,
    preprocess_node,
    research_agent_node,
    route_after_agent,
    route_agent,
    supervisor_agent_node,
    travel_planner_agent_node,
)
from .state import ChatState


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("memory", memory_agent_node)
    graph.add_node("supervisor", supervisor_agent_node)
    graph.add_node("dispatch", dispatcher_node)

    graph.add_node("research", research_agent_node)
    graph.add_node("coding", coding_agent_node)
    graph.add_node("travel", travel_planner_agent_node)
    graph.add_node("budget", budget_agent_node)

    graph.add_node("critic", critic_validator_agent_node)
    graph.add_node("formatter", formatter_node)

    graph.set_entry_point("preprocess")

    graph.add_edge("preprocess", "memory")
    graph.add_edge("memory", "supervisor")
    graph.add_edge("supervisor", "dispatch")

    graph.add_conditional_edges(
        "dispatch",
        route_agent,
        {
            "research": "research",
            "coding": "coding",
            "travel": "travel",
            "budget": "budget",
            "critic": "critic",
        },
    )

    for node_name in ["research", "coding", "travel", "budget"]:
        graph.add_conditional_edges(
            node_name,
            route_after_agent,
            {
                "dispatch": "dispatch",
                "critic": "critic",
            },
        )

    graph.add_edge("critic", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
