from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents import (
    budget_agent_node,
    coding_agent_node,
    critic_validator_agent_node,
    dispatcher_node,
    formatter_node,
    general_agent_node,
    memory_agent_node,
    preprocess_node,
    research_agent_node,
    retry_node,
    route_after_agent,
    route_after_critic,
    route_agent,
    supervisor_agent_node,
    travel_planner_agent_node,
)
from .state import ChatState


def build_graph():
    graph = StateGraph(ChatState)


    graph.add_node("preprocess",  preprocess_node)
    graph.add_node("memory_agent", memory_agent_node)
    graph.add_node("supervisor",  supervisor_agent_node)
    graph.add_node("dispatch",    dispatcher_node)

    graph.add_node("general",  general_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("coding",   coding_agent_node)
    graph.add_node("travel",   travel_planner_agent_node)
    graph.add_node("budget",   budget_agent_node)

    graph.add_node("critic",    critic_validator_agent_node)
    graph.add_node("retry",     retry_node)
    graph.add_node("formatter", formatter_node)

    
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess",   "memory_agent")
    graph.add_edge("memory_agent", "supervisor")
    graph.add_edge("supervisor",   "dispatch")

    # ── Conditional: dispatch → active agent (or critic if queue empty) ──────
    graph.add_conditional_edges(
        "dispatch",
        route_agent,
        {
            "general":  "general",
            "research": "research",
            "coding":   "coding",
            "travel":   "travel",
            "budget":   "budget",
            "critic":   "critic",
        },
    )

    #  after each agent → dispatch more or go to critic 
    for node_name in ["general", "research", "coding", "travel", "budget"]:
        graph.add_conditional_edges(
            node_name,
            route_after_agent,
            {
                "dispatch": "dispatch",
                "critic":   "critic",
            },
        )

    # critic → retry OR formatter 
    #   critic
    #     ├── needs_regeneration=True  AND retries remain → retry
    #     └── output OK  OR  retries exhausted → formatter
    #
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retry":     "retry",
            "formatter": "formatter",
        },
    )

    graph.add_edge("retry",     "dispatch")

    graph.add_edge("formatter", END)

    return graph.compile()