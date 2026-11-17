from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes import (
    pii_mask_node, route_node, retrieve_node, grade_documents_node,
    rewrite_query_node, generate_node, verify_citations_node,
    deanonymize_node, abstain_node,
)

def check_doc_quality(state: AgentState) -> str:
    if not state.get("graded_docs") and state.get("rewrite_count", 0) < 2:
        return "rewrite"
    return "generate"

def check_grounding(state: AgentState) -> str:
    return "finish" if state.get("is_grounded") else "abstain"

workflow = StateGraph(AgentState)
workflow.add_node("pii_mask", pii_mask_node)
workflow.add_node("route", route_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify_citations", verify_citations_node)
workflow.add_node("deanonymize", deanonymize_node)
workflow.add_node("abstain", abstain_node)

workflow.add_edge(START, "pii_mask")
workflow.add_edge("pii_mask", "route")
workflow.add_edge("route", "retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    check_doc_quality,
    {"rewrite": "rewrite_query", "generate": "generate"},
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", "verify_citations")
workflow.add_conditional_edges(
    "verify_citations",
    check_grounding,
    {"abstain": "abstain", "finish": "deanonymize"},
)
workflow.add_edge("abstain", END)
workflow.add_edge("deanonymize", END)

compliance_graph_app = workflow.compile()
