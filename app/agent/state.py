from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict):
    query: str
    anonymized_query: str
    pii_vault: Dict[str, str]
    category: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]
    generation: Optional[str]
    is_grounded: bool
    rewrite_count: int
    final_output: Optional[str]
