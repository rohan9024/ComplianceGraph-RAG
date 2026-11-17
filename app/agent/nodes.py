import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.config import settings
from app.ingestion.anonymizer import GDPRAnonymizer
from app.retrieval.vector_store import QdrantRetriever
from app.agent.state import AgentState

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0.0,
    api_key=settings.GROQ_API_KEY
)
anonymizer = GDPRAnonymizer()
retriever = QdrantRetriever()

def clean_json_response(raw_text: str) -> dict:
    """Extracts JSON even if wrapped in markdown backticks or explanations."""
    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}

def pii_mask_node(state: AgentState) -> dict:
    raw_query = state["query"]
    masked_query, vault = anonymizer.mask_pii(raw_query)
    return {
        "anonymized_query": masked_query,
        "pii_vault": vault,
        "rewrite_count": state.get("rewrite_count", 0)
    }

def route_node(state: AgentState) -> dict:
    prompt = f"""Classify the intent of this query for a corporate filing retrieval system into 'financial', 'regulatory', or 'general'.
Query: {state['anonymized_query']}
Output strict JSON: {{"category": "financial"}}"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    parsed = clean_json_response(res.content)
    cat = parsed.get("category", "financial")
    return {"category": cat}

def retrieve_node(state: AgentState) -> dict:
    query = state["anonymized_query"]
    # Strip auditor metadata or greetings to isolate the core question for vector search
    extract_prompt = f"""Extract ONLY the core financial question from this query. Remove greetings, names, and ID tags.
Query: {query}
Return ONLY the clean question text."""
    
    clean_q = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip().replace('"', '')
    hits = retriever.search(clean_q, top_k=4)
    return {"retrieved_docs": hits, "anonymized_query": clean_q}

def grade_documents_node(state: AgentState) -> dict:
    query = state["anonymized_query"]
    docs = state.get("retrieved_docs", [])
    passed = []

    for doc in docs:
        prompt = f"""Does this 10-K excerpt contain information relevant to answering: '{query}'?
Excerpt: {doc['text']}
Output JSON: {{"relevant": true}} or {{"relevant": false}}"""
        
        res = llm.invoke([HumanMessage(content=prompt)])
        parsed = clean_json_response(res.content)
        if parsed.get("relevant", True):
            passed.append(doc)

    return {"graded_docs": passed}

def rewrite_query_node(state: AgentState) -> dict:
    prompt = f"""Rewrite this question as a short, natural language question for semantic search over SEC 10-K documents.
Do NOT use Boolean operators (AND, OR, NOT) and do NOT use quotation marks.
Question: {state['anonymized_query']}
Return ONLY the plain text question."""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    clean_rewritten = res.content.strip().replace('"', '').replace('AND', '').replace('OR', '')
    return {
        "anonymized_query": clean_rewritten,
        "rewrite_count": state["rewrite_count"] + 1
    }

def generate_node(state: AgentState) -> dict:
    docs = state.get("graded_docs", [])
    if not docs:
        return {
            "generation": "No verified filing evidence available to answer this inquiry.",
            "is_grounded": True
        }

    context = "\n\n".join([f"[{d['id']}]: {d['text']}" for d in docs])
    prompt = f"""You are an audit-grade financial assistant. Answer the question using ONLY the provided excerpts.
Every fact or number MUST cite its chunk ID, e.g. [{docs[0]['id']}].

Context:
{context}

Question: {state['anonymized_query']}"""

    res = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": res.content}

def verify_citations_node(state: AgentState) -> dict:
    docs = state.get("graded_docs", [])
    if not docs:
        return {"is_grounded": True}
        
    context = "\n".join([d["text"] for d in docs])
    prompt = f"""Verify if the claims in this response are factual according to the context.
Context: {context}
Response: {state['generation']}
Output JSON: {{"grounded": true}} or {{"grounded": false}}"""

    res = llm.invoke([HumanMessage(content=prompt)])
    parsed = clean_json_response(res.content)
    return {"is_grounded": parsed.get("grounded", True)}

def deanonymize_node(state: AgentState) -> dict:
    output = anonymizer.deanonymize(state["generation"], state["pii_vault"])
    return {"final_output": output}

def abstain_node(state: AgentState) -> dict:
    return {"final_output": "[AUDIT ALERT]: Response suppressed due to ungrounded claims or hallucination detection."}