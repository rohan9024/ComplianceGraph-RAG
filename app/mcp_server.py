import asyncio
from mcp.server.fastmcp import FastMCP
from app.agent.graph import compliance_graph_app
from app.ingestion.anonymizer import GDPRAnonymizer

mcp = FastMCP("ComplianceGraph-RAG")
anonymizer = GDPRAnonymizer()

@mcp.tool()
async def query_kaggle_sec_knowledge_base(query: str) -> str:
    """Query the SEC 10-K knowledge base with PII masking and citation verification."""
    initial_state = {
        "query": query,
        "anonymized_query": "",
        "pii_vault": {},
        "category": None,
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": None,
        "is_grounded": True,
        "rewrite_count": 0,
        "final_output": None,
    }
    result = await asyncio.to_thread(compliance_graph_app.invoke, initial_state)
    return result.get("final_output", "No response.")

@mcp.tool()
def test_gdpr_pii_sanitization(text: str) -> str:
    """Detect and mask sensitive EU/German PII such as IBANs and tax IDs."""
    masked, vault = anonymizer.mask_pii(text, language="de")
    return f"Masked Text: {masked}\nVault Mapping: {vault}"

if __name__ == "__main__":
    mcp.run()
