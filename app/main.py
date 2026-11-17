from rich.console import Console
from rich.panel import Panel
from app.ingestion.kaggle_ingest import ingest_kaggle_dataset
from app.agent.graph import compliance_graph_app

console = Console()

def run_agent_interactive(user_query: str):
    console.print(Panel(
        f"[bold yellow]Client Query:[/bold yellow] {user_query}",
        title="Query Ingestion",
    ))
    state = {
        "query": user_query,
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
    result = compliance_graph_app.invoke(state)
    console.print(Panel(
        f"[cyan]Masked Query:[/cyan] {result.get('anonymized_query')}\n"
        f"[cyan]Surrogate Vault Keys:[/cyan] {list(result.get('pii_vault', {}).keys())}\n"
        f"[cyan]Graded 10-K Chunks:[/cyan] {len(result.get('graded_docs', []))}\n"
        f"[cyan]Grounded Verification:[/cyan] {result.get('is_grounded')}\n\n"
        f"[green]Final Response:[/green]\n{result.get('final_output')}",
        title="LangGraph Execution Engine",
    ))

if __name__ == "__main__":
    ingest_kaggle_dataset(max_contexts=100)
    test_prompt = (
        "Compliance Auditor Klaus Schmidt (IBAN: DE89370400440532013000, "
        "Tax-ID: 41 852 963 147) asks: What area did NVIDIA initially focus on "
        "before expanding into other markets?"
    )
    run_agent_interactive(test_prompt)
