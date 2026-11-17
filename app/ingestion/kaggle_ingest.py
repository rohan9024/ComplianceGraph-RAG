import pandas as pd
from pathlib import Path
from rich.console import Console
from app.config import settings
from app.retrieval.vector_store import QdrantRetriever

console = Console()

def ingest_kaggle_dataset(max_contexts: int = 200):
    csv_path = Path(settings.DATA_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {csv_path}. Run 'python -m scripts.download_kaggle_data' first."
        )

    console.print(f"[cyan]Reading financial dataset from {csv_path}...[/cyan]")
    dataframe = pd.read_csv(csv_path)
    dataframe.columns = [column.strip().lower() for column in dataframe.columns]
    required_columns = {"context"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    unique_contexts = dataframe.drop_duplicates(subset=["context"]).head(max_contexts)
    documents = []
    for index, row in unique_contexts.iterrows():
        context_text = str(row["context"]).strip()
        ticker = str(row.get("ticker", "SEC")).upper()
        filing = str(row.get("filing", "10-K"))
        if len(context_text) < 40:
            continue
        documents.append({
            "id": f"{ticker}_{filing}_chunk_{index}",
            "text": context_text,
            "ticker": ticker,
            "filing": filing,
            "source": f"{ticker} {filing} SEC Report",
        })

    console.print(f"[yellow]Indexing {len(documents)} contexts into Qdrant...[/yellow]")
    retriever = QdrantRetriever()
    retriever.index_documents(documents)
    console.print(
        f"[green]Ingestion complete: {len(documents)} chunks in '{settings.QDRANT_COLLECTION}'.[/green]"
    )

if __name__ == "__main__":
    ingest_kaggle_dataset()
