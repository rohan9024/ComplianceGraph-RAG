from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()
DATA_DIR = Path("data")
TARGET_CSV = DATA_DIR / "Financial-QA-10k.csv"
KAGGLE_DATASET = "yousefsaeedian/financial-qa-10k"

def download_via_kaggle_api() -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_CSV.exists():
        console.print(f"[bold green]Found existing dataset: {TARGET_CSV}[/bold green]")
        return True

    console.print(f"[yellow]Downloading Kaggle dataset: {KAGGLE_DATASET}[/yellow]")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
        if not TARGET_CSV.exists():
            candidates = list(DATA_DIR.glob("*.csv"))
            if candidates:
                candidates[0].replace(TARGET_CSV)
        if TARGET_CSV.exists():
            console.print("[bold green]Dataset downloaded and extracted.[/bold green]")
            return True
        raise FileNotFoundError(f"{TARGET_CSV} was not found after extraction.")
    except Exception as exc:
        console.print(f"[bold red]Kaggle download failed: {exc}[/bold red]")
        console.print(
            "Download manually from https://www.kaggle.com/datasets/yousefsaeedian/financial-qa-10k "
            f"and place Financial-QA-10k.csv in {DATA_DIR}/."
        )
        return False

def generate_offline_fallback() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_data = [
        {
            "question": "What area did NVIDIA initially focus on before expanding into other markets?",
            "answer": "NVIDIA initially focused on PC graphics.",
            "context": "Since our original focus on PC graphics, we have expanded into various markets including AI computing, data center hardware, and autonomous vehicle platforms.",
            "ticker": "NVDA",
            "filing": "2023_10K",
        },
        {
            "question": "What was Microsoft's cloud revenue performance in fiscal year 2023?",
            "answer": "Microsoft Cloud annual revenue surpassed $110 billion, up 27% in constant currency.",
            "context": "Microsoft Cloud revenue exceeded $110 billion for the fiscal year 2023, representing 27% growth in constant currency driven by enterprise Azure migrations and Office 365 commercial seats.",
            "ticker": "MSFT",
            "filing": "2023_10K",
        },
        {
            "question": "What is Apple's primary liquidity reserve and debt position reported in the filing?",
            "answer": "Apple reported total cash, cash equivalents, and marketable securities of $162.1 billion with term debt of $111.1 billion.",
            "context": "As of September 30, 2023, the Company had total cash, cash equivalents and marketable securities of $162.1 billion, and total term debt of $111.1 billion across commercial paper and fixed-rate notes.",
            "ticker": "AAPL",
            "filing": "2023_10K",
        },
        {
            "question": "What are Tesla's principal capital expenditure priorities according to their annual filing?",
            "answer": "Manufacturing facility expansion, next-generation vehicle architecture, and expanding supercharging infrastructure.",
            "context": "Capital expenditures in 2023 were $8.90 billion, focused primarily on manufacturing capacity expansions at Gigafactory Texas and Berlin, next-generation vehicle architecture development, and Supercharger network scaling.",
            "ticker": "TSLA",
            "filing": "2023_10K",
        },
    ]
    pd.DataFrame(sample_data).to_csv(TARGET_CSV, index=False)
    console.print(f"[bold green]Created fallback seed dataset: {TARGET_CSV}[/bold green]")

if __name__ == "__main__":
    if not download_via_kaggle_api() and not TARGET_CSV.exists():
        generate_offline_fallback()
