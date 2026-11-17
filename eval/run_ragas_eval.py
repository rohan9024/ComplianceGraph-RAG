import os
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from langchain_groq import ChatGroq
from app.config import settings
from app.agent.graph import compliance_graph_app

def evaluate_kaggle_split(num_samples: int = 5):
    csv_path = Path(settings.DATA_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    sample_df = df.dropna(subset=["question", "answer", "context"]).head(num_samples)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"\nEvaluating LangGraph pipeline across {len(sample_df)} Kaggle test cases on Groq...")

    for idx, row in sample_df.iterrows():
        q = str(row["question"]).strip()
        gt = str(row["answer"]).strip()

        state = {
            "query": q,
            "anonymized_query": "",
            "pii_vault": {},
            "category": None,
            "retrieved_docs": [],
            "graded_docs": [],
            "generation": None,
            "is_grounded": True,
            "rewrite_count": 0,
            "final_output": None
        }
        res = compliance_graph_app.invoke(state)

        questions.append(q)
        answers.append(res.get("final_output", ""))
        contexts.append([d["text"] for d in res.get("graded_docs", [])])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    eval_llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=0.0,
        api_key=settings.GROQ_API_KEY
    )
    
    print("\nRunning Ragas scoring (Faithfulness & Context Precision)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
        llm=eval_llm
    )

    results_df = scores.to_pandas()
    print("\n================ RAGAS BENCHMARK RESULTS ================")
    print(results_df[["faithfulness", "context_precision"]].mean())
    results_df.to_csv("ragas_kaggle_benchmark_results.csv", index=False)
    print("\n✓ Full audit metrics saved to 'ragas_kaggle_benchmark_results.csv'!")

if __name__ == "__main__":
    evaluate_kaggle_split(num_samples=5)