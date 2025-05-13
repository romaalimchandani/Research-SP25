import pandas as pd

# Recreate the CRA cosine similarity results
#examples
cra_results = [ {
        "problem": "cottage + Swiss + cake",
        "solution": "cheese",
        "llm_answer": "cheese",
        "human_accuracy": 0.78,
        "llm_correct": True,
        "avg_cos_sim": 0.412
    },
    {
        "problem": "light + birthday + stick",
        "solution": "candle",
        "llm_answer": "fire",
        "human_accuracy": 0.66,
        "llm_correct": False,
        "avg_cos_sim": 0.260
    },
    {
        "problem": "spoon + cloth + card",
        "solution": "table",
        "llm_answer": "table",
        "human_accuracy": 0.52,
        "llm_correct": True,
        "avg_cos_sim": 0.295
    },
    {
        "problem": "water + mine + deep",
        "solution": "well",
        "llm_answer": "well",
        "human_accuracy": 0.71,
        "llm_correct": True,
        "avg_cos_sim": 0.372
    }
]

df = pd.DataFrame(cra_results)

# Save to file
csv_path = "/mnt/data/cra_cosine_similarity_results.csv"
df.to_csv(csv_path, index=False)
csv_path
