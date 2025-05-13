import pandas as pd

def load_cra_data_from_excel(path="Chat GPT VS Human.xlsx"):
    df = pd.read_excel(path, sheet_name="Sheet1")
    cra_data = []

    for _, row in df.iterrows():
        try:
            triplet = str(row["Unnamed: 1"]).strip().split("/")
            if len(triplet) != 3:
                continue

            solution = str(row["correct answers"]).strip().lower()
            llm_answer = str(row["GPT Guess"]).strip().lower()
            human_accuracy = float(row["Human % w/ 15 sec"]) / 100.0 if pd.notna(row["Human % w/ 15 sec"]) else None

            cra_data.append({
                "words": triplet,
                "solution": solution,
                "human_accuracy": human_accuracy,
                "llm_answer": llm_answer
            })
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

    return cra_data
