import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Loading GloVe Vectors
def load_glove(glove_path="glove.6B.100d.txt"):
    print("Loading GloVe embeddings...")
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"Loaded {len(glove)} word vectors.")
    return glove

# CRA Data Example (Replace or expand as needed)
cra_data = [
    {"words": ["cottage", "Swiss", "cake"], "solution": "cheese", "human_accuracy": 0.78, "llm_answer": "cheese"},
    {"words": ["light", "birthday", "stick"], "solution": "candle", "human_accuracy": 0.66, "llm_answer": "fire"},
    {"words": ["spoon", "cloth", "card"], "solution": "table", "human_accuracy": 0.52, "llm_answer": "table"},
    {"words": ["water", "mine", "deep"], "solution": "well", "human_accuracy": 0.71, "llm_answer": "well"},
    # Add more if needed
]

# Cosine Similarity 
def cos_sim(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]

# Analysis
def analyze_cra(glove, cra_data):
    vectors, labels, types = [], [], []
    results = []

    for item in cra_data:
        cue_words = item["words"]
        cue_vecs = [glove.get(w.lower()) for w in cue_words if w.lower() in glove]
        solution_vec = glove.get(item["solution"].lower())
        llm_vec = glove.get(item["llm_answer"].lower())

        if len(cue_vecs) < 3 or solution_vec is None:
            print(f"Skipping: missing vectors for {item['words'] + [item['solution']]}")
            continue

        # Compute cosine similarities for each cue
        individual_sims = [cos_sim(v, solution_vec) for v in cue_vecs]
        avg_sim = np.mean(individual_sims)
        llm_correct = item["llm_answer"].lower() == item["solution"].lower()

        # Build results entry
        results.append({
            "problem": " + ".join(cue_words),
            "cue_1": cue_words[0],
            "cue_2": cue_words[1],
            "cue_3": cue_words[2],
            "solution": item["solution"],
            "llm_answer": item["llm_answer"],
            "human_accuracy": item["human_accuracy"],
            "llm_correct": llm_correct,
            "cos_sim_cue_1": individual_sims[0],
            "cos_sim_cue_2": individual_sims[1],
            "cos_sim_cue_3": individual_sims[2],
            "avg_cos_sim": avg_sim
        })

        # For PCA plot
        for word in cue_words:
            if word.lower() in glove:
                vectors.append(glove[word.lower()])
                labels.append(word)
                types.append("cue")

        if item["solution"].lower() in glove:
            vectors.append(glove[item["solution"].lower()])
            labels.append(item["solution"])
            types.append("solution")

    return results, np.array(vectors), labels, types

# Plotting semantic space
def plot_3d_semantic_space(vectors, labels, types):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vectors)

    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["word"] = labels
    df["type"] = types

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', text='word', color='type',
        title="Interactive 3D Semantic Space (GloVe + CRA)",
        opacity=0.85, width=800, height=600
    )
    fig.update_traces(marker=dict(size=6))
    fig.show()

# Main Script
if __name__ == "__main__":
    glove = load_glove("glove.6B.100d.txt")  # Use "glove_sample.txt" for testing smaller scale
    results, vecs, labels, types = analyze_cra(glove, cra_data)

    # Save results to CSV
    df = pd.DataFrame(results)
    print("\nCRA Analysis Results:")
    print(df)
    df.to_csv("cra_cosine_similarity_results.csv", index=False)
    print("Results saved to 'cra_cosine_similarity_results.csv'")

    # Optional: correlation between human accuracy and LLM correctness
    correlation = df["human_accuracy"].corr(df["llm_correct"].astype(float))
    print(f"\nCorrelation (LLM correctness vs human accuracy): {correlation:.2f}")

    # Plot 3D space
    plot_3d_semantic_space(vecs, labels, types)
