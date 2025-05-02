from flask import Flask, render_template, jsonify
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)

def load_embeddings(file_path, max_words=200):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

def reduce_dimensions(embeddings):
    words = list(embeddings.keys())
    matrix = np.array([embeddings[word] for word in words])
    reduced = PCA(n_components=2).fit_transform(matrix)
    return [{"word": words[i], "x": float(reduced[i][0]), "y": float(reduced[i][1])}
            for i in range(len(words))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    embeddings = load_embeddings('glove_sample.txt')
    reduced = reduce_dimensions(embeddings)
    return jsonify(reduced)

if __name__ == '__main__':
    app.run(debug=True)
