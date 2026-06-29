import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Path configuration
CORPUS_PATH = "01_korpus_raw"
RESULTS_PATH = "03_ergebnisse"
os.makedirs(RESULTS_PATH, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_stories_list(folder_path):
    """Reads all 25 story files locally and returns titles and contents."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    return [(filename.replace(".txt", ""), open(os.path.join(folder_path, filename), "r", encoding="utf-8").read()) for filename in files]

def get_embedding(text, model="text-embedding-3-small"):
    """Fetches the semantic vector embedding from the OpenAI API."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    # Truncate text if it exceeds token boundaries for a safe embedding call
    payload = {"input": text[:8000], "model": model}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return None

if __name__ == "__main__":
    print("Starting Step 3: Embedding-Based Clustering Analysis...")
    
    if not OPENAI_API_KEY:
        print("CRITICAL ERROR: Please set the OPENAI_API_KEY in your environment.")
    else:
        stories = get_stories_list(CORPUS_PATH)
        print(f"Generating embeddings for {len(stories)} stories...")
        
        titles = []
        vectors = []
        
        for title, content in stories:
            print(f"  -> Embedding file: {title}")
            vec = get_embedding(content)
            if vec:
                titles.append(title)
                vectors.append(vec)
        
        # Convert list of vectors into a NumPy array for mathematical operations
        X = np.array(vectors)
        
        # Reduce dimensions from 1536 to 2 using Principal Component Analysis (PCA)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Initialize the scatter plot
        plt.figure(figsize=(12, 10))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], color='darkblue', edgecolors='black', s=100, alpha=0.7)
        
        # Annotate each data point with its respective story number
        for i, title in enumerate(titles):
            # Format title mapping for terminal print
            clean_title = title.split("_", 1)[-1].replace("_", " ").title() if "_" in title else title
            plt.annotate(
                f"{i+1:02d}", # Displays the two-digit story index on the plot
                (X_2d[i, 0], X_2d[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=9,
                weight='bold'
            )
            print(f"Index {i+1:02d}: {clean_title}")
            
        # Configure English plot labels and styling
        plt.title("Thematic Clustering of Otto Anthes' Stories (PCA Projection)", fontsize=14, pad=15)
        plt.xlabel("Principal Component 1 (Variance)", fontsize=11)
        plt.ylabel("Principal Component 2 (Variance)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save the finalized visualization matrix
        plot_path = os.path.join(RESULTS_PATH, "thematic_clustering_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSUCCESS: Cluster visualization saved to: {plot_path}")
        print("Note: Stories plotted in close proximity share highly similar semantic themes and spatial motifs.")