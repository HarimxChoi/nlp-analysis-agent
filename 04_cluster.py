import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional, List, Dict, Any
import html  # for HTML entity processing

# ==============================================================================
# 1. Config & Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- paths and global variables ---
BASE_PATH = r'.'
INPUT_FILE = os.path.join(BASE_PATH, 'data', 'bid_list_classification.xlsx')
SHEET_NAME = 'Sheet1'

OUTPUT_DIR = os.path.join(BASE_PATH, 'data', 'clustered_data')
MODEL_DIR = r'./trained_model/best_model_fold_1'

# Font for visualization
FONT_PATH = r'C:\Windows\Fonts\malgun.ttf'
font_prop = fm.FontProperties(fname=FONT_PATH, size=12)

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Outputs will be saved to '{OUTPUT_DIR}'.")


# ==============================================================================
# 2. Data preprocessing and loading
# ==============================================================================

def clean_html_entities(text: str) -> str:
    """Convert HTML entity codes (e.g., () to actual characters."""
    if pd.isna(text):
        return ""
    return html.unescape(str(text))

def load_and_preprocess_data(filepath: str, sheet_name: str, text_column: str) -> Optional[pd.DataFrame]:
    """Load Excel file and clean the specified text column."""
    logger.info(f"Input file loading start: {filepath}")
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"File loaded. Total {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Excel load error: {e}", exc_info=True)
        return None

    if text_column not in df.columns:
        logger.error(f"Required text column '{text_column}' not found.")
        return None

    # --- core preprocessing ---
    logger.info(f"Cleaning HTML entities in '{text_column}' column...")
    df[f'{text_column}_cleaned'] = df[text_column].apply(clean_html_entities)
    logger.info("Cleaning done.")

    # Sample before/after
    logger.info("--- Text cleaning example (Before -> After) ---")
    sample_df = df.head(5)
    for index, row in sample_df.iterrows():
        logger.info(f"Before: {row[text_column]}")
        logger.info(f"After:  {row[f'{text_column}_cleaned']}")
        logger.info("-" * 20)

    return df

# ==============================================================================
# 3. AI model feature extractor
# ==============================================================================
class SemanticVectorizer:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model(model_dir)
        logger.info(f"SemanticVectorizer ready on {self.device}.")

    def load_model(self, model_dir: str):
        """Load model and tokenizer."""
        logger.info("Loading semantic vector extractor model...")
        try:
            # Only the base RoBERTa is needed (excluding classification layer).
            base_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=2, output_hidden_states=True)
            self.model = PeftModel.from_pretrained(base_model, model_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model.eval()
            logger.info("Semantic vector extractor model loaded.")
        except Exception as e:
            logger.error(f"[FATAL] Model load failed: {e}", exc_info=True)
            sys.exit(1)

    def texts_to_vectors(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Take text list as input, return list of [CLS] token embedding vectors."""
        all_cls_embeddings = []
        logger.info(f"Extracting semantic vectors for {len(texts)} texts...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting semantic vectors"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract [CLS] token vector (first token) from the last hidden layer
            last_hidden_states = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_states[:, 0, :].cpu().numpy()
            all_cls_embeddings.append(cls_embeddings)

        logger.info("Semantic vector extraction done.")
        return np.vstack(all_cls_embeddings)

# ==============================================================================
# 4. Clustering and visualization pipeline
# ==============================================================================
def run_clustering_pipeline(df: pd.DataFrame, text_column_cleaned: str, vectorizer: SemanticVectorizer):
    # --- Step 1: extract semantic vectors ---
    texts = df[text_column_cleaned].fillna('').tolist()
    embeddings = vectorizer.texts_to_vectors(texts)

    # --- Step 2: dim reduction (for visualization) ---
    logger.info("UMAP dim reduction start (may take a while)...")
    reducer = UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    df['x'] = embedding_2d[:, 0]
    df['y'] = embedding_2d[:, 1]
    logger.info("UMAP dim reduction done.")

    # --- Step 3: HDBSCAN clustering ---
    logger.info("HDBSCAN clustering start...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=15,  # tune per data
                                metric='euclidean', cluster_selection_method='eom')
    df['cluster_id'] = clusterer.fit_predict(embedding_2d)
    num_clusters = df['cluster_id'].nunique()
    num_outliers = (df['cluster_id'] == -1).sum()
    logger.info(f"HDBSCAN clustering done. Found {num_clusters-1} clusters and {num_outliers} outliers.")

    # --- Step 4: visualization and save ---
    plt.figure(figsize=(20, 15))

    unique_labels = sorted(list(set(df['cluster_id'])))
    num_clusters = len(unique_labels)

    # Create colormap object
    # get_cmap is deprecated since Matplotlib 3.7; colormaps[] is preferred.
    try:
        colors = plt.colormaps['hsv'](np.linspace(0, 1, num_clusters))
    except AttributeError:
        colors = plt.cm.get_cmap('hsv', num_clusters)

    # Iterate labels and map colors
    for i, k in enumerate(unique_labels):
        if k == -1:
            col = (0.5, 0.5, 0.5, 0.1)  # noise: translucent gray
            label_text = 'Noise'
        else:
            col = colors[i]
            label_text = f'Cluster {k}'

        class_member_mask = (df['cluster_id'] == k)
        xy = df[class_member_mask][['x', 'y']]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6 if k != -1 else 3, label=label_text)

    plt.title('notice_name semantic clustering visualization (UMAP)', fontproperties=font_prop, size=20)
    plt.xlabel("UMAP 1", fontproperties=font_prop)
    plt.ylabel("UMAP 2", fontproperties=font_prop)

    visualization_path = os.path.join(OUTPUT_DIR, 'semantic_clustering_visualization.png')
    plt.savefig(visualization_path, dpi=300)
    logger.info(f"Clustering visualization image saved: {visualization_path}")
    plt.close()

    # --- Step 5: save result dataframe ---
    output_path = os.path.join(OUTPUT_DIR, 'clustered_bidding_data.parquet')
    df.to_parquet(output_path, index=False)
    logger.info(f"Clustering result data saved: {output_path}")

    return df

# ==============================================================================
# 5. Main
# ==============================================================================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Semantic clustering pipeline start")
    logger.info("=" * 50)

    # 1. Load data and preprocess
    text_column_to_clean = 'service_title'
    main_df = load_and_preprocess_data(INPUT_FILE, SHEET_NAME, text_column_to_clean)

    if main_df is None:
        sys.exit(1)

    # 2. Init vectorizer
    vectorizer = SemanticVectorizer(model_dir=MODEL_DIR)

    # 3. Run clustering pipeline using the cleaned 'service_title' column
    clustered_df = run_clustering_pipeline(main_df, f'{text_column_to_clean}_cleaned', vectorizer)

    # 4. Verify results
    logger.info("\n--- Per-cluster sample data ---")
    for cluster_id in sorted(clustered_df['cluster_id'].unique()):
        if cluster_id == -1:
            continue
        logger.info(f"\n[ Cluster {cluster_id} ]")
        sample_names = clustered_df[clustered_df['cluster_id'] == cluster_id][f'{text_column_to_clean}_cleaned'].head(5).tolist()
        for name in sample_names:
            logger.info(f"- {name}")

    logger.info("\n" + "=" * 50)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 50)
