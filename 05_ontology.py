import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List

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
# Use the parquet file output from the previous script as input
INPUT_FILE = os.path.join(BASE_PATH, 'data', 'clustered_data', 'clustered_bidding_data.parquet')
OUTPUT_FILE = os.path.join(BASE_PATH, 'data', 'clustered_data', 'cluster_analysis_report.xlsx')

# text column to analyze (cleaned column produced by previous script)
TEXT_COLUMN = 'service_title_cleaned'
# cluster ID column name
CLUSTER_COLUMN = 'cluster_id'

# Stop words excluded in TF-IDF (add as needed)
CUSTOM_STOP_WORDS = ['service', 'construction', 'business', 'for', 'related', 'misc', 'and']

logger.info(f"input file: {INPUT_FILE}")
logger.info(f"output file: {OUTPUT_FILE}")

# ==============================================================================
# 2. Data loading and analysis functions
# ==============================================================================

def load_clustered_data(filepath: str) -> pd.DataFrame:
    """Load the parquet file containing clustering results."""
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}. Run the clustering script first.")
        sys.exit(1)

    logger.info(f"Loading clustering result file: {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Load done. {len(df)} rows, {df[CLUSTER_COLUMN].nunique()} unique cluster IDs.")
    return df

def get_top_tfidf_keywords(corpus: List[str], top_n: int = 20) -> List[str]:
    """Extract top TF-IDF keywords from the given text corpus."""
    if not corpus:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words=CUSTOM_STOP_WORDS,
            max_features=2000,  # max words to analyze
            token_pattern=r'(?u)\b\w\w+\b'  # extract words of 2+ characters
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Sum TF-IDF score per word
        sum_tfidf = tfidf_matrix.sum(axis=0)
        feature_names = vectorizer.get_feature_names_out()

        # Sort by score descending and extract top N
        tfidf_scores = zip(feature_names, np.asarray(sum_tfidf).ravel())
        sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

        return [word for word, score in sorted_tfidf_scores[:top_n]]
    except ValueError:
        # Too few texts or no content; TF-IDF cannot be computed
        return ["(keyword extraction failed)"]

def analyze_clusters(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Analyze each cluster's characteristics and return as dict."""
    logger.info("Per-cluster analysis start...")
    analysis_results = {}
    unique_clusters = sorted(df[CLUSTER_COLUMN].unique())

    for cluster_id in unique_clusters:
        cluster_df = df[df[CLUSTER_COLUMN] == cluster_id]

        # Extract text data for this cluster
        corpus = cluster_df[TEXT_COLUMN].dropna().tolist()

        # Extract top keywords
        keywords = get_top_tfidf_keywords(corpus, top_n=20)

        # Extract sample data (max 30)
        sample_size = min(len(cluster_df), 30)
        samples = cluster_df[TEXT_COLUMN].sample(n=sample_size, random_state=42).tolist()

        analysis_results[cluster_id] = {
            'size': len(cluster_df),
            'keywords': keywords,
            'samples': samples
        }

        # Log noise(-1) cluster vs. normal cluster separately
        cluster_type = "outlier(Noise)" if cluster_id == -1 else f"cluster {cluster_id}"
        logger.info(f" - {cluster_type} (size: {len(cluster_df)}) analysis done. representative keywords: {' '.join(keywords[:5])}")

    logger.info("Cluster analysis done.")
    return analysis_results

def save_analysis_to_excel(results: Dict[int, Dict[str, Any]], filepath: str):
    """Save analysis results to Excel. Each cluster is saved on a separate sheet."""
    logger.info(f"Saving analysis results to Excel: {filepath}")
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 1. Summary sheet
        summary_data = []
        for cid, data in results.items():
            cluster_name = "Noise" if cid == -1 else f"Cluster {cid}"
            summary_data.append({
                'Cluster ID': cid,
                'Name': cluster_name,
                'Size': data['size'],
                'Top 5 Keywords': ', '.join(data['keywords'][:5])
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 2. Per-cluster detail sheets
        for cid, data in results.items():
            sheet_name = f"Cluster_{cid}" if cid != -1 else "Noise"

            # Build dataframe of keywords and samples
            max_len = max(len(data['keywords']), len(data['samples']))
            detail_data = {
                'Top Keywords': pd.Series(data['keywords'], index=range(len(data['keywords']))),
                'Sample Texts': pd.Series(data['samples'], index=range(len(data['samples'])))
            }
            detail_df = pd.DataFrame(detail_data)
            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Excel file save done.")

# ==============================================================================
# 3. Main
# ==============================================================================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Cluster analysis and report generation pipeline start")
    logger.info("=" * 50)

    # 1. Load data
    clustered_df = load_clustered_data(INPUT_FILE)

    # 2. Analyze clusters
    analysis_results = analyze_clusters(clustered_df)

    # 3. Save results to Excel
    save_analysis_to_excel(analysis_results, OUTPUT_FILE)

    logger.info("\n" + "=" * 50)
    logger.info(f"Pipeline completed successfully. Verify '{OUTPUT_FILE}'.")
    logger.info("=" * 50)
