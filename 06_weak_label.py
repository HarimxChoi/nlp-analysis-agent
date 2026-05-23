import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
import os
import logging
import re
import html
import json
from peft import PeftModel

# --- 0. Logger and base config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Paths and config values ---
DATA_DIR = r'./data'
BID_LIST_FILE = os.path.join(DATA_DIR, 'bid_list.xlsx')
KNOWLEDGE_BASE_FILE = os.path.join(DATA_DIR, 'final_clustered_label.xlsx')
OUTPUT_FILE = os.path.join(DATA_DIR, 'model_only_classification_result_final.xlsx')

CONFIDENCE_THRESHOLD = 0.9

# --- 2. Asset loading and pre-processing ---
logger.info("=" * 20 + " System init and asset loading start " + "=" * 20)

try:
    MODEL_NAME = "jhgan/ko-sbert-sts"
    embedding_model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Semantic similarity model loaded: {MODEL_NAME}")

    logger.info(f"Embedding model loaded: {MODEL_NAME}")

    bid_df = pd.read_excel(BID_LIST_FILE)
    logger.info(f"Analysis target data loaded: {len(bid_df)} notices")

    knowledge_base_df = pd.read_excel(KNOWLEDGE_BASE_FILE, sheet_name='Sheet1')

    # Rename actual column 'label' to standard 'Final_Label'
    if 'label' in knowledge_base_df.columns:
        knowledge_base_df.rename(columns={'label': 'Final_Label'}, inplace=True)

    if 'Final_Label' not in knowledge_base_df.columns:
        raise KeyError("'label' or 'Final_Label' column not found in knowledge base file.")

    logger.info("Master knowledge base loaded.")

except Exception as e:
    logger.error(f"[FATAL] Asset load failed: {e}")
    sys.exit(1)

# --- 3. Core function definitions ---

def get_sbert_embedding(texts, model):
    """Compute embeddings of a text list using SBERT in a single batch."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def classify_with_sbert_similarity(title, knowledge_base_embeddings, knowledge_base_labels, model, threshold):
    """Classify a notice_name by SBERT embedding similarity."""
    if not title or not isinstance(title, str):
        return 'input error', 0.0, 'input error'

    try:
        title_embedding = get_sbert_embedding([title], model)

        # cosine similarity
        similarities = cosine_similarity(title_embedding, knowledge_base_embeddings)[0]

        best_index = np.argmax(similarities)
        best_label = knowledge_base_labels[best_index]
        best_score = similarities[best_index]

        status = "auto_confirmed" if best_score >= threshold else "manual_review_needed"
        return best_label, best_score, status
    except Exception as e:
        logger.error(f"'{title[:30]}...' classification error: {e}")
        return 'classification error', 0.0, 'error'

def clean_text(text):
    """HTML entity decoding + remove unwanted special characters."""
    if not isinstance(text, str):
        return ""
    # 1. HTML entity decoding (e.g., ) -> ), & -> &)
    text = html.unescape(text)
    # 2. Collapse multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sbert_embedding(texts, model, device, batch_size=64):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, device=device)

def get_embedding_ft(texts: list, model, tokenizer, device, batch_size=32):
    """Compute embeddings of a text list using the fine-tuned RoBERTa.
    Uses mean pooling (not the [CLS] token)."""
    model.to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Finetuned model embedding"):
        batch = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # Move all inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get last hidden state via base_model
            outputs = model.base_model(**inputs)

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        # Exclude padding tokens from compute using attention mask
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

def calculate_max_sim_matrix(title_embeddings, kb_embeddings_by_label_list, knowledge_base_df):
    """Compute the Max-Sim score matrix between given notice embeddings and per-label keyword embedding lists."""
    # Final score matrix (n_notices x n_labels)
    max_similarity_matrix = np.zeros((len(title_embeddings), len(knowledge_base_df)))

    # Iterate per label
    for j, keyword_embs_list in enumerate(tqdm(kb_embeddings_by_label_list, desc="Max-Sim matrix compute")):

        if len(keyword_embs_list) == 0:
            continue

        # Similarity matrix between all notices and the current label's keywords
        # Shape: (n_notices, n_keywords_in_label_j)
        sim_matrix_per_label = cosine_similarity(title_embeddings, keyword_embs_list)

        # For each notice, take the max similarity with the current label's keywords
        # Shape: (n_notices,)
        max_sim_per_title = np.max(sim_matrix_per_label, axis=1)

        # Save the computed max similarity in the j-th column of the final matrix
        max_similarity_matrix[:, j] = max_sim_per_title

    return max_similarity_matrix

# --- 4. Main run block ---

def main():
    # --- 1. Init config and model/data loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load both models.
    try:
        # Model 1: SBERT
        sbert_model_name = "jhgan/ko-sbert-sts"
        sbert_model = SentenceTransformer(sbert_model_name)
        sbert_model.to(device)
        logger.info(f"SBERT model loaded: {sbert_model_name}")

        # Model 2: Fine-tuned RoBERTa
        FINETUNED_MODEL_DIR = r'./trained_model/best_model_fold_1'
        ft_base_model_name = "klue/roberta-large"
        ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
        base_model = AutoModel.from_pretrained(ft_base_model_name)
        finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)
        finetuned_model.to(device)
        finetuned_model.eval()
        logger.info(f"Fine-tuned RoBERTa loaded: {FINETUNED_MODEL_DIR}")

    except Exception as e:
        logger.error(f"[FATAL] Model load failed: {e}")
        return

    # --- 2. Text preprocessing ---
    if 'service_title' not in bid_df.columns:
        logger.error(f"'service_title' column not found in input file.")
        return
    bid_df['service_title_cleaned'] = bid_df['service_title'].apply(clean_text)
    logger.info("Notice_name text preprocessing done.")

    # --- 3. Two-channel embedding of knowledge base ---
    logger.info("=" * 10 + " Knowledge base pre-processing (2-channel embedding) " + "=" * 10)
    try:
        if 'keywords' not in knowledge_base_df.columns:
            raise KeyError("'keywords' column not found in knowledge base file.")

        knowledge_base_df['Keywords_List'] = knowledge_base_df['keywords'].astype(str).apply(
            lambda x: [kw.strip() for kw in x.split(',') if kw.strip()]
        )
        all_unique_keywords = list(set(kw for sublist in knowledge_base_df['Keywords_List'] for kw in sublist))

        # Channel 1: SBERT keyword embedding
        logger.info(f"[Channel 1: SBERT] embedding {len(all_unique_keywords)} keywords...")
        sbert_keyword_embeddings = get_sbert_embedding(all_unique_keywords, sbert_model, device)
        sbert_kw_to_emb = dict(zip(all_unique_keywords, sbert_keyword_embeddings))
        knowledge_base_df['SBERT_Embeddings_List'] = knowledge_base_df['Keywords_List'].apply(lambda kws: np.array([sbert_kw_to_emb[kw] for kw in kws if kw in sbert_kw_to_emb]))

        # Channel 2: Fine-tuned model keyword embedding
        logger.info(f"[Channel 2: Finetuned] embedding {len(all_unique_keywords)} keywords...")
        ft_keyword_embeddings = get_embedding_ft(all_unique_keywords, finetuned_model, ft_tokenizer, device)
        ft_kw_to_emb = dict(zip(all_unique_keywords, ft_keyword_embeddings))
        knowledge_base_df['FT_Embeddings_List'] = knowledge_base_df['Keywords_List'].apply(lambda kws: np.array([ft_kw_to_emb[kw] for kw in kws if kw in ft_kw_to_emb]))

        knowledge_base_labels = knowledge_base_df['Final_Label'].values
        logger.info("Knowledge base 2-channel embedding done.")

    except Exception as e:
        logger.error(f"[FATAL] Knowledge base processing failed: {e}")
        return

    # --- 4. Hybrid classification pipeline ---
    logger.info("=" * 10 + " 2-channel hybrid classification start " + "=" * 10)

    # Init result columns
    bid_df['Predicted_Label'] = None
    bid_df['Confidence_Score'] = 0.0
    bid_df['Status'] = ''

    # 4-1. Hard rules
    logger.info("Step 1: applying hard rules...")

    # 'supervision' rule
    supervision_mask = bid_df['service_title_cleaned'].str.contains('supervision|construction_management|cm', case=False, na=False)
    bid_df.loc[supervision_mask, 'Predicted_Label'] = 'construction_management'
    bid_df.loc[supervision_mask, 'Status'] = 'rule:supervision'

    # 'safety_inspection' rule
    s_diag_mask = bid_df['service_title_cleaned'].str.contains('safety_inspection|safety_check|performance_assessment', case=False, na=False) & bid_df['Predicted_Label'].isnull()
    bid_df.loc[s_diag_mask, 'Predicted_Label'] = 'structural_safety_inspection'
    bid_df.loc[s_diag_mask & bid_df['service_title_cleaned'].str.contains('transportation', case=False), 'Predicted_Label'] = 'traffic_safety_inspection'
    bid_df.loc[s_diag_mask & bid_df['service_title_cleaned'].str.contains('harbor|pier|lock_facility', case=False), 'Predicted_Label'] = 'harbor_safety_inspection'
    bid_df.loc[s_diag_mask & bid_df['service_title_cleaned'].str.contains('dam|river|floodgate|reservoir|intake|pump_station|valve|water_supply', case=False), 'Predicted_Label'] = 'water_resources_safety_inspection'
    bid_df.loc[s_diag_mask & bid_df['service_title_cleaned'].str.contains('railway|railway_track|high_speed', case=False), 'Predicted_Label'] = 'railway_track_safety_inspection'
    bid_df.loc[s_diag_mask, 'Status'] = 'rule:safety_inspection'

    # 'impact_assessment' rule
    i_assess_mask = bid_df['service_title_cleaned'].str.contains('impact_assessment', case=False, na=False) & bid_df['Predicted_Label'].isnull()
    bid_df.loc[i_assess_mask, 'Predicted_Label'] = 'environmental_impact_assessment'
    bid_df.loc[i_assess_mask & bid_df['service_title_cleaned'].str.contains('disaster|protection', case=False), 'Predicted_Label'] = 'disaster_impact_assessment'
    bid_df.loc[i_assess_mask & bid_df['service_title_cleaned'].str.contains('transportation', case=False), 'Predicted_Label'] = 'traffic_impact_assessment'
    bid_df.loc[i_assess_mask & bid_df['service_title_cleaned'].str.contains('power', case=False), 'Predicted_Label'] = 'power_impact_assessment'
    bid_df.loc[i_assess_mask, 'Status'] = 'rule:impact_assessment'

    # 'technical_inspection' rule
    t_diag_mask = bid_df['service_title_cleaned'].str.contains('technical_inspection', case=False, na=False) & bid_df['Predicted_Label'].isnull()
    bid_df.loc[t_diag_mask, 'Predicted_Label'] = 'technical_inspection'
    bid_df.loc[t_diag_mask, 'Status'] = 'rule:technical_inspection'

    # 'soil' rule
    soil_mask = bid_df['service_title_cleaned'].str.contains('soil_pollution|soil_environment|soil_cleansing', case=False, na=False) & bid_df['Predicted_Label'].isnull()
    bid_df.loc[soil_mask, 'Predicted_Label'] = 'soil_survey'
    bid_df.loc[soil_mask, 'Status'] = 'rule:soil'

    rule_applied_mask = bid_df['Predicted_Label'].notnull()
    bid_df.loc[rule_applied_mask, 'Confidence_Score'] = 1.0
    logger.info(f"Hard rules applied: {rule_applied_mask.sum()} records processed.")

    # 4-2. SBERT Max-Sim classification for remaining data
    remaining_mask = bid_df['Predicted_Label'].isnull()
    if remaining_mask.any():
        logger.info(f"2-channel ensemble classification start ({remaining_mask.sum()} targets)")
        titles_to_process = bid_df.loc[remaining_mask, 'service_title_cleaned'].tolist()

        # Channel 1: SBERT scores
        logger.info("[Channel 1] SBERT-based score compute...")
        sbert_title_embeddings = get_sbert_embedding(titles_to_process, sbert_model, device)
        sbert_scores = calculate_max_sim_matrix(sbert_title_embeddings, knowledge_base_df['SBERT_Embeddings_List'].tolist(), knowledge_base_df)

        # Channel 2: Fine-tuned model scores
        logger.info("[Channel 2] Fine-tuned model score compute...")
        ft_title_embeddings = get_embedding_ft(titles_to_process, finetuned_model, ft_tokenizer, device)
        ft_scores = calculate_max_sim_matrix(ft_title_embeddings, knowledge_base_df['FT_Embeddings_List'].tolist(), knowledge_base_df)

        # Weighted-mean ensemble
        W_SBERT = 0.9
        W_FINETUNED = 0.1
        ensemble_scores = (W_SBERT * sbert_scores) + (W_FINETUNED * ft_scores)

        best_indices = np.argmax(ensemble_scores, axis=1)
        best_scores = np.max(ensemble_scores, axis=1)

        # Assign results
        bid_df.loc[remaining_mask, 'Predicted_Label'] = knowledge_base_labels[best_indices]
        bid_df.loc[remaining_mask, 'Final_Ensemble_Score'] = best_scores
        bid_df.loc[remaining_mask, 'Status'] = np.where(best_scores >= CONFIDENCE_THRESHOLD, 'auto_confirmed (ensemble)', 'manual_review_needed (AI)')

        # Per-channel scores for analysis
        bid_df.loc[remaining_mask, 'SBERT_Score'] = np.max(sbert_scores, axis=1)
        bid_df.loc[remaining_mask, 'Finetuned_Score'] = np.max(ft_scores, axis=1)
        logger.info("2-channel ensemble classification done.")
    else:
        logger.info("All records handled by rules; skipping 2-channel ensemble classification.")

    # --- 5. Final result processing, summary, and validation ---
    logger.info("=" * 10 + " Final result, summary, and validation " + "=" * 10)

    # Map auxiliary info
    div_col_name = 'div'
    dept_col_name = 'dep'
    label_info_df = knowledge_base_df.set_index('Final_Label')
    bid_df['Predicted_Div'] = bid_df['Predicted_Label'].map(label_info_df.get(div_col_name)).fillna('N/A')
    bid_df['Predicted_Dept'] = bid_df['Predicted_Label'].map(label_info_df.get(dept_col_name)).fillna('N/A')
    logger.info("Auxiliary info mapping done.")

    # Department-match validation
    actual_dept_col_name = 'participating_dept'
    if actual_dept_col_name not in bid_df.columns:
        logger.warning(f"'{actual_dept_col_name}' column not present; skipping match validation.")
        bid_df['Dept_Match_Status'] = 'validation_skipped'
    else:
        try:
            # Assume dept.json file is in the same folder as the code
            with open('dept.json', 'r', encoding='utf-8') as f:
                raw_to_standard_map = json.load(f)
            bid_df['Standard_Dept_Actual'] = bid_df[actual_dept_col_name].map(raw_to_standard_map).fillna(bid_df[actual_dept_col_name])
        except FileNotFoundError:
            logger.warning("'dept.json' not found; skipping dept_name standardization.")
            bid_df['Standard_Dept_Actual'] = bid_df[actual_dept_col_name]

        def check_dept_match(row):
            predicted_depts = str(row['Predicted_Dept']).split(', ')
            actual_dept = str(row['Standard_Dept_Actual'])
            if actual_dept in predicted_depts:
                return 'match'
            if pd.isnull(row['Predicted_Label']) or 'error' in str(row['Predicted_Label']) or pd.isnull(actual_dept) or actual_dept == 'nan':
                return 'undetermined'
            return 'mismatch'
        bid_df['Dept_Match_Status'] = bid_df.apply(check_dept_match, axis=1)
        logger.info("Department match validation done.")

    # Final summary output
    label_counts = bid_df['Predicted_Label'].value_counts().reset_index()
    label_counts.columns = ['Final_Label', 'Count']
    print("\n--- [Summary] classifications per label ---")
    with pd.option_context('display.max_rows', None):
        print(label_counts)

    if 'Dept_Match_Status' in bid_df.columns:
        match_summary = bid_df['Dept_Match_Status'].value_counts()
        print("\n--- [Summary] dept match status ---")
        print(match_summary)

    # Save result
    try:
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            bid_df.to_excel(writer, sheet_name='Classification_Result', index=False)
            label_counts.to_excel(writer, sheet_name='Label_Summary', index=False)
        logger.info(f"All tasks done. Results saved to: {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Result file save failed: {e}")

# --- Run script ---
if __name__ == '__main__':
    main()
