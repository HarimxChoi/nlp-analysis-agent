import os
import glob
import pandas as pd
import torch
from datasets import Dataset
import html
import re

# --- path config ---
BASE_DIR = r'.'
# 'eligible' data path
PARTICIPATED_BIDS_PATH = os.path.join(BASE_DIR, 'biddingAnalysis', 'data', 'biddingResult')
# 'ineligible' data path
INELIGIBLE_BIDS_PATH = os.path.join(BASE_DIR, 'bidNLP', 'output', 'ineligible.xlsx')
# final output save path
OUTPUT_PATH = os.path.join(BASE_DIR, 'bidNLP', 'output')

os.makedirs(OUTPUT_PATH, exist_ok=True)


# --- function definitions ---

def read_participated_bids(path, company_name='COMPANY_NAME'):
    """Load all 'eligible' data for the given company."""
    print(f"Loading [bid participation] data for '{company_name}'...")
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filename, low_memory=False, encoding='euc-kr')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filename, low_memory=False, encoding='cp949')
                except UnicodeDecodeError:
                    print(f"Encoding error, skipped: {filename}")
                    continue
        if 'bid_company_name' in df.columns:
            df = df[df['bid_company_name'] == company_name]
            df_list.append(df)

    if not df_list:
        print("WARN: bid participation data file not found.")
        return pd.DataFrame(columns=['notice_name'])

    merged_df = pd.concat(df_list, ignore_index=True)
    if 'notice_name' in merged_df.columns:
        merged_df = merged_df[['notice_name']].drop_duplicates(subset=['notice_name'], keep='last').reset_index(drop=True)
        print(f"Loaded {len(merged_df)} unique [bid participation] notices.")
        return merged_df
    return pd.DataFrame(columns=['notice_name'])

def read_ineligible_bids(path):
    """Load ineligible data."""
    print("Loading ineligible data...")
    if not os.path.exists(path):
        print(f"WARN: 'ineligible' data file not found: {path}")
        return pd.DataFrame(columns=['notice_name'])
    try:
        # read 'ineligible' sheet explicitly
        df = pd.read_excel(path, sheet_name='ineligible')
        # use only 'notice_name' column
        if 'notice_name' in df.columns:
            df = df[['notice_name']]
            print(f"Loaded {len(df)} 'ineligible' records.")
            return df
        else:
            print("WARN: 'ineligible.xlsx' missing 'notice_name' column.")
            return pd.DataFrame(columns=['notice_name'])
    except Exception as e:
        print(f"'ineligible.xlsx' load error: {e}")
        return pd.DataFrame(columns=['notice_name'])

def preprocess_bid_title_strategy(text: str) -> str:
    """Comprehensive preprocessing for notice_name text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    parentheses_content = re.findall(r'[\(\[](.+?)[\)\]]', text)
    text = re.sub(r'[\(\[].+?[\)\]]', ' ', text)
    text = text.replace('/', ' ')
    text = re.sub(r"[:#´']", "", text)
    if parentheses_content:
        final_text = text + " " + " ".join(parentheses_content)
    else:
        final_text = text
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    return final_text

def main():
    """main function"""
    # 1. Load data from each source
    print("--- 1. data loading start ---")
    df_eligible = read_participated_bids(PARTICIPATED_BIDS_PATH)
    df_ineligible = read_ineligible_bids(INELIGIBLE_BIDS_PATH)

    if df_eligible.empty or df_ineligible.empty:
        print("\nInsufficient required data. Stopping.")
        return

    # 2. Sample to balance ratio 1:1
    print("\n--- 2. 1:1 ratio sampling start ---")
    num_ineligible = len(df_ineligible)
    num_eligible = len(df_eligible)

    # Sample eligible data to match ineligible count
    num_to_sample = min(num_ineligible, num_eligible)
    print(f"Counted: 'ineligible' = {num_ineligible}, 'eligible' = {num_eligible}.")
    print(f"Sampling {num_to_sample} records per class for 1:1 ratio.")

    df_ineligible_sampled = df_ineligible.sample(n=num_to_sample, random_state=42)
    df_eligible_sampled = df_eligible.sample(n=num_to_sample, random_state=42)

    # 3. Build dataframe and preprocess
    print("\n--- 3. Final dataset build and preprocessing start ---")
    # assign labels
    df_ineligible_sampled['label'] = 0
    df_eligible_sampled['label'] = 1

    # unify column names
    df_ineligible_sampled = df_ineligible_sampled.rename(columns={'notice_name': 'service_title'})
    df_eligible_sampled = df_eligible_sampled.rename(columns={'notice_name': 'service_title'})

    # combine data
    final_df = pd.concat([df_ineligible_sampled, df_eligible_sampled], ignore_index=True)

    # preprocess text
    final_df['service_title'] = final_df['service_title'].apply(preprocess_bid_title_strategy)

    # remove duplicates and empty rows
    final_df = final_df.drop_duplicates(subset=['service_title'], keep='last').reset_index(drop=True)
    final_df = final_df.dropna(subset=['service_title'])
    final_df = final_df[final_df['service_title'] != '']

    # shuffle dataset (prevent training bias)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Verify final results
    print("\n--- 4. Verify final results ---")
    print("\nFinal dataset label distribution:")
    print(final_df['label'].value_counts())
    print(final_df['label'].value_counts(normalize=True).sort_index())

    # Data is balanced 1:1, weighted loss not needed
    print("\nData is balanced, weighted loss not used.")

    # 5. Convert and save final dataset
    dataset = Dataset.from_pandas(final_df)

    output_file_path = os.path.join(OUTPUT_PATH, 'preprocessed_balanced_dataset.csv')
    final_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    print(f"\nFinal dataset ready. {len(dataset)} records.")
    print(f"Preprocessed data saved to: {output_file_path}")


if __name__ == '__main__':
    main()
