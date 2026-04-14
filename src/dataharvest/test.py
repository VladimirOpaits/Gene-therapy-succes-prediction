import sys
from pathlib import Path

try:
    from .fdaparser import FDA_PCh_Parser
    from .aifilter import ComponentExtractor
    from .params import OPENAI_API_KEY, LLM_PROVIDER
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from fdaparser import FDA_PCh_Parser
    from aifilter import ComponentExtractor
    from params import OPENAI_API_KEY, LLM_PROVIDER

def test_fda_parser():
    print("Testing FDA Parser...")
    parser = FDA_PCh_Parser()
    df = parser.fetch_training_data_df(limit=5)
    print(f"✓ Fetched {len(df)} trials")
    print(f"  Columns: {list(df.columns)}")

    print("\n--- Testing Drug Extraction & Enrichment ---")
    df_enriched = parser.extract_drug_names(df, use_llm=False)
    print(f"Extracted drugs sample:")
    for idx, (nct, drugs) in enumerate(zip(df_enriched['nct_id'].head(3), df_enriched['extracted_drugs'].head(3))):
        print(f"  {nct}: {drugs}")

    print("\nAdding drug info (PubChem/Wikipedia)...")
    df_with_info = parser.add_drug_info(df_enriched, use_extracted=True)
    for idx, row in df_with_info.head(2).iterrows():
        print(f"\n  Trial: {row['title'][:60]}...")
        if row['drug_info']:
            for drug in row['drug_info']:
                print(f"    • {drug['name']} ({drug['source']}): {drug['description'][:80]}...")
        else:
            print(f"    • No drug info found")

    return df_with_info

def test_component_extractor(df):
    if not OPENAI_API_KEY:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping ComponentExtractor test.")
        print("   Create .env file with OPENAI_API_KEY to enable LLM tests.")
        return None

    print(f"\nTesting ComponentExtractor (Provider: {LLM_PROVIDER})...")
    extractor = ComponentExtractor()

    sample_texts = [
        "A Phase 2 trial of AAV-based gene therapy for spinal muscular atrophy",
        "CRISPR-Cas9 treatment for sickle cell disease",
        df.iloc[0]["title"] if len(df) > 0 else "Unknown trial"
    ]

    print("\nExtracting components from sample texts:")
    for text in sample_texts:
        components = extractor.extract_components(text)
        print(f"  Text: {text[:60]}...")
        print(f"  Components: {components}\n")

    print("Enriching full dataframe...")
    df_enriched = extractor.enrich_dataframe(df, text_column="title")
    print(f"✓ Enriched {len(df_enriched)} rows with extracted_components")

    return df_enriched

def main():
    print("=== Clinical Trial Data Pipeline Tests ===\n")

    df = test_fda_parser()

    df_enriched = test_component_extractor(df)

    if df_enriched is not None:
        print("\n--- Component Extraction Results ---")
        for _, row in df_enriched.head(3).iterrows():
            print(f"\nTrial: {row['title'][:60]}...")
            print(f"  Extracted components: {row['extracted_components']}")
            if 'drug_info' in df_enriched.columns and row['drug_info']:
                print(f"  Drug info: {len(row['drug_info'])} drugs found")

if __name__ == "__main__":
    main()