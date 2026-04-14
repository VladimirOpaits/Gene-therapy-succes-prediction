#!/usr/bin/env python
"""
Demo: Enhanced Drug Extraction Pipeline with PubChem + Wikipedia Fallback

This script demonstrates the improved drug extraction capabilities:
1. Extract drug names from trial titles using LLM
2. Normalize and deduplicate drug names
3. Search for drug info in PubChem first, fall back to Wikipedia
4. Return structured drug information
"""

from src.dataharvest.fdaparser import FDA_PCh_Parser

def main():
    print("=" * 70)
    print("DEMO: Enhanced Drug Extraction Pipeline")
    print("=" * 70)

    parser = FDA_PCh_Parser()

    print("\n[1/4] Fetching clinical trials...")
    df = parser.fetch_training_data_df(query="gene therapy", limit=10)
    print(f"✓ Fetched {len(df)} trials\n")

    print("[2/4] Extracting drug names from trial data...")
    df = parser.extract_drug_names(df, use_llm=False)

    print("Sample extracted drugs:")
    for i, (nct, title, drugs) in enumerate(zip(
        df['nct_id'].head(5),
        df['title'].head(5),
        df['extracted_drugs'].head(5)
    ), 1):
        print(f"\n  {i}. {nct}")
        print(f"     Title: {title[:65]}...")
        print(f"     Drugs: {drugs}")

    print("\n[3/4] Enriching with drug information (PubChem → Wikipedia)...")
    df_enriched = parser.add_drug_info(df, use_extracted=True)

    print("\n[4/4] Results:")
    print("-" * 70)

    successful_trials = df_enriched[df_enriched['drug_info'].apply(len) > 0]
    print(f"\nTrials with drug information: {len(successful_trials)}/{len(df_enriched)}")
    print(f"Total drugs found: {sum(df_enriched['drug_info'].apply(len))}")

    print("\nDetailed Results:")
    for _, row in df_enriched.iterrows():
        if row['drug_info']:
            print(f"\n  • {row['nct_id']}: {row['title'][:60]}...")
            for drug in row['drug_info']:
                print(f"    ├─ {drug['name']} ({drug['source'].upper()})")
                print(f"    │  {drug['description'][:70]}...")
            print()

    print("-" * 70)
    print("\n✓ Demo complete!")
    print("\nNext steps:")
    print("  - Use this enriched data for ML model training")
    print("  - Filter by sponsor_class and phases for better feature engineering")
    print("  - Combine with market data (Polymarket) for edge detection")

if __name__ == "__main__":
    main()
