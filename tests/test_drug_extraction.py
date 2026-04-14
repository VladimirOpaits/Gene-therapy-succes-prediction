"""
Unit tests for drug extraction and enrichment pipeline.
"""

import pytest
import pandas as pd
from src.dataharvest.fdaparser import FDA_PCh_Parser


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing drug extraction."""
    return pd.DataFrame({
        'nct_id': ['NCT001', 'NCT002', 'NCT003'],
        'title': [
            'Phase I trial of AAV gene therapy',
            'CRISPR-Cas9 treatment study',
            'Placebo controlled trial'
        ],
        'drugs': [
            'AAV6, Hemophilia B',
            'CRISPR-Cas9',
            'N/A'
        ],
        'status': ['COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING'],
        'sponsor': ['Hospital A', 'University B', 'Pharma C'],
        'sponsor_class': ['INDUSTRY', 'NIH', 'INDUSTRY'],
        'phases': ['PHASE2', 'PHASE1', 'PHASE3'],
        'results_date': ['2023-01-15', None, None]
    })


class TestDrugNameExtraction:
    """Test drug name extraction from trial data."""

    def test_extract_drug_names_fallback(self, sample_df):
        """Test extraction using fallback (non-LLM) method."""
        parser = FDA_PCh_Parser()
        df = parser.extract_drug_names(sample_df, use_llm=False)

        assert 'extracted_drugs' in df.columns
        assert len(df) == 3

        assert df.loc[0, 'extracted_drugs'] == ['AAV6', 'Hemophilia B']
        assert df.loc[1, 'extracted_drugs'] == ['CRISPR-Cas9']
        assert df.loc[2, 'extracted_drugs'] == []

    def test_parse_drugs_fallback(self, sample_df):
        """Test fallback drug parsing from raw drugs column."""
        parser = FDA_PCh_Parser()
        df = parser._parse_drugs_fallback(sample_df)

        assert 'extracted_drugs' in df.columns
        assert df.loc[0, 'extracted_drugs'] == ['AAV6', 'Hemophilia B']
        assert df.loc[1, 'extracted_drugs'] == ['CRISPR-Cas9']
        assert df.loc[2, 'extracted_drugs'] == []


class TestDrugSearch:
    """Test drug information search functionality."""

    def test_search_pubchem(self):
        """Test PubChem search for known drugs."""
        parser = FDA_PCh_Parser()

        result = parser._search_pubchem('aspirin')
        assert result is not None
        assert result['source'] == 'pubchem'
        assert result['name'] == 'aspirin'
        assert 'description' in result

    def test_search_pubchem_not_found(self):
        """Test PubChem search returns None for unknown drugs."""
        parser = FDA_PCh_Parser()
        result = parser._search_pubchem('xyzunknowndrug12345')
        assert result is None

    def test_search_wikipedia(self):
        """Test Wikipedia search for drugs."""
        parser = FDA_PCh_Parser()

        result = parser._search_wikipedia('Aspirin')
        if result:
            assert result['source'] == 'wikipedia'
            assert result['name'] == 'Aspirin'
            assert 'description' in result

    def test_search_wikipedia_empty(self):
        """Test Wikipedia search with empty input."""
        parser = FDA_PCh_Parser()
        result = parser._search_wikipedia('')
        assert result is None

    def test_search_drug_info_fallback(self):
        """Test search_drug_info with PubChem → Wikipedia fallback."""
        parser = FDA_PCh_Parser()

        result = parser.search_drug_info('aspirin')
        assert result is not None
        assert result['name'] == 'aspirin'
        assert result['source'] in ['pubchem', 'wikipedia']

    def test_search_drug_info_empty(self):
        """Test search_drug_info with empty input."""
        parser = FDA_PCh_Parser()
        result = parser.search_drug_info('')
        assert result is None


class TestDrugEnrichment:
    """Test drug information enrichment pipeline."""

    def test_add_drug_info_structure(self, sample_df):
        """Test that add_drug_info creates correct column structure."""
        parser = FDA_PCh_Parser()
        df = parser.extract_drug_names(sample_df, use_llm=False)
        df_enriched = parser.add_drug_info(df, use_extracted=True)

        assert 'drug_info' in df_enriched.columns
        assert isinstance(df_enriched.loc[0, 'drug_info'], list)

    def test_add_drug_info_empty_drugs(self, sample_df):
        """Test add_drug_info with trials without drugs."""
        parser = FDA_PCh_Parser()
        df = parser.extract_drug_names(sample_df, use_llm=False)
        df_enriched = parser.add_drug_info(df, use_extracted=True)

        empty_drug_row = df_enriched[df_enriched['nct_id'] == 'NCT003'].iloc[0]
        assert empty_drug_row['drug_info'] == []

    def test_add_drug_info_multiple_drugs(self):
        """Test add_drug_info processes multiple drugs per trial."""
        parser = FDA_PCh_Parser()

        test_df = pd.DataFrame({
            'nct_id': ['NCT001'],
            'title': ['Multi-drug trial'],
            'drugs': ['aspirin, ibuprofen'],
            'extracted_drugs': [['aspirin', 'ibuprofen']]
        })

        df_enriched = parser.add_drug_info(test_df, use_extracted=True)
        drug_count = len(df_enriched.iloc[0]['drug_info'])

        assert drug_count >= 1


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.slow
    def test_end_to_end_pipeline(self, sample_df):
        """Test complete extraction → enrichment pipeline."""
        parser = FDA_PCh_Parser()

        df = parser.extract_drug_names(sample_df, use_llm=False)
        df_enriched = parser.add_drug_info(df, use_extracted=True)

        assert 'extracted_drugs' in df_enriched.columns
        assert 'drug_info' in df_enriched.columns
        assert len(df_enriched) == len(sample_df)

        for _, row in df_enriched.iterrows():
            assert isinstance(row['drug_info'], list)
            for drug_info in row['drug_info']:
                assert 'name' in drug_info
                assert 'source' in drug_info
                assert 'description' in drug_info
                assert drug_info['source'] in ['pubchem', 'wikipedia']
