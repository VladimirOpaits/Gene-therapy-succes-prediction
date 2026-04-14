import requests
import pandas as pd
from typing import List, Dict, Optional

class FDA_PCh_Parser:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    
    def _get_raw_data(self, params):
        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            print(f"DEBUG: API Error {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json().get('studies', [])

    def fetch_training_data(self, query="gene therapy", limit=10):
        params = {
            "query.term": query,
            "filter.overallStatus": "COMPLETED|TERMINATED|WITHDRAWN",
            "pageSize": limit,
            "fields": "NCTId,OfficialTitle,OverallStatus,LeadSponsorName,LeadSponsorClass,Phase,InterventionName,ResultsFirstPostDate"
        }
        return self._get_raw_data(params)

    def fetch_oracle_leads(self, query="gene therapy", limit=50):
        params = {
            "query.term": query,
            "filter.overallStatus": "RECRUITING|ACTIVE_NOT_RECRUITING",
            "pageSize": limit,
            "fields": "NCTId,OfficialTitle,OverallStatus,LeadSponsorName,LeadSponsorClass,Phase,InterventionName,PrimaryCompletionDate"
        }
        return self._get_raw_data(params)
    
    def _flatten_study(self, study_json):
        ps = study_json.get('protocolSection', {})
        ident = ps.get('identificationModule', {})
        status_mod = ps.get('statusModule', {})
        sponsor_mod = ps.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsor_mod.get('leadSponsor', {})
        design_mod = ps.get('designModule', {})
        phases_list = design_mod.get('phases', [])
        arms_mod = ps.get('armsInterventionsModule', {})
        interventions = arms_mod.get('interventions', [])
        drug_names = [i.get('name') for i in interventions if i.get('name')]

        return {
            "nct_id": ident.get('nctId'),
            "title": ident.get('officialTitle'),
            "status": status_mod.get('overallStatus'),
            "drugs": ", ".join(drug_names) if drug_names else "N/A",
            "sponsor": lead_sponsor.get('name'),
            "sponsor_class": lead_sponsor.get('class'), 
            "phases": ", ".join(phases_list) if phases_list else "N/A",
            "results_date": status_mod.get('resultsFirstPostDateStruct', {}).get('date'),
        }

    def fetch_training_data_df(self, query="gene therapy", limit=100):
        raw_data = self.fetch_training_data(query, limit)
        flattened_data = [self._flatten_study(s) for s in raw_data]
        return pd.DataFrame(flattened_data)

    def fetch_oracle_leads_df(self, query="gene therapy", limit=50):
        raw_data = self.fetch_oracle_leads(query, limit)
        flattened_data = [self._flatten_study(s) for s in raw_data]
        return pd.DataFrame(flattened_data)
    
    def extract_drug_names(self, df: pd.DataFrame, use_llm: bool = False) -> pd.DataFrame:
        if use_llm:
            try:
                from .aifilter import ComponentExtractor
                extractor = ComponentExtractor()
                df = df.copy()
                df['extracted_drugs'] = df.apply(
                    lambda row: extractor.extract_components(
                        f"{row['title']} {row['drugs']}"
                    ) if pd.notna(row['drugs']) else [],
                    axis=1
                )
            except ImportError:
                print("⚠️ aifilter not available. Using fallback drug parsing...")
                df = self._parse_drugs_fallback(df)
        else:
            df = self._parse_drugs_fallback(df)

        return df

    def _parse_drugs_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['extracted_drugs'] = df['drugs'].apply(
            lambda x: [d.strip() for d in str(x).split(',') if d.strip() and d.strip() != "N/A"]
        )
        return df

    def _search_pubchem(self, drug_name: str) -> Optional[Dict]:
        if not drug_name or drug_name == "N/A":
            return None

        url = f"{self.PUBCHEM_URL}/{drug_name}/description/JSON"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                info = r.json().get('InformationList', {}).get('Information', [])
                for item in info:
                    if 'Description' in item:
                        return {
                            "name": drug_name,
                            "source": "pubchem",
                            "description": item['Description']
                        }
                return None
        except requests.RequestException:
            return None
        return None

    def _search_wikipedia(self, drug_name: str) -> Optional[Dict]:
        if not drug_name or drug_name == "N/A":
            return None

        try:
            import wikipedia
        except ImportError:
            return None

        try:
            result = wikipedia.summary(drug_name, sentences=2, auto_suggest=False)
            return {
                "name": drug_name,
                "source": "wikipedia",
                "description": result
            }
        except wikipedia.exceptions.DisambiguationError:
            return None
        except wikipedia.exceptions.PageError:
            return None
        except Exception:
            return None

    def search_drug_info(self, drug_name: str) -> Optional[Dict]:
        if not drug_name or drug_name.strip() == "":
            return None

        drug_name = drug_name.strip()

        pubchem_result = self._search_pubchem(drug_name)
        if pubchem_result:
            return pubchem_result

        wikipedia_result = self._search_wikipedia(drug_name)
        if wikipedia_result:
            return wikipedia_result

        return None

    def get_drug_summary(self, drug_name):
        if not drug_name or drug_name == "N/A":
            return "N/A"

        name = drug_name.split(',')[0].strip()
        url = f"{self.PUBCHEM_URL}/{name}/description/JSON"

        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                info = r.json().get('InformationList', {}).get('Information', [])
                for item in info:
                    if 'Description' in item:
                        return item['Description']
                return "No description found"
        except:
            return "Search failed"
        return "N/A"
    
    def add_drug_info(self, df: pd.DataFrame, use_extracted: bool = True) -> pd.DataFrame:
        print(f"Enriching {len(df)} rows with drug info...")
        df = df.copy()

        if use_extracted and 'extracted_drugs' not in df.columns:
            df = self.extract_drug_names(df, use_llm=False)

        drug_column = 'extracted_drugs' if use_extracted and 'extracted_drugs' in df.columns else 'drugs'

        def enrich_drugs(drug_list):
            if isinstance(drug_list, str):
                drug_list = [d.strip() for d in drug_list.split(',') if d.strip() and d.strip() != "N/A"]

            results = []
            for drug in drug_list:
                if drug and drug != "N/A":
                    info = self.search_drug_info(drug)
                    if info:
                        results.append(info)
            return results if results else []

        df['drug_info'] = df[drug_column].apply(enrich_drugs)

        successful = df[df['drug_info'].apply(len) > 0]
        print(f"✓ Successfully enriched: {len(successful)}/{len(df)} trials")
        print(f"  Total drugs found: {sum(df['drug_info'].apply(len))}")

        return df

    def add_drug_summaries(self, df):
        print(f"Enriching {len(df)} rows...")
        df['drug_summary'] = df['drugs'].apply(self.get_drug_summary)

        failed = df[df['drug_summary'].isin(['N/A', 'No description found', 'Search failed'])]
        print(f"Success: {len(df) - len(failed)}")
        print(f"Failed: {len(failed)}")

        if not failed.empty:
            print("Failed drugs samples:", failed['drugs'].unique()[:5])

        return df