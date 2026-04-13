import requests
import pandas as pd

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
    
    def add_drug_summaries(self, df):
        print(f"Enriching {len(df)} rows...")
        df['drug_summary'] = df['drugs'].apply(self.get_drug_summary)
        
        failed = df[df['drug_summary'].isin(['N/A', 'No description found', 'Search failed'])]
        print(f"Success: {len(df) - len(failed)}")
        print(f"Failed: {len(failed)}")
        
        if not failed.empty:
            print("Failed drugs samples:", failed['drugs'].unique()[:5])
            
        return df