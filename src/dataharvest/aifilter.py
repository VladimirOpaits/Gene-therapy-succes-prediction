import pandas as pd
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from .params import OPENAI_API_KEY, OPENAI_MODEL

class ComponentExtractor:
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider

        if provider == "openai":
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=model or OPENAI_MODEL,
                temperature=0
            )
        else:
            raise NotImplementedError("Anthropic provider not yet implemented. Use openai for now.")

        self.parser = CommaSeparatedListOutputParser()

        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a biomedical expert. Extract all drug/therapeutic component names from the following clinical trial information.
Return only the component names as a comma-separated list (e.g., "Drug A, Drug B, Protein X").
If no clear components are mentioned, return "N/A".

Text:
{text}

Component names:""",
            output_parser=self.parser
        )

        self.chain = self.prompt | self.llm | self.parser

    def extract_components(self, text: str) -> list:
        if not text or text == "N/A":
            return []

        try:
            result = self.chain.invoke({"text": text})
            return [c.strip() for c in result if c.strip() and c.strip() != "N/A"]
        except Exception as e:
            print(f"Error extracting components from '{text[:50]}...': {e}")
            return []

    def enrich_dataframe(self, df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        print(f"Extracting components from {len(df)} rows...")
        df["extracted_components"] = df[text_column].apply(self.extract_components)
        print(f"✓ Extraction complete. Added {len(df)} rows with component data.")

        return df


def example_usage():
    from src.dataharvest.fdaparser import FDA_PCh_Parser

    parser = FDA_PCh_Parser()
    df = parser.fetch_training_data_df(query="gene therapy", limit=5)

    extractor = ComponentExtractor()
    df_enriched = extractor.enrich_dataframe(df, text_column="title")

    print("\nEnriched DataFrame:")
    print(df_enriched[["title", "extracted_components"]].head())

    return df_enriched


if __name__ == "__main__":
    example_usage()