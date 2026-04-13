from fdaparser import FDA_PCh_Parser

parser = FDA_PCh_Parser()
df = parser.fetch_training_data_df(limit=20)
df = parser.add_drug_summaries(df)
df.to_csv("enriched_data.csv")