# import os
# from dotenv import load_dotenv
# # from google.cloud import bigquery
# from google import genai
# from google.oauth2 import service_account

# api_key = os.getenv("GEMINI_API_KEY")
# project = os.getenv("PROJECT_ID")

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-cred.json'
# creds = service_account.Credentials.from_service_account_file("service-cred.json")

# ai_client = genai.Client(api_key=api_key)


# # client = bigquery.Client(project=project, location="EU", credentials=creds)

# # def run_bigquery(query):
# #     try:
# #         query_job = client.query(query)
# #         return query_job.to_dataframe()
# #     except Exception as e:
# #         st.error(f"BigQuery Error: {e}")
# #         return None


# # def nl_to_sql_with_gemini(query):
# #     """Use Gemini to translate natural language queries to SQL for the EBI-MGnify dataset."""
# #     try:
# #         # gemini_model = genai.GenerativeModel('gemini-pro')
        
# #         prompt = f"""
# #         Translate the following natural language query into a BigQuery SQL query for the EBI-MGnify dataset.
        
# #         The main table is `bigquery-public-data.ebi_mgnify.protein`, which has these key columns:
# #         - mgyp (protein ID)
# #         - sequence (amino acid sequence)
# #         - pfam (Pfam domain information)
        
# #         For Pfam domain searches, use the SEARCH function like: SEARCH(pfam, 'PF00000')
# #         For keyword searches in descriptions, use SEARCH(description, 'keyword')
        
# #         Always include the protein ID, sequence, and relevant annotation fields in the results.
# #         Limit results to 1 unless otherwise specified.
        
# #         USER QUERY: {query}
        
# #         SQL QUERY:
# #         """
        
# #         # response = gemini_model.generate_content(prompt)
# #         response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
# #         sql = response.text.strip()
        
# #         sql = re.sub(r'```sql', '', sql)
# #         sql = re.sub(r'```', '', sql)
        
# #         return sql.strip()
# #     except Exception as e:
# #         st.error(f"Error generating SQL: {e}")
# #         return
