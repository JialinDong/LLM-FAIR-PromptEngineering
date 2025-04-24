
import openai
import os
import pandas as pd
import time

# ✅ Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"
AZURE_OPENAI_API_KEY = ""  # Replace with actual API key
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION = "2024-02-01"

# ✅ Configure Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ✅ Function to query Azure OpenAI
def evaluate_fair_principles(dataset_name, website_link):
    """Call Azure OpenAI GPT-4o model to evaluate FAIR principles based on a defined scoring rubric."""
    prompt = {
        "model": AZURE_DEPLOYMENT_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in dataset evaluation and FAIR principles assessment."
            },
            {
                "role": "user",
                "content": f"""
Evaluate the FAIR (Findable, Accessible, Interoperable, Reusable) principles for the dataset: **{dataset_name}** available at {website_link}.

Use the following scoring system:

### 1. Findable (Max: 17)
- **Identifiers**:
  - 8 points: DOI, PURL, ARK, Handle
  - 3 points: Web address (URL)
  - 1 point: Local identifier
  - 0 points: No identifier
- **Identifier in metadata**:
  - 1 point: Yes
  - 0 points: No
- **Metadata description**:
  - 4 points: Comprehensive, machine-readable
  - 3 points: Comprehensive, non-standard
  - 2 points: Brief title and description
  - 0 points: No metadata
- **Repository inclusion**:
  - 4 points: Discoverable in multiple repositories
  - 2 points: Generalist/domain-specific repository
  - 0 points: Not described in any repository

**Final Findable Score (%) = (Sum of scores / 17) * 100**

---

### 2. Accessible (Max: 10)
- **Data accessibility**:
  - 5 points: Publicly accessible
  - 5 points: Fully accessible with stated conditions
  - 4 points: De-identified subset accessible
  - 3 points: Embargoed access
  - 2 points: Unspecified conditional access
  - 1 point: Metadata only
  - 0 points: No access
- **Online availability**:
  - 4 points: Standard API (e.g., OGC)
  - 3 points: Non-standard API
  - 2 points: Direct file download
  - 1 point: By arrangement
  - 0 points: No access
- **Metadata persistence**:
  - 1 point: Yes
  - 0 points: No

**Final Accessible Score (%) = (Sum of scores / 10) * 100**

---

### 3. Interoperable (Max: 8)
- **Data format**:
  - 2 points: Structured, open, machine-readable
  - 1 point: Structured, non-machine-readable
  - 0 points: Proprietary format
- **Vocabularies/Ontologies**:
  - 3 points: Standardized, open, resolvable
  - 2 points: Standardized but not resolvable
  - 1 point: No standard applied
  - 0 points: No description
- **Metadata linking**:
  - 3 points: Linked data format (e.g., RDF)
  - 2 points: URI links to related metadata
  - 0 points: No links

**Final Interoperable Score (%) = (Sum of scores / 8) * 100**

---

### 4. Reusable (Max: 7)
- **License type**:
  - 4 points: Machine-readable license (e.g., Creative Commons)
  - 3 points: Standard text-based license
  - 3 points: Non-standard machine-readable with conditions
  - 2 points: Non-standard text-based
  - 0 points: No license
- **Provenance information**:
  - 3 points: Fully recorded, machine-readable
  - 2 points: Fully recorded, text format
  - 1 point: Partially recorded
  - 0 points: No provenance

**Final Reusable Score (%) = (Sum of scores / 7) * 100**

---

### 5. Final FAIR Score:
**Final FAIR Score (%) = (Findable Score + Accessible Score + Interoperable Score + Reusable Score) / 4**

---

**Return ONLY a structured table in the following format:**

| Dataset Name | F-Score (X/17) | A-Score (X/10) | I-Score (X/8) | R-Score (X/7) | Total FAIR Score (%) |

Each value should be replaced with the evaluated score. **Do not return any explanation or extra text.**
"""
            }
        ],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 512
    }

    response = client.chat.completions.create(**prompt)
    return response.choices[0].message.content

# ✅ Load dataset information from CSV
# input_csv = "PFAS_GW_DataFair_try2.csv"  # Update with your actual file path
# output_csv = "PFAS_GW_fair_scores_try4.csv"  # Output file
input_csv = "SelectData.csv"  # PFAS_DataFair_OTHER
output_csv = "SelectData_LLM1_fair_scores_4o.csv"  # Output file

df = pd.read_csv(input_csv)

# ✅ Ensure column names are correctly referenced
if "Website Link" not in df.columns:
    raise ValueError("Column 'Website Link' not found in CSV file. Check the column names.")

# ✅ Store results
results = []

for index, row in df.iterrows():
    dataset_name = row["Dataset Name"]  # Ensure this column exists
    website_link = row["Website Link"]

    print(f"Evaluating dataset: {dataset_name} ({website_link})...")

    try:
        fair_evaluation = evaluate_fair_principles(dataset_name, website_link)
        results.append({
            "Dataset Name": dataset_name,
            "Website Link": website_link,
            "FAIR Evaluation": fair_evaluation
        })
    except Exception as e:
        print(f"Error processing {dataset_name} ({website_link}): {e}")

    # ✅ To avoid hitting rate limits
    time.sleep(3)

# ✅ Convert to DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print("FAIR evaluation completed. Results saved to:", output_csv)

