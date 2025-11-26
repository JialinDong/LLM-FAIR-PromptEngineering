#AZURE_DEPLOYMENT_NAME = "gpt-4o"
#AZURE_API_VERSION = "2024-02-01"

import openai
import os
import pandas as pd
import time
import re 

# ✅ Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"
AZURE_OPENAI_API_KEY = "xxx"  # Replace with actual API key
AZURE_DEPLOYMENT_NAME = "gpt-4o"  # gpt-4o , gpt-4o-mini , gpt-4-turbo , gpt-4 , gpt-3.5-turbo 
AZURE_API_VERSION = "2024-02-01"

# ✅ Configure Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ✅ Scoring rubric (rules)
SCORING_RULES = """
Evaluate the FAIR (Findable, Accessible, Interoperable, Reusable) principles using the scoring rubric:

### 1. Findable (Max: 17)
- Identifiers:
  - 8: DOI, PURL, ARK, Handle
  - 3: URL
  - 1: Local ID
  - 0: None
- Identifier in metadata: 1 or 0
- Metadata description:
  - 4: Comprehensive, machine-readable
  - 3: Comprehensive, non-standard
  - 2: Basic title/desc
  - 0: None
- Repository inclusion:
  - 4: Multiple repos
  - 2: General/domain-specific
  - 0: None

### 2. Accessible (Max: 10)
- Data access:
  - 5: Public or stated conditions
  - 4: De-identified subset
  - 3: Embargoed
  - 2: Unclear
  - 1: Metadata only
  - 0: No access
- Online availability:
  - 4: Standard API
  - 3: Non-standard API
  - 2: File download
  - 1: On request
  - 0: None
- Metadata persistence: 1 or 0

### 3. Interoperable (Max: 8)
- Format:
  - 2: Open machine-readable
  - 1: Structured non-machine-readable
  - 0: Proprietary
- Vocab/ontologies:
  - 3: Open & resolvable
  - 2: Standardized only
  - 1: No standard
  - 0: No description
- Metadata linking:
  - 3: Linked data (e.g., RDF)
  - 2: URI links
  - 0: None

### 4. Reusable (Max: 7)
- License:
  - 4: Machine-readable (e.g., CC)
  - 3: Standard text
  - 2: Non-standard
  - 0: No license
- Provenance:
  - 3: Machine-readable
  - 2: Full, text format
  - 1: Partial
  - 0: None
"""


# ✅ Few-shot + Chain-of-Thought Example

FEWSHOT_EXAMPLE = """
Example:

Dataset: "EPA UCMR3 PFAS Data"
Link: https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule#3

Step-by-step evaluation:
- Findable: URL (3), identifier in metadata (1), basic metadata (3), listed in EPA repo (2) → F-Score = 9/17
- Accessible: Public access (5), direct file download (2), persistent metadata (1) → A-Score = 8/10
- Interoperable: Open CSV format (2), uses standardized vocabularies (3), no linked metadata (2) → I-Score = 7/8
- Reusable: Machine-readable license missing (2), provenance described in text (2) → R-Score = 4/7

Final score:
| Dataset Name               | F-Score (9/17) | A-Score (8/10) | I-Score (7/8) | R-Score (4/7) |

---

Now evaluate the following dataset:
"""


# ✅ Chain-of-Thought + Few-shot Prompt
def evaluate_fair_principles(dataset_name, website_link):
    prompt = {
        "model": AZURE_DEPLOYMENT_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in dataset evaluation and FAIR principles assessment."
            },
            {
                "role": "user",
                "content": SCORING_RULES + "\n" + FEWSHOT_EXAMPLE + f"""
Dataset: "{dataset_name}"
Link: {website_link}

Think step-by-step using the rubric, and then return a markdown table like this:
| Dataset Name | F-Score (X/17) | A-Score (X/10) | I-Score (X/8) | R-Score (X/7) |
"""
            }
        ],
        "temperature": 0.2,   # T=0.2
        "top_p": 1,
        "max_tokens": 800
    }

    response = client.chat.completions.create(**prompt)
    return response.choices[0].message.content


# ✅ Score extractor
def extract_scores_from_markdown(markdown_str):
    pattern = r'\|\s*(.*?)\s*\|\s*F-Score \((\d+)/17\)\s*\|\s*A-Score \((\d+)/10\)\s*\|\s*I-Score \((\d+)/8\)\s*\|\s*R-Score \((\d+)/7\)\s*\|'
    match = re.search(pattern, markdown_str)
    if match:
        return {
            "Dataset Name (Parsed)": match.group(1).strip(),
            "F-Score": int(match.group(2)),
            "A-Score": int(match.group(3)),
            "I-Score": int(match.group(4)),
            "R-Score": int(match.group(5))
        }
    else:
        return {
            "Dataset Name (Parsed)": None,
            "F-Score": None,
            "A-Score": None,
            "I-Score": None,
            "R-Score": None
        }

# ✅ Score validator
def check_fair_score_consistency(row):
    checks = {
        "F-Score Valid": (row["F-Score"] is not None) and (0 <= row["F-Score"] <= 17),
        "A-Score Valid": (row["A-Score"] is not None) and (0 <= row["A-Score"] <= 10),
        "I-Score Valid": (row["I-Score"] is not None) and (0 <= row["I-Score"] <= 8),
        "R-Score Valid": (row["R-Score"] is not None) and (0 <= row["R-Score"] <= 7),
    }
    checks["All Valid"] = all(checks.values())
    return checks

# ✅ Load your input CSV
input_csv = "SelectData.csv"   # PFAS_GW_DataFair_try2   SelectData   PFAS_GW_DataFair    All_data_LLM2
# input_csv = "PFAS_DataFair_OTHER.csv"
df = pd.read_csv(input_csv)

if "Website Link" not in df.columns:
    raise ValueError("Column 'Website Link' not found in CSV.")

# ✅ Run evaluations
results = []

for index, row in df.iterrows():
    dataset_name = row["Dataset Name"]
    website_link = row["Website Link"]

    print(f"Evaluating: {dataset_name} ({website_link})")

    try:
        fair_output = evaluate_fair_principles(dataset_name, website_link)
        print(fair_output)

        # Parse and validate
        parsed = extract_scores_from_markdown(fair_output)
        checks = check_fair_score_consistency(parsed)

        result = {
            "Dataset Name": dataset_name,
            "Website Link": website_link,
            "FAIR Evaluation": fair_output
        }
        result.update(parsed)
        result.update(checks)

        results.append(result)

    except Exception as e:
        print(f"Error with {dataset_name}: {e}")
        results.append({
            "Dataset Name": dataset_name,
            "Website Link": website_link,
            "FAIR Evaluation": f"Error: {str(e)}",
            "F-Score": None,
            "A-Score": None,
            "I-Score": None,
            "R-Score": None,
            "F-Score Valid": False,
            "A-Score Valid": False,
            "I-Score Valid": False,
            "R-Score Valid": False,
            "All Valid": False
        })

    time.sleep(3)

# ✅ Save to final CSV
output_csv = "SelectData_LLM2_epa_T02-2_gpt-4o.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print("✅ All done! Output saved to:", output_csv) 





