#AZURE_DEPLOYMENT_NAME = "gpt-4o"
#AZURE_API_VERSION = "2024-02-01"

import openai
import os
import pandas as pd
import time
import re
import requests
from bs4 import BeautifulSoup

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

# ================================
# Website Scraper
# ================================
def scrape_website(url):
    """Fetch HTML content and extract key metadata."""
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = soup.title.string.strip() if soup.title else "Not found"

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"].strip() if meta_desc else "Not found"

        # Extract license keywords from text
        text = soup.get_text(" ", strip=True)
        license_keywords = ["license", "creativecommons", "CC-BY", "public domain", "creative commons"]
        found_license = [kw for kw in license_keywords if kw.lower() in text.lower()]
        license_info = ", ".join(found_license) if found_license else "Not detected"

        # Extract file formats
        file_formats = re.findall(r'\.(csv|xlsx?|json|xml|zip|shp|kml|txt|pdf)', html, re.IGNORECASE)
        file_formats = list(set(file_formats)) if file_formats else ["None detected"]

        # Extract downloadable links
        download_links = [
            a["href"] for a in soup.find_all("a", href=True)
            if any(ext in a["href"].lower() for ext in [".csv", ".json", ".zip", ".xlsx", ".xml"])
        ]
        download_links = download_links if download_links else ["None detected"]

        return {
            "title": title,
            "description": description,
            "license_info": license_info,
            "file_formats": file_formats,
            "download_links": download_links,
            "raw_text_snippet": text[:2000]  # limit size
        }

    except Exception as e:
        return {
            "title": "Error scraping website",
            "description": str(e),
            "license_info": "N/A",
            "file_formats": [],
            "download_links": [],
            "raw_text_snippet": ""
        }


# ================================
# FAIR Scoring Rubric
# ================================
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
| Dataset Name | F-Score (9/17) | A-Score (8/10) | I-Score (7/8) | R-Score (4/7) |

---

Example:

Dataset: "National Earthquake Information Database"
Link: https://www.gns.cri.nz/data-and-resources/national-earthquake-information-database/

Step-by-step evaluation:
- Findable: 12/17
- Accessible: 8/10
- Interoperable: 5/8
- Reusable: 5/7

Final score:
| Dataset Name | F-Score (12/17) | A-Score (8/10) | I-Score (5/8) | R-Score (5/7) |

---

Now evaluate the following dataset:
"""

# ================================
# LLM FAIR Evaluation (Scraped)
# ================================
def evaluate_fair_principles(dataset_name, website_link, scraped_data):
    scraped_text = f"""
### Extracted Website Content for LLM Evaluation
- Title: {scraped_data['title']}
- Description: {scraped_data['description']}
- License Detected: {scraped_data['license_info']}
- File Formats: {", ".join(scraped_data['file_formats'])}
- Downloadable Links: {scraped_data['download_links']}
- Raw Page Text Snippet:
{scraped_data['raw_text_snippet']}
"""

    prompt = {
        "model": AZURE_DEPLOYMENT_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert in dataset evaluation and FAIR principles assessment."},
            {
                "role": "user",
                "content": SCORING_RULES
                           + FEWSHOT_EXAMPLE
                           + scraped_text
                           + f"""
Dataset: "{dataset_name}"
URL: {website_link}

Evaluate the dataset using the extracted website content and the FAIR rubric.
Return ONLY the markdown table:
| Dataset Name | F-Score (X/17) | A-Score (X/10) | I-Score (X/8) | R-Score (X/7) |
"""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    response = client.chat.completions.create(**prompt)
    return response.choices[0].message.content

# ================================
# Markdown Score Extractor
# ================================
def extract_scores_from_markdown(markdown_str):
    pattern = r'\|\s*(.*?)\s*\|\s*F-Score \((\d+)/17\).*A-Score \((\d+)/10\).*I-Score \((\d+)/8\).*R-Score \((\d+)/7\)'
    match = re.search(pattern, markdown_str, re.DOTALL)

    if match:
        return {
            "Dataset Name (Parsed)": match.group(1).strip(),
            "F-Score": int(match.group(2)),
            "A-Score": int(match.group(3)),
            "I-Score": int(match.group(4)),
            "R-Score": int(match.group(5)),
        }
    else:
        return {"Dataset Name (Parsed)": None, "F-Score": None, "A-Score": None, "I-Score": None, "R-Score": None}

# ================================
# Score Validator
# ================================
def check_fair_score_consistency(row):
    checks = {
        "F-Score Valid": row["F-Score"] is not None and 0 <= row["F-Score"] <= 17,
        "A-Score Valid": row["A-Score"] is not None and 0 <= row["A-Score"] <= 10,
        "I-Score Valid": row["I-Score"] is not None and 0 <= row["I-Score"] <= 8,
        "R-Score Valid": row["R-Score"] is not None and 0 <= row["R-Score"] <= 7,
    }
    checks["All Valid"] = all(checks.values())
    return checks

# ================================
# Main Pipeline
# ================================
input_csv = "SelectData.csv"
df = pd.read_csv(input_csv)

results = []

for _, row in df.iterrows():
    dataset_name = row["Dataset Name"]
    website_link = row["Website Link"]

    print(f"\n🔍 Scraping website: {website_link}")
    scraped = scrape_website(website_link)

    print(f" Evaluating FAIR for: {dataset_name}")
    fair_output = evaluate_fair_principles(dataset_name, website_link, scraped)

    # Parse and check scores
    parsed = extract_scores_from_markdown(fair_output)
    checks = check_fair_score_consistency(parsed)

    result = {
        "Dataset Name": dataset_name,
        "Website Link": website_link,
        "FAIR Evaluation Raw Output": fair_output,
        "Scraped Title": scraped["title"],
        "Scraped License": scraped["license_info"],
        "Scraped File Formats": scraped["file_formats"],
    }
    result.update(parsed)
    result.update(checks)

    results.append(result)
    time.sleep(3)

# ================================
# Save Results
# ================================
output_csv = "SelectData_LLM_scraped_gpt4o.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)

print("\n✅ All done! Output saved to:", output_csv)
