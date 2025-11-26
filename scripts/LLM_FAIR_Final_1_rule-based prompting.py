#AZURE_DEPLOYMENT_NAME = "gpt-4o"
#AZURE_API_VERSION = "2024-02-01"

import openai
import os
import pandas as pd
import time
import re
import requests
from bs4 import BeautifulSoup

# ================================
# 1. Azure OpenAI Configuration
# ================================

AZURE_OPENAI_ENDPOINT = "https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"
AZURE_OPENAI_API_KEY = "xxx"  # Replace with actual API key
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION = "2024-02-01"


client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ================================
# 2. Website Scraper
# ================================
def scrape_website(url):
    """Fetch HTML content and extract key metadata for FAIR evaluation."""
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        soup = BeautifulSoup(html, "lxml")

        # Title
        title = soup.title.string.strip() if soup.title else "Not found"

        # Meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"].strip() if meta_desc else "Not found"

        # Page text for license cues + context
        text = soup.get_text(" ", strip=True)
        license_keywords = ["license", "creativecommons", "CC-BY", "public domain", "creative commons"]
        found_license = [kw for kw in license_keywords if kw.lower() in text.lower()]
        license_info = ", ".join(found_license) if found_license else "Not detected"

        # File formats from links / HTML
        file_formats = re.findall(r'\.(csv|xlsx?|json|xml|zip|shp|kml|txt|pdf)', html, re.IGNORECASE)
        file_formats = list(set(file_formats)) if file_formats else ["None detected"]

        # Downloadable links
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
            "raw_text_snippet": text[:2000]  # trim for token control
        }

    except Exception as e:
        return {
            "title": "Error scraping website",
            "description": str(e),
            "license_info": "N/A",
            "file_formats": ["N/A"],
            "download_links": ["N/A"],
            "raw_text_snippet": ""
        }

# ================================
# 3. FAIR Scoring Rubric 
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

# ================================
# 4. Zero-shot CoT FAIR Evaluation
# ================================
def evaluate_fair_principles(dataset_name, website_link, scraped_data):
    """
    Zero-shot CoT:
    - No examples
    - Ask model to think step-by-step internally
    - But only output the final markdown table (for parsing)
    """
    scraped_text = f"""
### Extracted Website Content
- Title: {scraped_data['title']}
- Description: {scraped_data['description']}
- License: {scraped_data['license_info']}
- File Formats: {scraped_data['file_formats']}
- Download Links: {scraped_data['download_links']}
- Raw Page Text Snippet:
{scraped_data['raw_text_snippet']}
"""

    user_content = (
        SCORING_RULES
        + scraped_text
        + f"""
Using ONLY the extracted webpage content above, evaluate the FAIR principles for:

Dataset: "{dataset_name}"
URL: {website_link}

First, think step-by-step internally using the rubric to decide the scores.
Then, provide ONLY the final answer as a markdown table in this exact format:

| Dataset Name | F-Score (X/17) | A-Score (X/10) | I-Score (X/8) | R-Score (X/7) |

Do not include any explanations, reasoning text, or additional commentary in your final output.
"""
    )

    prompt = {
        "model": AZURE_DEPLOYMENT_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in FAIR data assessment. Follow the rubric exactly and be consistent."
            },
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }

    response = client.chat.completions.create(**prompt)
    return response.choices[0].message.content

# ================================
# 5. Markdown Score Extraction (regex)
# ================================
def extract_scores(markdown_str):
    pattern = (
        r'\|\s*(.*?)\s*\|'
        r'\s*F-Score \((\d+)/17\)\s*\|'
        r'\s*A-Score \((\d+)/10\)\s*\|'
        r'\s*I-Score \((\d+)/8\)\s*\|'
        r'\s*R-Score \((\d+)/7\)\s*\|'
    )
    match = re.search(pattern, markdown_str, flags=re.DOTALL)

    if match:
        return {
            "Dataset Name (Parsed)": match.group(1).strip(),
            "F-Score": int(match.group(2)),
            "A-Score": int(match.group(3)),
            "I-Score": int(match.group(4)),
            "R-Score": int(match.group(5)),
            "Parse Success": True
        }
    else:
        return {
            "Dataset Name (Parsed)": None,
            "F-Score": None,
            "A-Score": None,
            "I-Score": None,
            "R-Score": None,
            "Parse Success": False
        }

# ================================
# 6. Score Validation
# ================================
def check_valid(row):
    if not row.get("Parse Success"):
        return False
    return (
        0 <= row["F-Score"] <= 17 and
        0 <= row["A-Score"] <= 10 and
        0 <= row["I-Score"] <= 8 and
        0 <= row["R-Score"] <= 7
    )

# ================================
# 7. Main Pipeline
# ================================
input_csv = "SelectData.csv"
output_csv = "SelectData_ZEROSHOT_CoT_scraped_gpt4o.csv"

df = pd.read_csv(input_csv)
if "Website Link" not in df.columns or "Dataset Name" not in df.columns:
    raise ValueError("CSV must contain 'Dataset Name' and 'Website Link' columns.")

results = []

for idx, row in df.iterrows():
    dataset_name = row["Dataset Name"]
    website_link = row["Website Link"]

    print(f"\n🔍 Scraping: {website_link}")
    scraped = scrape_website(website_link)

    print(f"🤖 Zero-shot CoT FAIR evaluation for: {dataset_name}")
    fair_output = evaluate_fair_principles(dataset_name, website_link, scraped)

    parsed = extract_scores(fair_output)
    valid = check_valid(parsed)

    result = {
        "Dataset Name": dataset_name,
        "Website Link": website_link,
        "FAIR Raw Output": fair_output,
        "Scraped Title": scraped["title"],
        "Scraped License": scraped["license_info"],
        "Scraped File Formats": ", ".join(scraped["file_formats"]),
        "Parse Success": parsed["Parse Success"],
        "Valid Scores": valid,
        "Parsed Dataset Name": parsed["Dataset Name (Parsed)"],
        "F-Score": parsed["F-Score"],
        "A-Score": parsed["A-Score"],
        "I-Score": parsed["I-Score"],
        "R-Score": parsed["R-Score"]
    }

    results.append(result)
    time.sleep(3)  # to respect rate limits

# ================================
# 8. Save Results
# ================================
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print("\n✅ Zero-shot CoT FAIR evaluation complete. Results saved to:", output_csv)



