
# 🔍 LLM FAIR Prompt Evaluation

This project provides a minimal and modular framework to evaluate the FAIRness (Findability, Accessibility, Interoperability, and Reusability) of environmental datasets using Large Language Models (LLMs) via rule-based and prompt-driven evaluation strategies.

---

## 🗂️ Input Data Format

Your input CSV (e.g., `sample_dataset.csv`) should contain:

```csv
Dataset Name,Website Link
UCMR5,https://www.epa.gov/system/files/documents/2023-06/ucmr5-data_061523.csv
USGS PFAS Data,https://cida.usgs.gov/pubs/data/pfas/
...
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/llm-fair-prompt-evaluation.git
cd llm-fair-prompt-evaluation
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Add your Azure OpenAI API credentials**:

Edit the following variables at the top of `scripts/llm_fair_eval_rule_based.py`:

```python
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_DEPLOYMENT_NAME = "your-deployment-name"
AZURE_OPENAI_ENDPOINT = "your-api-endpoint"
AZURE_API_VERSION = "2024-02-01"
```

---

## ▶️ Running the Script

```bash
python scripts/llm_fair_eval_rule_based.py
```

- The script loads the CSV from the `data/` folder.
- Sends each dataset name and link to GPT-4 using a structured FAIR prompt.
- Parses and saves results to a new CSV file (e.g., `SelectData_LLM1_fair_scores_4o.csv`).

---

## 📈 Output Format

The resulting CSV includes:
- Dataset Name
- Website Link
- FAIR Evaluation (as a formatted markdown table)

You can post-process the output to extract and visualize component scores.

---

## 🧪 Use Case

- Assess and benchmark open environmental datasets (e.g., PFAS, groundwater) using a standardized FAIR rubric and LLM interpretation.

---

## 📄 License

MIT License. See `LICENSE` for more details.

---

## 🙋 Contact

For questions or feedback, please open an issue or contact Jialin at [jialind2@uci.edu].
```

