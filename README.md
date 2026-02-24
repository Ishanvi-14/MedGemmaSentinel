
# Sentinel Oncology Clinical Auditor (v2.0)

**Sentinel** is an advanced, offline clinical auditor designed to assist oncology professionals in auditing patient records against **RECIST 1.1** and **NCCN guidelines** (:contentReference[oaicite:0]{index=0}).  

Built for the Google HAI-DEF competition, this version introduces six major enhancement areas focusing on **safety, explainability, and patient-centric communication**.

---

## 🌟 Key Features

### 1. Multi-Format Input System

Sentinel supports a wide array of clinical data sources, automatically detecting the format and normalizing biomarkers.

**Supported Formats:**
- PDF lab reports  
- HL7 FHIR JSON  
- Synthea JSON  
- CSV  
- Plain text clinical notes  

**NLP Extraction:**
- Uses MedGemma (gemma3:4b via Ollama) to extract structured biomarkers from unstructured text.

---

### 2. Clinical Safety Guardrails (Adversarial Audit)

To prevent hallucinations and ensure data integrity, Sentinel employs a **dual-extraction strategy**:

- **Dual Extraction:** Biomarkers are extracted twice using different prompt strategies.  
- **Conflict Detection:** Automatically flags discrepancies (e.g., tumor size differences >10%) and halts the workflow for human review.  

This ensures audit reliability before guideline comparison is performed.

---

### 3. Explainable RAG (Retrieval-Augmented Generation)

All audit findings are backed by traceable evidence from NCCN guidelines.

- **Rich Metadata:** Each recommendation includes:
  - Source document  
  - Page number  
  - Section title  
  - Version  

- **Confidence Scoring:**  
  Visual indicators (0–100%) based on vector similarity distance help clinicians assess guideline relevance.

---

### 4. Digital Twin & Longitudinal Analysis

Visualize the patient’s disease trajectory over time.

- **Trend Tracking:**  
  Linear regression analysis of historical biomarkers (requires ≥3 data points).

- **Intervention Simulation:**  
  Models potential impact of chemotherapy or surgery on biomarker growth rates using evidence-based heuristics.

---

### 5. Multi-Lingual Patient Translation

Bridges communication between clinicians and patients.

- **Simplification:**  
  Converts technical jargon into a 5th-grade reading level.

- **Languages Supported:**  
  - English  
  - Spanish  
  - Hindi  
  - Marathi  

Original medical terminology is preserved in parentheses for accuracy.

---

### 6. Professional Export System

- **Clinical PDF Reports:**  
  Includes color-coded severity indicators:
  - 🟢 Low Concern  
  - 🟡 Monitor  
  - 🔴 Action Required  

- **Structured JSON Export:**  
  Fully compatible with legacy Sentinel systems for institutional integration.

---

## 🛠 Tech Stack

| Component        | Technology |
|-----------------|------------|
| Core Logic       | Python 3.9+ |
| LLM              | MedGemma (gemma3:4b) via Ollama |
| Orchestration    | LangGraph |
| Vector Database  | LanceDB |
| Audit Logs       | SQLite |
| Frontend         | Streamlit |
| Testing          | Pytest & Hypothesis |

---

## 🚀 Quick Start

### Prerequisites

- Ollama installed and running  
- Pull the required model:

```bash
ollama pull gemma3:4b
````

---

### Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/sentinel-v2.git
cd sentinel-v2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Initialize the system:

```bash
# Initializes LanceDB with NCCN guidelines and SQLite tables
python scripts/initialize_system.py
```

Run Sentinel:

```bash
streamlit run app.py
```

---

## 🧪 Testing & Reliability

Sentinel follows a **dual-testing strategy** to ensure clinical-grade robustness:

### 1. Unit Tests

Validates specific oncology edge cases.

### 2. Property-Based Testing

Uses Hypothesis to execute 100+ randomized iterations per property.
Example:

* Verifying unit normalization between mm and cm always holds true.
* Ensuring tumor progression thresholds comply with RECIST 1.1.

---

## 🔒 Security & Privacy

* **100% Offline Operation**
  No internet connectivity required.

* **Local Processing Only**
  All Protected Health Information (PHI) remains on the local machine.
  No external API calls are made.

---

## 🎯 Mission

Sentinel v2.0 is designed to:

* Enhance clinical safety
* Increase audit transparency
* Improve patient communication
* Maintain strict privacy standards

Built to support oncologists with explainable, verifiable, and patient-centered clinical auditing.

```
```
