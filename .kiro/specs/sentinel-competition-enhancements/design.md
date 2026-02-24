# Design Document: Sentinel Competition Enhancements

## Overview

This design document describes the architecture and implementation approach for enhancing the Sentinel oncology clinical auditor system. The enhancements add six major capability areas while maintaining the existing tech stack (Python, LangGraph, LanceDB, Streamlit, Ollama/MedGemma) and ensuring offline operation with sub-30-second response times.

The design follows a modular approach where each enhancement integrates into the existing LangGraph workflow as new nodes or augmented existing nodes. This ensures backward compatibility while enabling progressive feature adoption.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI Layer                        │
│  (File Upload, Language Selection, Report Download, Viz)        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    LangGraph Orchestration                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Input Parser │→ │Safety Auditor│→ │ RAG Retrieval│         │
│  │    Node      │  │    Node      │  │     Node     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ RECIST Audit │  │Digital Twin  │  │   Patient    │         │
│  │    Node      │  │    Node      │  │  Translator  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    Report Generation Layer                       │
│              (PDF Generator, JSON Exporter)                      │
└─────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    Data Persistence Layer                        │
│         (LanceDB Vector Store, Audit Log, Patient DB)           │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow Enhancement

The existing LangGraph workflow will be extended with new nodes:

```python
# Existing workflow (preserved)
workflow = StateGraph(AuditState)
workflow.add_node("extract_biomarkers", extract_biomarkers_node)
workflow.add_node("retrieve_guidelines", retrieve_guidelines_node)
workflow.add_node("audit_recist", audit_recist_node)

# New nodes (added)
workflow.add_node("parse_input", parse_input_node)  # Multi-format parsing
workflow.add_node("safety_check", safety_auditor_node)  # Hallucination check
workflow.add_node("enrich_rag", enrich_rag_metadata_node)  # Explainable RAG
workflow.add_node("analyze_trends", digital_twin_node)  # Longitudinal analysis
workflow.add_node("translate_patient", patient_translator_node)  # Translation
workflow.add_node("generate_reports", report_generator_node)  # PDF/JSON export

# Enhanced workflow edges
workflow.add_edge("parse_input", "extract_biomarkers")
workflow.add_edge("extract_biomarkers", "safety_check")
workflow.add_conditional_edge("safety_check", route_safety_check)
workflow.add_edge("safety_check", "retrieve_guidelines")
workflow.add_edge("retrieve_guidelines", "enrich_rag")
workflow.add_edge("enrich_rag", "audit_recist")
workflow.add_edge("audit_recist", "analyze_trends")
workflow.add_conditional_edge("analyze_trends", route_translation)
workflow.add_edge("translate_patient", "generate_reports")
```

## Components and Interfaces

### 1. Input Parser Component

**Purpose:** Detect and parse multiple clinical document formats into normalized biomarker data.

**Interface:**
```python
class InputParser:
    def detect_format(self, file_content: bytes, filename: str) -> str:
        """
        Detect input format from content and filename.
        Returns: "synthea_json" | "fhir_json" | "pdf" | "csv" | "text"
        """
        pass
    
    def parse(self, file_content: bytes, format: str) -> ParsedClinicalData:
        """
        Parse file content based on detected format.
        Returns: Normalized clinical data with biomarkers
        """
        pass
    
    def normalize_biomarkers(self, raw_data: dict) -> List[Biomarker]:
        """
        Convert format-specific data to common biomarker representation.
        Returns: List of normalized biomarker objects
        """
        pass

class Biomarker:
    name: str  # e.g., "tumor_size", "CEA", "EGFR"
    value: float
    unit: str  # e.g., "mm", "ng/mL", "mutation_status"
    timestamp: datetime
    source_field: str  # Original field name for traceability
    confidence: float  # 0.0-1.0
```

**Format Detection Logic:**
- PDF: Check for PDF magic bytes (`%PDF`)
- FHIR JSON: Check for `resourceType` field
- Synthea JSON: Check for `entry` array with Synthea-specific structure
- CSV: Check for comma-separated structure with header row
- Text: Default fallback for unstructured clinical notes

**Parsing Strategies:**
- **PDF:** Use PyPDF2 or pdfplumber to extract text, then apply regex patterns for biomarker extraction
- **FHIR JSON:** Navigate FHIR resource structure (Observation, DiagnosticReport) to extract coded values
- **Synthea JSON:** Use existing parser logic (backward compatibility)
- **CSV:** Use pandas to read tabular data, map columns to biomarkers
- **Text:** Use MedGemma with structured prompts to extract biomarkers via NLP

### 2. Safety Auditor Component

**Purpose:** Validate biomarker extractions by performing adversarial duplicate extraction and consistency checking.

**Interface:**
```python
class SafetyAuditor:
    def extract_with_prompt_a(self, clinical_data: str) -> List[Biomarker]:
        """Extract biomarkers using primary prompt strategy."""
        pass
    
    def extract_with_prompt_b(self, clinical_data: str) -> List[Biomarker]:
        """Extract biomarkers using alternative prompt strategy."""
        pass
    
    def compare_extractions(
        self, 
        extraction_a: List[Biomarker], 
        extraction_b: List[Biomarker]
    ) -> ComparisonResult:
        """
        Compare two extractions for consistency.
        Returns: Conflicts, confidence scores, and pass/fail status
        """
        pass
    
    def normalize_units(self, value: float, unit: str) -> Tuple[float, str]:
        """
        Convert to standard units for comparison.
        Examples: 20mm -> 2.0cm, 0.002g/L -> 2ng/mL
        """
        pass
    
    def calculate_confidence(
        self, 
        extraction_a: Biomarker, 
        extraction_b: Biomarker
    ) -> float:
        """
        Calculate confidence score (0-100%) based on agreement.
        100% = perfect match, 0% = complete disagreement
        """
        pass

class ComparisonResult:
    conflicts: List[Conflict]
    all_biomarkers: List[Biomarker]  # Merged with confidence scores
    requires_human_review: bool
    overall_confidence: float

class Conflict:
    biomarker_name: str
    value_a: float
    value_b: float
    unit: str
    discrepancy_percentage: float
    timestamp: datetime
```

**Prompt Strategies:**
- **Prompt A (Direct):** "Extract the following biomarkers from the clinical data: tumor size, CEA level, EGFR status. Provide values with units."
- **Prompt B (Contextual):** "You are a clinical data specialist. Review this patient record and identify all measurable cancer biomarkers, including their values and measurement units."

**Consistency Thresholds:**
- Continuous values (tumor size, CEA): ±10% tolerance
- Categorical values (EGFR mutation): Exact match required
- Missing in one extraction: Flag for review

### 3. Enhanced Vector Store Component

**Purpose:** Extend LanceDB vector store to return rich metadata for explainable RAG.

**Interface:**
```python
class EnhancedVectorStore:
    def search_with_metadata(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[GuidelineResult]:
        """
        Search guidelines and return results with full metadata.
        """
        pass
    
    def calculate_confidence_score(self, vector_distance: float) -> float:
        """
        Convert vector distance to confidence percentage (0-100%).
        Uses inverse exponential scaling.
        """
        pass

class GuidelineResult:
    content: str  # Guideline text excerpt
    source_document: str  # e.g., "NCCN_Colon_Cancer_v2.2023.pdf"
    page_number: int
    section_title: str  # e.g., "Treatment Response Criteria"
    guideline_version: str  # e.g., "2.2023"
    publication_date: date
    confidence_score: float  # 0-100%
    vector_distance: float  # Raw distance for debugging
```

**Metadata Storage:**
When ingesting guidelines into LanceDB, store metadata alongside embeddings:
```python
# During ingestion
lance_table.add([
    {
        "vector": embedding,
        "text": chunk_text,
        "source_document": "NCCN_Colon_Cancer_v2.2023.pdf",
        "page_number": 42,
        "section_title": "RECIST 1.1 Criteria",
        "guideline_version": "2.2023",
        "publication_date": "2023-03-15"
    }
])
```

**Confidence Score Calculation:**
```python
def calculate_confidence_score(vector_distance: float) -> float:
    # Lower distance = higher confidence
    # Typical cosine distances range 0.0-2.0
    # Map to 0-100% using inverse exponential
    confidence = 100 * math.exp(-vector_distance)
    return min(100.0, max(0.0, confidence))
```

### 4. Digital Twin Component

**Purpose:** Analyze biomarker trends over time and simulate future scenarios.

**Interface:**
```python
class DigitalTwin:
    def load_patient_history(self, patient_id: str) -> List[AuditRecord]:
        """Load all historical audit records for a patient."""
        pass
    
    def calculate_trends(
        self, 
        history: List[AuditRecord]
    ) -> Dict[str, TrendAnalysis]:
        """
        Calculate trend for each biomarker using linear regression.
        Returns: Dictionary mapping biomarker name to trend analysis
        """
        pass
    
    def predict_next_value(
        self, 
        trend: TrendAnalysis, 
        days_ahead: int
    ) -> PredictedValue:
        """
        Predict biomarker value N days in the future.
        """
        pass
    
    def calculate_threshold_timeline(
        self, 
        trend: TrendAnalysis, 
        threshold: float
    ) -> Optional[date]:
        """
        Calculate when biomarker will reach critical threshold.
        Returns None if threshold won't be reached.
        """
        pass
    
    def simulate_intervention(
        self, 
        trend: TrendAnalysis, 
        intervention: Intervention
    ) -> TrendAnalysis:
        """
        Simulate impact of medication/treatment change on trend.
        Uses simple heuristics (e.g., -20% growth rate for chemo).
        """
        pass

class TrendAnalysis:
    biomarker_name: str
    slope: float  # Rate of change per day
    intercept: float
    r_squared: float  # Goodness of fit
    data_points: int
    sufficient_data: bool  # True if >= 3 data points

class PredictedValue:
    value: float
    unit: str
    prediction_date: date
    confidence_interval: Tuple[float, float]  # (lower, upper)

class Intervention:
    type: str  # "chemotherapy", "immunotherapy", "surgery"
    expected_impact: float  # Percentage change in slope
```

**Trend Calculation:**
- Use scipy.stats.linregress for simple linear regression
- Require minimum 3 data points for trend analysis
- Calculate R² to assess trend reliability
- Flag trends with R² < 0.5 as "insufficient data"

**Simulation Heuristics:**
```python
INTERVENTION_IMPACTS = {
    "chemotherapy": -0.30,  # 30% reduction in growth rate
    "immunotherapy": -0.25,
    "targeted_therapy": -0.40,
    "surgery": -0.80,  # Immediate 80% reduction
    "radiation": -0.35
}
```

### 5. Patient Translator Component

**Purpose:** Convert technical audit findings to patient-friendly language in multiple languages.

**Interface:**
```python
class PatientTranslator:
    def simplify_findings(
        self, 
        audit_findings: List[Finding], 
        reading_level: int = 5
    ) -> List[SimplifiedFinding]:
        """
        Convert technical findings to specified reading level.
        Uses MedGemma with reading-level-specific prompts.
        """
        pass
    
    def translate_to_language(
        self, 
        simplified_findings: List[SimplifiedFinding], 
        target_language: str
    ) -> List[TranslatedFinding]:
        """
        Translate simplified findings to target language.
        Supported: "en", "es", "hi", "mr"
        """
        pass
    
    def add_visual_aids(
        self, 
        finding: SimplifiedFinding
    ) -> SimplifiedFinding:
        """
        Add emoji/icon indicators for severity levels.
        """
        pass
    
    def validate_medical_accuracy(
        self, 
        original: Finding, 
        simplified: SimplifiedFinding
    ) -> bool:
        """
        Verify that simplification preserves medical meaning.
        Checks for critical term preservation.
        """
        pass

class SimplifiedFinding:
    original_text: str
    simplified_text: str
    reading_level: int
    severity_icon: str  # "🟢", "🟡", "🔴"
    key_terms_preserved: List[str]

class TranslatedFinding:
    simplified_finding: SimplifiedFinding
    language: str
    translated_text: str
```

**Simplification Prompt Template:**
```python
SIMPLIFICATION_PROMPT = """
You are a medical translator helping patients understand their test results.

Original finding: {technical_finding}

Rewrite this for a patient with a 5th-grade reading level. Rules:
1. Use simple words (avoid: "metastatic", use: "cancer spread")
2. Keep sentences short (max 15 words)
3. Preserve critical medical facts
4. Use analogies when helpful
5. Maintain accuracy - do not minimize serious findings

Simplified version:
"""
```

**Translation Approach:**
- Use MedGemma with language-specific prompts
- Preserve medical terminology in original language (in parentheses)
- Example: "El tumor ha crecido (tumor growth) de 2cm a 3cm"

### 6. Report Generator Component

**Purpose:** Create professional PDF and JSON exports of audit results.

**Interface:**
```python
class ReportGenerator:
    def generate_pdf_report(
        self, 
        audit_result: AuditResult, 
        include_trends: bool = True
    ) -> bytes:
        """
        Generate professional PDF report.
        Returns: PDF file as bytes
        """
        pass
    
    def generate_json_export(
        self, 
        audit_result: AuditResult
    ) -> dict:
        """
        Generate structured JSON export for system integration.
        """
        pass
    
    def create_patient_pdf(
        self, 
        translated_findings: List[TranslatedFinding], 
        language: str
    ) -> bytes:
        """
        Generate patient-friendly PDF in specified language.
        """
        pass

class PDFReport:
    sections: List[PDFSection]
    
class PDFSection:
    title: str
    content: Union[str, Table, Chart]

# PDF Structure
# 1. Header (Patient ID, Date, Sentinel Version)
# 2. Patient Summary
# 3. Extracted Biomarkers Table
# 4. Audit Findings (with severity indicators)
# 5. Referenced Guidelines (with page numbers and confidence)
# 6. Trend Visualizations (if available)
# 7. Footer (Timestamp, Disclaimer)
```

**PDF Generation:**
- Use ReportLab library for PDF creation
- Use matplotlib/plotly for embedding charts
- Include color-coded severity indicators:
  - 🟢 Green: No concerns
  - 🟡 Yellow: Monitor closely
  - 🔴 Red: Immediate attention required

**JSON Export Schema:**
```json
{
  "report_metadata": {
    "patient_id": "string",
    "audit_timestamp": "ISO8601",
    "sentinel_version": "string",
    "report_id": "uuid"
  },
  "biomarkers": [
    {
      "name": "string",
      "value": "number",
      "unit": "string",
      "confidence_score": "number",
      "extraction_method": "string"
    }
  ],
  "audit_findings": [
    {
      "finding_id": "string",
      "severity": "low|medium|high",
      "description": "string",
      "guideline_reference": {
        "source": "string",
        "page": "number",
        "confidence": "number"
      }
    }
  ],
  "trends": {
    "biomarker_name": {
      "slope": "number",
      "r_squared": "number",
      "predictions": []
    }
  },
  "safety_audit": {
    "conflicts_detected": "boolean",
    "overall_confidence": "number",
    "conflicts": []
  }
}
```

## Data Models

### Core Data Structures

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Union
from enum import Enum

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class BiomarkerType(Enum):
    TUMOR_SIZE = "tumor_size"
    CEA = "CEA"
    EGFR = "EGFR"
    CA_19_9 = "CA_19_9"
    CUSTOM = "custom"

@dataclass
class Biomarker:
    name: str
    value: float
    unit: str
    timestamp: datetime
    source_field: str
    confidence: float
    biomarker_type: BiomarkerType

@dataclass
class ParsedClinicalData:
    patient_id: str
    biomarkers: List[Biomarker]
    raw_text: str
    format_type: str
    parse_timestamp: datetime
    metadata: Dict[str, any]

@dataclass
class Finding:
    finding_id: str
    severity: SeverityLevel
    description: str
    biomarker_name: Optional[str]
    guideline_reference: Optional['GuidelineResult']
    timestamp: datetime

@dataclass
class AuditResult:
    audit_id: str
    patient_id: str
    biomarkers: List[Biomarker]
    findings: List[Finding]
    guidelines_used: List['GuidelineResult']
    safety_check: 'ComparisonResult'
    trends: Optional[Dict[str, 'TrendAnalysis']]
    timestamp: datetime
    sentinel_version: str

@dataclass
class AuditState:
    """LangGraph state object"""
    raw_input: bytes
    filename: str
    parsed_data: Optional[ParsedClinicalData]
    biomarkers_extraction_a: Optional[List[Biomarker]]
    biomarkers_extraction_b: Optional[List[Biomarker]]
    safety_result: Optional[ComparisonResult]
    guidelines: Optional[List[GuidelineResult]]
    audit_findings: Optional[List[Finding]]
    trends: Optional[Dict[str, TrendAnalysis]]
    translated_findings: Optional[List[TranslatedFinding]]
    pdf_report: Optional[bytes]
    json_export: Optional[dict]
    error: Optional[str]
    requires_human_review: bool
```

### Database Schema Extensions

**Audit Log Table (SQLite):**
```sql
CREATE TABLE audit_log (
    audit_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    input_format TEXT,
    biomarkers_json TEXT,  -- JSON array of biomarkers
    findings_json TEXT,    -- JSON array of findings
    safety_conflicts_json TEXT,  -- JSON array of conflicts
    overall_confidence REAL,
    requires_review BOOLEAN,
    sentinel_version TEXT
);

CREATE INDEX idx_patient_timestamp ON audit_log(patient_id, timestamp);
```

**Patient History Table:**
```sql
CREATE TABLE patient_history (
    record_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    audit_id TEXT REFERENCES audit_log(audit_id),
    biomarker_name TEXT,
    biomarker_value REAL,
    biomarker_unit TEXT,
    measurement_date DATE,
    confidence REAL
);

CREATE INDEX idx_patient_biomarker ON patient_history(patient_id, biomarker_name, measurement_date);
```

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Input Parsing Properties

**Property 1: Format-Specific Extraction**
*For any* valid clinical document in a supported format (PDF, FHIR JSON, Synthea JSON, CSV, plain text), parsing should successfully extract biomarkers with confidence scores.
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

**Property 2: Format Auto-Detection**
*For any* file with valid format signatures, the Input_Parser should correctly identify the format type.
**Validates: Requirements 1.6**

**Property 3: Invalid Format Error Handling**
*For any* unsupported or malformed file, the Input_Parser should return a descriptive error message listing supported formats.
**Validates: Requirements 1.7**

**Property 4: Cross-Format Normalization**
*For any* biomarker data parsed from different formats representing the same clinical information, the normalized internal representation should be equivalent.
**Validates: Requirements 1.8**

### Report Generation Properties

**Property 5: PDF Report Completeness**
*For any* completed audit result, the generated PDF should contain all required sections: patient summary, biomarkers table with confidence scores, audit findings with severity indicators, referenced guidelines with page numbers and confidence scores, timestamp, and version information.
**Validates: Requirements 2.1, 2.2, 2.3, 2.7**

**Property 6: JSON Export Completeness**
*For any* completed audit result, the generated JSON export should contain all audit data in structured format matching the defined schema.
**Validates: Requirements 2.4**

**Property 7: Trend Visualization Inclusion**
*For any* audit result with biomarker trend data, the generated PDF should include trend visualizations.
**Validates: Requirements 2.8**

### Safety Auditor Properties

**Property 8: Dual Extraction Execution**
*For any* clinical data input, the Safety_Auditor should perform exactly two biomarker extractions using different prompts.
**Validates: Requirements 3.1**

**Property 9: Unit Normalization for Comparison**
*For any* pair of biomarker values with equivalent measurements in different units (e.g., 20mm and 2.0cm), the Safety_Auditor should recognize them as matching after normalization.
**Validates: Requirements 3.2**

**Property 10: Conflict Detection and Flagging**
*For any* pair of duplicate extractions where values differ beyond acceptable tolerance, the Safety_Auditor should flag a Clinical_Conflict.
**Validates: Requirements 3.3**

**Property 11: Workflow Routing Based on Conflicts**
*For any* safety audit result, the workflow should require human review if conflicts are detected, and should proceed automatically if no conflicts are detected.
**Validates: Requirements 3.4, 3.7**

**Property 12: Confidence Score Assignment**
*For any* extracted biomarker, the Safety_Auditor should assign a confidence score between 0.0 and 1.0.
**Validates: Requirements 3.5**

**Property 13: Conflict Logging**
*For any* detected Clinical_Conflict, the Safety_Auditor should log complete conflict details including both extraction results, discrepancy percentage, and timestamp.
**Validates: Requirements 3.6**

**Property 14: Low Confidence Flagging**
*For any* biomarker extraction with confidence score below threshold (e.g., 0.6), the Safety_Auditor should flag the value for review regardless of conflict status.
**Validates: Requirements 3.8**

### Explainable RAG Properties

**Property 15: Guideline Metadata Completeness**
*For any* guideline retrieved from Vector_Store, the result should include complete metadata: source document name, page number, section title, guideline version, publication date, and confidence score (0-100%).
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

**Property 16: Confidence Score Range Validity**
*For any* guideline retrieval result, the confidence score should be within the valid range of 0-100%.
**Validates: Requirements 4.5**

**Property 17: Recommendation Sorting by Confidence**
*For any* set of multiple guideline recommendations, they should be sorted in descending order by confidence score.
**Validates: Requirements 4.8**

### Longitudinal Simulation Properties

**Property 18: Trend Calculation for Patient History**
*For any* patient with multiple audit records (≥3), the Digital_Twin should calculate trend analysis for each biomarker present in the history.
**Validates: Requirements 5.1**

**Property 19: Prediction with Sufficient Data**
*For any* biomarker trend with sufficient historical data (≥3 points, R² ≥ 0.5), the Digital_Twin should generate predictions for future values.
**Validates: Requirements 5.3**

**Property 20: Threshold Timeline Calculation**
*For any* biomarker trend and critical threshold, the Digital_Twin should calculate the projected date when the threshold will be reached (or return None if threshold won't be reached).
**Validates: Requirements 5.4**

**Property 21: Intervention Impact Simulation**
*For any* biomarker trend and intervention type, the Digital_Twin should modify the trend prediction according to intervention-specific heuristics.
**Validates: Requirements 5.5**

**Property 22: Insufficient Data Messaging**
*For any* patient history with fewer than 3 data points for a biomarker, the Digital_Twin should return a message indicating minimum data requirements.
**Validates: Requirements 5.8**

### Patient Translation Properties

**Property 23: Reading Level Simplification**
*For any* technical audit finding, the Patient_Translator should generate simplified text with a readability score appropriate for 5th-grade level (Flesch-Kincaid grade ≤ 5.5).
**Validates: Requirements 6.1**

**Property 24: Medical Accuracy Preservation**
*For any* simplified finding, critical medical terms and factual information from the original finding should be preserved in the simplified version.
**Validates: Requirements 6.4**

**Property 25: Severity Icon Inclusion**
*For any* patient report with findings, each finding should include a visual severity indicator (icon/emoji).
**Validates: Requirements 6.5**

**Property 26: Patient PDF Generation**
*For any* translation request, the Report_Generator should create a patient-friendly PDF with simplified language and visual aids.
**Validates: Requirements 6.6**

**Property 27: Medical Terminology Preservation in Translation**
*For any* finding translated to a non-English language, key medical terms should be preserved (either in original form or with original term in parentheses).
**Validates: Requirements 6.7**

### System Reliability Properties

**Property 28: Descriptive Error Messages**
*For any* error condition during processing, the system should return a descriptive error message with recovery guidance.
**Validates: Requirements 7.3**

**Property 29: Input Validation Errors**
*For any* invalid input data, the system should return specific validation errors indicating what is invalid and why.
**Validates: Requirements 7.4**

**Property 30: Operation Logging**
*For any* processing operation, the system should create a log entry with operation type, timestamp, and outcome.
**Validates: Requirements 7.5**

### Backward Compatibility Properties

**Property 31: Existing Functionality Preservation**
*For any* Synthea JSON input that was valid in the pre-enhancement system, the enhanced system should produce identical biomarker extraction and RECIST audit results.
**Validates: Requirements 8.1, 8.2, 8.3**

**Property 32: JSON Schema Compatibility**
*For any* JSON export generated by the enhanced system, it should contain all fields that were present in the pre-enhancement system's JSON output.
**Validates: Requirements 8.7**

**Property 33: Feature-Disabled Compatibility**
*For any* audit request processed with all new features disabled, the system should behave identically to the pre-enhancement version.
**Validates: Requirements 8.6**

**Property 34: Fault Isolation**
*For any* error occurring in new enhancement features, the core auditing functionality (biomarker extraction, guideline retrieval, RECIST audit) should continue to function.
**Validates: Requirements 8.8**

## Error Handling

### Error Categories and Handling Strategies

**1. Input Parsing Errors**
- **Invalid Format**: Return error with list of supported formats
- **Corrupted File**: Return error indicating file corruption with recovery steps
- **Missing Required Fields**: Return error specifying which fields are missing
- **Unsupported Encoding**: Return error with supported encoding list

**2. Extraction Errors**
- **No Biomarkers Found**: Log warning, continue with empty biomarker list
- **Ambiguous Values**: Flag for human review with ambiguity details
- **MedGemma Timeout**: Retry once, then return error if still failing
- **Malformed LLM Response**: Log error, attempt structured parsing, flag for review

**3. Safety Audit Errors**
- **Extraction Mismatch**: Flag conflict, require human review (expected behavior)
- **Unit Conversion Failure**: Log error, use raw values, flag for review
- **Confidence Calculation Error**: Default to 0.5 confidence, log error

**4. RAG Retrieval Errors**
- **Vector Store Unavailable**: Return error, cannot proceed with audit
- **No Guidelines Found**: Log warning, proceed with empty guideline list
- **Metadata Missing**: Use default values, log warning
- **Embedding Generation Failure**: Retry once, then return error

**5. Trend Analysis Errors**
- **Insufficient Data**: Return message to user, skip trend analysis
- **Invalid Historical Data**: Filter out invalid points, recalculate
- **Regression Failure**: Log error, skip predictions for that biomarker

**6. Translation Errors**
- **Unsupported Language**: Return error with supported language list
- **Translation Quality Low**: Log warning, include disclaimer in output
- **MedGemma Unavailable**: Return error, cannot proceed with translation

**7. Report Generation Errors**
- **PDF Generation Failure**: Log error, provide JSON export only
- **Chart Rendering Failure**: Log error, generate PDF without charts
- **File Write Error**: Return error with disk space/permissions guidance

### Error Recovery Strategies

```python
class ErrorHandler:
    def handle_error(self, error: Exception, context: dict) -> ErrorResponse:
        """
        Central error handling with context-aware recovery.
        """
        if isinstance(error, InputParsingError):
            return self._handle_parsing_error(error, context)
        elif isinstance(error, ExtractionError):
            return self._handle_extraction_error(error, context)
        elif isinstance(error, RAGError):
            return self._handle_rag_error(error, context)
        # ... other error types
        else:
            return self._handle_unknown_error(error, context)
    
    def _handle_parsing_error(self, error: InputParsingError, context: dict) -> ErrorResponse:
        """
        Parsing errors are user-facing - provide clear guidance.
        """
        return ErrorResponse(
            error_type="input_parsing",
            message=f"Unable to parse file: {error.message}",
            recovery_steps=[
                "Verify file format is one of: PDF, FHIR JSON, Synthea JSON, CSV, plain text",
                "Check that file is not corrupted",
                "Ensure file contains clinical data with biomarkers"
            ],
            can_retry=True,
            severity="medium"
        )
    
    def _handle_extraction_error(self, error: ExtractionError, context: dict) -> ErrorResponse:
        """
        Extraction errors may be transient - allow retry.
        """
        if context.get("retry_count", 0) < 1:
            return ErrorResponse(
                error_type="extraction",
                message="Biomarker extraction failed, retrying...",
                recovery_steps=["Automatic retry in progress"],
                can_retry=True,
                severity="low"
            )
        else:
            return ErrorResponse(
                error_type="extraction",
                message="Biomarker extraction failed after retry",
                recovery_steps=[
                    "Check that MedGemma is running (ollama list)",
                    "Verify clinical data contains extractable biomarkers",
                    "Review logs for detailed error information"
                ],
                can_retry=False,
                severity="high"
            )

@dataclass
class ErrorResponse:
    error_type: str
    message: str
    recovery_steps: List[str]
    can_retry: bool
    severity: str  # "low", "medium", "high"
    timestamp: datetime = field(default_factory=datetime.now)
```

### Graceful Degradation

When non-critical features fail, the system should continue with core functionality:

```python
def execute_audit_workflow(state: AuditState) -> AuditState:
    """
    Execute audit with graceful degradation.
    """
    try:
        # Core functionality (required)
        state = parse_input_node(state)
        state = extract_biomarkers_node(state)
        state = retrieve_guidelines_node(state)
        state = audit_recist_node(state)
    except Exception as e:
        # Core failure - cannot continue
        state.error = f"Core audit failed: {str(e)}"
        return state
    
    # Enhancement features (optional - degrade gracefully)
    try:
        state = safety_auditor_node(state)
    except Exception as e:
        logger.warning(f"Safety audit failed: {e}")
        state.safety_result = None  # Continue without safety check
    
    try:
        state = digital_twin_node(state)
    except Exception as e:
        logger.warning(f"Trend analysis failed: {e}")
        state.trends = None  # Continue without trends
    
    try:
        state = patient_translator_node(state)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        state.translated_findings = None  # Continue without translation
    
    # Report generation (required)
    try:
        state = generate_reports_node(state)
    except Exception as e:
        state.error = f"Report generation failed: {str(e)}"
    
    return state
```

## Testing Strategy

### Dual Testing Approach

This project requires both unit testing and property-based testing to ensure comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, error conditions, and integration points
- **Property tests**: Verify universal properties across all inputs using randomized test data

Both testing approaches are complementary and necessary. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing Configuration

**Library Selection**: Use **Hypothesis** for Python property-based testing

**Test Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `# Feature: sentinel-competition-enhancements, Property {number}: {property_text}`

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st
import hypothesis.strategies as st

# Feature: sentinel-competition-enhancements, Property 9: Unit Normalization for Comparison
@given(
    value=st.floats(min_value=0.1, max_value=1000.0),
    unit_pair=st.sampled_from([("mm", "cm"), ("ng/mL", "g/L"), ("mg", "g")])
)
@settings(max_examples=100)
def test_unit_normalization_equivalence(value, unit_pair):
    """
    Property: For any pair of biomarker values with equivalent measurements
    in different units, the Safety_Auditor should recognize them as matching
    after normalization.
    """
    unit_a, unit_b = unit_pair
    
    # Create two biomarkers with equivalent values in different units
    biomarker_a = Biomarker(name="test", value=value, unit=unit_a, ...)
    biomarker_b = Biomarker(name="test", value=convert_units(value, unit_a, unit_b), unit=unit_b, ...)
    
    # Normalize both
    norm_a = safety_auditor.normalize_units(biomarker_a.value, biomarker_a.unit)
    norm_b = safety_auditor.normalize_units(biomarker_b.value, biomarker_b.unit)
    
    # Assert they match after normalization
    assert norm_a[0] == pytest.approx(norm_b[0], rel=0.01)
    assert norm_a[1] == norm_b[1]  # Same normalized unit
```

### Unit Testing Strategy

**Focus Areas for Unit Tests**:
1. **Specific Examples**: Test known clinical scenarios with expected outputs
2. **Edge Cases**: Empty inputs, boundary values, special characters
3. **Error Conditions**: Invalid formats, missing data, malformed inputs
4. **Integration Points**: LangGraph node transitions, database operations, UI interactions

**Example Unit Test**:
```python
def test_pdf_parsing_with_standard_lab_report():
    """
    Unit test: Verify PDF parsing works with a standard lab report format.
    """
    # Load sample PDF
    with open("tests/fixtures/standard_lab_report.pdf", "rb") as f:
        pdf_content = f.read()
    
    # Parse
    parser = InputParser()
    result = parser.parse(pdf_content, "pdf")
    
    # Assert expected biomarkers are extracted
    assert len(result.biomarkers) == 3
    assert any(b.name == "CEA" and b.value == 5.2 for b in result.biomarkers)
    assert any(b.name == "tumor_size" and b.value == 25.0 for b in result.biomarkers)
```

### Test Coverage Requirements

**Minimum Coverage Targets**:
- Core parsing logic: 90% line coverage
- Safety auditor: 95% line coverage (critical for patient safety)
- Report generation: 85% line coverage
- Error handling: 80% line coverage
- UI components: 70% line coverage (harder to test, focus on logic)

**Property Test Coverage**:
- Each correctness property (1-34) must have at least one property-based test
- Properties marked as "example" in prework should have unit tests instead

### Testing Tools and Infrastructure

**Required Libraries**:
```python
# Testing
pytest==7.4.0
hypothesis==6.82.0
pytest-cov==4.1.0
pytest-mock==3.11.1

# Test fixtures
faker==19.2.0  # Generate fake clinical data
factory-boy==3.2.1  # Test data factories
```

**Test Organization**:
```
tests/
├── unit/
│   ├── test_input_parser.py
│   ├── test_safety_auditor.py
│   ├── test_vector_store.py
│   ├── test_digital_twin.py
│   ├── test_patient_translator.py
│   └── test_report_generator.py
├── property/
│   ├── test_properties_input.py
│   ├── test_properties_safety.py
│   ├── test_properties_rag.py
│   ├── test_properties_trends.py
│   ├── test_properties_translation.py
│   └── test_properties_compatibility.py
├── integration/
│   ├── test_langgraph_workflow.py
│   ├── test_end_to_end.py
│   └── test_ui_integration.py
├── fixtures/
│   ├── sample_synthea.json
│   ├── sample_fhir.json
│   ├── sample_lab_report.pdf
│   ├── sample_clinical_notes.txt
│   └── sample_lab_results.csv
└── conftest.py  # Shared fixtures and configuration
```

### Continuous Testing

**Pre-commit Hooks**:
- Run unit tests on changed files
- Run linting (black, flake8, mypy)
- Check test coverage hasn't decreased

**CI/CD Pipeline**:
1. Run all unit tests
2. Run all property tests (100 iterations each)
3. Generate coverage report
4. Run integration tests
5. Performance benchmarks (ensure <30s response time)

## Implementation Notes

### Technology Stack Constraints

All enhancements must use the existing technology stack:
- **Language**: Python 3.9+
- **LLM**: MedGemma (gemma3:4b) via Ollama
- **Orchestration**: LangGraph
- **Vector DB**: LanceDB
- **UI**: Streamlit
- **PDF Generation**: ReportLab
- **Charts**: Plotly
- **Testing**: Pytest + Hypothesis

### Performance Considerations

**30-Second Response Time Budget**:
- Input parsing: 2-3 seconds
- Dual biomarker extraction: 8-10 seconds (2 LLM calls)
- Safety audit comparison: 1 second
- RAG retrieval: 2-3 seconds
- RECIST audit: 3-4 seconds
- Trend analysis: 1-2 seconds
- Patient translation: 4-5 seconds (1 LLM call)
- Report generation: 3-4 seconds
- **Total**: ~25-32 seconds (within budget)

**Optimization Strategies**:
- Cache MedGemma embeddings for repeated queries
- Parallelize independent LangGraph nodes where possible
- Use streaming for PDF generation to reduce memory
- Limit trend analysis to last 20 data points
- Pre-compile regex patterns for parsing

### Offline Operation Requirements

**No Internet Dependencies**:
- All models run locally via Ollama
- LanceDB is file-based (no cloud connection)
- Guidelines pre-loaded into vector store
- Translation dictionaries bundled with application
- No external API calls

**Bundled Resources**:
- NCCN guidelines (pre-embedded in LanceDB)
- Medical terminology dictionaries
- Unit conversion tables
- Reading level word lists
- Language translation models (if needed beyond MedGemma)

### Security and Privacy

**Data Handling**:
- All patient data stays on local machine
- No telemetry or analytics sent externally
- Audit logs stored locally with encryption at rest
- Temporary files cleaned up after processing
- No patient data in error messages or logs

**Input Validation**:
- Sanitize all file uploads
- Validate file sizes (max 50MB)
- Check for malicious content in PDFs
- Escape special characters in text inputs
- Validate JSON schema before parsing

## Deployment Considerations

### System Requirements

**Minimum Hardware**:
- CPU: 4 cores, 2.5 GHz
- RAM: 16 GB
- Storage: 20 GB free space
- GPU: Optional (CPU inference acceptable)

**Software Dependencies**:
- Python 3.9+
- Ollama with MedGemma model
- SQLite 3.35+
- Operating System: Linux, macOS, or Windows

### Installation Steps

1. Install Ollama and pull MedGemma model
2. Install Python dependencies via requirements.txt
3. Initialize LanceDB vector store with NCCN guidelines
4. Run database migrations for audit log tables
5. Configure Streamlit settings
6. Run initial system health check

### Configuration

**Environment Variables**:
```bash
SENTINEL_VERSION=2.0.0
OLLAMA_HOST=http://localhost:11434
LANCEDB_PATH=./data/vector_store
AUDIT_LOG_PATH=./data/audit_log.db
MAX_UPLOAD_SIZE_MB=50
SAFETY_CONFIDENCE_THRESHOLD=0.6
TREND_MIN_DATA_POINTS=3
```

**Feature Flags**:
```python
FEATURES = {
    "multi_format_input": True,
    "safety_auditor": True,
    "explainable_rag": True,
    "digital_twin": True,
    "patient_translation": True,
    "pdf_export": True,
    "json_export": True
}
```
