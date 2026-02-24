# Requirements Document: Sentinel Competition Enhancements

## Introduction

This document specifies the requirements for enhancing the Sentinel oncology clinical auditor system for the Google HAI-DEF competition. The enhancements focus on expanding input capabilities, improving output quality, adding clinical safety guardrails, providing explainable AI features, enabling longitudinal analysis, and supporting patient-friendly communication across multiple languages.

The system must maintain its core functionality while adding six major enhancement areas that align with competition criteria: effective HAI-DEF model usage, problem domain relevance, impact potential, product feasibility, and execution quality.

## Glossary

- **Sentinel**: The oncology clinical auditor system being enhanced
- **MedGemma**: Medical language model accessed via Ollama (gemma3:4b)
- **LangGraph**: Agentic orchestration framework for workflow management
- **LanceDB**: Vector database for RAG (Retrieval-Augmented Generation)
- **RAG**: Retrieval-Augmented Generation for guideline retrieval
- **RECIST_1.1**: Response Evaluation Criteria In Solid Tumors version 1.1
- **NCCN**: National Comprehensive Cancer Network guidelines
- **Biomarker**: Measurable clinical indicator (tumor size, CEA, EGFR, etc.)
- **Safety_Auditor**: LangGraph node that validates extraction consistency
- **Clinical_Conflict**: Discrepancy detected between duplicate extractions
- **Vector_Store**: LanceDB-based storage for guideline embeddings
- **Digital_Twin**: Longitudinal simulation of patient biomarker trends
- **Patient_Translator**: Component that converts technical findings to patient-friendly language
- **Input_Parser**: Component that detects and parses multiple clinical data formats
- **Report_Generator**: Component that creates PDF and JSON exports
- **Confidence_Score**: Numerical indicator (0-100%) of extraction or retrieval certainty

## Requirements

### Requirement 1: Multi-Format Input System

**User Story:** As a clinical user, I want to upload various clinical document formats, so that I can audit cases regardless of their source system format.

#### Acceptance Criteria

1. WHEN a user uploads a PDF clinical report, THE Input_Parser SHALL extract structured clinical data including biomarkers
2. WHEN a user uploads an HL7 FHIR JSON file, THE Input_Parser SHALL parse FHIR resources and extract biomarkers
3. WHEN a user uploads plain text clinical notes, THE Input_Parser SHALL identify and extract biomarkers using NLP
4. WHEN a user uploads CSV lab results, THE Input_Parser SHALL parse tabular data and extract biomarkers
5. WHEN a user uploads a Synthea JSON file, THE Input_Parser SHALL maintain backward compatibility with existing parsing logic
6. WHEN an uploaded file format is ambiguous, THE Input_Parser SHALL auto-detect the format based on content structure
7. WHEN format detection fails, THE Input_Parser SHALL return a descriptive error message indicating supported formats
8. WHEN parsing any format, THE Input_Parser SHALL normalize extracted biomarkers to a common internal representation

### Requirement 2: Exportable Output System

**User Story:** As a clinical user, I want to export audit results as professional reports, so that I can share findings with colleagues and integrate with other systems.

#### Acceptance Criteria

1. WHEN an audit completes successfully, THE Report_Generator SHALL create a PDF report containing patient summary, biomarkers table, audit findings, referenced guidelines, trend visualizations, timestamp, and version information
2. WHEN generating a PDF report, THE Report_Generator SHALL include severity indicators for each audit finding
3. WHEN generating a PDF report, THE Report_Generator SHALL include page numbers for referenced guidelines
4. WHEN an audit completes successfully, THE Report_Generator SHALL create a JSON export containing all audit data in structured format
5. WHEN a user requests PDF download, THE Streamlit_UI SHALL provide the generated PDF file
6. WHEN a user requests JSON download, THE Streamlit_UI SHALL provide the generated JSON file
7. WHEN generating reports, THE Report_Generator SHALL include timestamp and system version information
8. WHEN biomarker trends exist, THE Report_Generator SHALL include trend visualizations in the PDF

### Requirement 3: Adversarial Hallucination Check

**User Story:** As a clinical safety officer, I want the system to validate its own extractions, so that I can trust the biomarker values used for clinical decisions.

#### Acceptance Criteria

1. WHEN biomarkers are extracted from clinical data, THE Safety_Auditor SHALL extract the same biomarkers twice using different prompts
2. WHEN comparing duplicate extractions, THE Safety_Auditor SHALL normalize units for comparison (e.g., 20mm equals 2.0cm)
3. WHEN duplicate extractions differ beyond acceptable tolerance, THE Safety_Auditor SHALL flag a Clinical_Conflict
4. WHEN a Clinical_Conflict is detected, THE Safety_Auditor SHALL halt the workflow and require human review
5. WHEN extracting biomarkers, THE Safety_Auditor SHALL assign confidence scores to each extracted value
6. WHEN a Clinical_Conflict occurs, THE Safety_Auditor SHALL log the conflict details including both extraction results and timestamp
7. WHEN no conflicts are detected, THE Safety_Auditor SHALL allow the workflow to proceed automatically
8. WHEN confidence scores are below a threshold, THE Safety_Auditor SHALL flag values for review even without conflicts

### Requirement 4: Explainable RAG System

**User Story:** As a clinical user, I want to see the source and confidence of guideline recommendations, so that I can verify the system's reasoning and trust its suggestions.

#### Acceptance Criteria

1. WHEN retrieving guidelines from Vector_Store, THE Vector_Store SHALL return source document name for each result
2. WHEN retrieving guidelines from Vector_Store, THE Vector_Store SHALL return page number for each result
3. WHEN retrieving guidelines from Vector_Store, THE Vector_Store SHALL return section title for each result
4. WHEN retrieving guidelines from Vector_Store, THE Vector_Store SHALL return guideline version and publication date for each result
5. WHEN retrieving guidelines from Vector_Store, THE Vector_Store SHALL calculate confidence score (0-100%) from vector distance
6. WHEN displaying recommendations, THE Streamlit_UI SHALL show visual confidence indicators for each guideline
7. WHEN a user clicks a guideline reference, THE Streamlit_UI SHALL display the original guideline excerpt
8. WHEN displaying multiple recommendations, THE Streamlit_UI SHALL sort them by confidence score in descending order

### Requirement 5: Longitudinal Simulation System

**User Story:** As a clinical user, I want to analyze biomarker trends over time and simulate future scenarios, so that I can make proactive treatment decisions.

#### Acceptance Criteria

1. WHEN multiple audit records exist for a patient, THE Digital_Twin SHALL calculate trend analysis for each biomarker
2. WHEN displaying trends, THE Streamlit_UI SHALL visualize biomarker changes over time using interactive Plotly charts
3. WHEN sufficient historical data exists, THE Digital_Twin SHALL predict next biomarker values using linear regression
4. WHEN predicting future values, THE Digital_Twin SHALL calculate projected timeline to critical thresholds
5. WHEN a user inputs a medication change, THE Digital_Twin SHALL allow simulation of projected biomarker impact
6. WHEN simulating scenarios, THE Digital_Twin SHALL use simple heuristics or linear models (not complex ML)
7. WHEN displaying simulations, THE Streamlit_UI SHALL clearly distinguish predicted values from actual measurements
8. WHEN insufficient data exists for trends, THE Digital_Twin SHALL display a message indicating minimum data requirements

### Requirement 6: Multi-Lingual Patient Translation

**User Story:** As a clinical user, I want to translate technical audit findings into patient-friendly language, so that patients can understand their results regardless of their language or medical literacy.

#### Acceptance Criteria

1. WHEN a user requests patient translation, THE Patient_Translator SHALL convert technical audit findings to 5th-grade reading level
2. WHEN translating findings, THE Patient_Translator SHALL support English, Spanish, Hindi, and Marathi languages
3. WHEN simplifying medical jargon, THE Patient_Translator SHALL use MedGemma to generate patient-friendly explanations
4. WHEN generating patient summaries, THE Patient_Translator SHALL maintain medical accuracy while simplifying language
5. WHEN creating patient reports, THE Report_Generator SHALL include visual aids such as icons for severity levels
6. WHEN a user requests patient translation, THE Report_Generator SHALL create a patient-friendly summary PDF
7. WHEN translating to non-English languages, THE Patient_Translator SHALL preserve medical terminology accuracy
8. WHEN displaying patient summaries, THE Streamlit_UI SHALL provide language selection options

### Requirement 7: System Performance and Reliability

**User Story:** As a system administrator, I want the enhanced system to maintain performance and reliability standards, so that it remains suitable for clinical production use.

#### Acceptance Criteria

1. WHEN processing any audit request, THE Sentinel SHALL complete the full workflow within 30 seconds
2. WHEN running in production, THE Sentinel SHALL operate entirely offline without internet connectivity
3. WHEN errors occur during processing, THE Sentinel SHALL provide descriptive error messages with recovery guidance
4. WHEN handling invalid input, THE Sentinel SHALL validate data and return specific validation errors
5. WHEN processing requests, THE Sentinel SHALL log all operations for audit trail purposes
6. WHEN system resources are constrained, THE Sentinel SHALL gracefully degrade functionality rather than crash
7. WHEN multiple users access the system, THE Streamlit_UI SHALL handle concurrent requests without data corruption
8. WHEN storing patient data, THE Sentinel SHALL ensure data privacy and security compliance

### Requirement 8: Backward Compatibility and Integration

**User Story:** As a system maintainer, I want the enhanced system to maintain compatibility with existing functionality, so that current users experience no disruption.

#### Acceptance Criteria

1. WHEN processing Synthea JSON files, THE Sentinel SHALL maintain identical behavior to the current system
2. WHEN existing biomarker extraction is performed, THE Sentinel SHALL produce the same results as before enhancements
3. WHEN RECIST 1.1 auditing is performed, THE Sentinel SHALL apply the same criteria as the current system
4. WHEN NCCN guidelines are retrieved, THE Sentinel SHALL use the existing LanceDB vector store
5. WHEN the Streamlit UI displays basic visualizations, THE Sentinel SHALL maintain the current display format
6. WHEN new features are disabled, THE Sentinel SHALL function identically to the pre-enhancement version
7. WHEN JSON exports are generated, THE Sentinel SHALL include all fields from the current system output
8. WHEN errors occur in new features, THE Sentinel SHALL not prevent core auditing functionality from working
