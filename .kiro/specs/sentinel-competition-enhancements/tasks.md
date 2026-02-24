# Implementation Plan: Sentinel Competition Enhancements

## Overview

This implementation plan breaks down the Sentinel competition enhancements into discrete, incremental coding tasks. The plan follows a logical progression: first establishing the foundation with multi-format input parsing, then adding safety features, enhancing RAG capabilities, implementing longitudinal analysis, adding patient translation, and finally creating exportable reports. Each task builds on previous work and includes testing sub-tasks to validate functionality early.

The implementation uses Python with the existing tech stack: LangGraph for orchestration, LanceDB for vector storage, Streamlit for UI, Ollama/MedGemma for LLM capabilities, and Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project infrastructure and testing framework
  - Create directory structure for new components (input_parser/, safety_auditor/, digital_twin/, patient_translator/, report_generator/)
  - Install required dependencies (Hypothesis, ReportLab, PyPDF2/pdfplumber, pandas, scipy)
  - Set up pytest configuration with coverage reporting
  - Create test fixtures directory with sample files (PDF, FHIR JSON, CSV, text)
  - Configure Hypothesis settings for 100 iterations per property test
  - _Requirements: 7.2, 8.4_

- [x] 2. Implement multi-format input parser
  - [x] 2.1 Create InputParser class with format detection logic
    - Implement detect_format() method using magic bytes and content structure analysis
    - Support detection for PDF, FHIR JSON, Synthea JSON, CSV, and plain text
    - Return descriptive errors for unsupported formats
    - _Requirements: 1.6, 1.7_
  
  - [x] 2.2 Implement format-specific parsers
    - Create parse_pdf() using PyPDF2/pdfplumber with regex biomarker extraction
    - Create parse_fhir_json() navigating FHIR Observation/DiagnosticReport resources
    - Create parse_synthea_json() maintaining backward compatibility with existing logic
    - Create parse_csv() using pandas with column mapping
    - Create parse_text() using MedGemma with structured prompts
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 2.3 Implement biomarker normalization
    - Create normalize_biomarkers() to convert format-specific data to common Biomarker objects
    - Ensure all biomarkers have name, value, unit, timestamp, source_field, and confidence
    - _Requirements: 1.8_
  
  - [x] 2.4 Write property test for format detection
    - **Property 2: Format Auto-Detection**
    - **Validates: Requirements 1.6**
  
  - [x] 2.5 Write property test for cross-format normalization
    - **Property 4: Cross-Format Normalization**
    - **Validates: Requirements 1.8**
  
  - [x] 2.6 Write property test for invalid format error handling
    - **Property 3: Invalid Format Error Handling**
    - **Validates: Requirements 1.7**
  
  - [x] 2.7 Write unit tests for each format parser
    - Test PDF parsing with sample lab report
    - Test FHIR JSON parsing with sample FHIR bundle
    - Test Synthea JSON backward compatibility
    - Test CSV parsing with sample lab results
    - Test plain text extraction with clinical notes
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Integrate input parser into LangGraph workflow
  - [x] 3.1 Create parse_input_node for LangGraph
    - Implement node function that takes AuditState and returns updated state
    - Handle file upload from Streamlit UI
    - Update AuditState with ParsedClinicalData
    - Add error handling with descriptive messages
    - _Requirements: 1.1-1.8, 7.3, 7.4_
  
  - [x] 3.2 Update Streamlit UI for multi-format upload
    - Modify file uploader to accept multiple formats (.pdf, .json, .csv, .txt)
    - Display detected format to user
    - Show parsing errors with recovery guidance
    - _Requirements: 1.6, 7.3_
  
  - [x] 3.3 Add parse_input_node to workflow graph
    - Insert node at beginning of workflow before extract_biomarkers
    - Update workflow edges: parse_input → extract_biomarkers
    - _Requirements: 1.1-1.8_

- [x] 4. Checkpoint - Ensure multi-format parsing works
  - Test uploading each format type through UI
  - Verify biomarkers are extracted correctly
  - Ensure backward compatibility with existing Synthea JSON files
  - Ask the user if questions arise


- [x] 5. Implement Safety Auditor component
  - [x] 5.1 Create SafetyAuditor class with dual extraction
    - Implement extract_with_prompt_a() using direct biomarker extraction prompt
    - Implement extract_with_prompt_b() using contextual clinical specialist prompt
    - Both methods should call MedGemma via Ollama and parse structured responses
    - _Requirements: 3.1_
  
  - [x] 5.2 Implement unit normalization and comparison logic
    - Create normalize_units() for common unit conversions (mm↔cm, ng/mL↔g/L, etc.)
    - Create compare_extractions() to identify conflicts with tolerance thresholds
    - Implement calculate_confidence() based on extraction agreement
    - _Requirements: 3.2, 3.5_
  
  - [x] 5.3 Implement conflict detection and logging
    - Create ComparisonResult and Conflict dataclasses
    - Implement conflict flagging when discrepancy exceeds 10% for continuous values
    - Implement conflict logging to audit_log database table
    - Add low-confidence flagging (threshold < 0.6)
    - _Requirements: 3.3, 3.6, 3.8_
  
  - [x] 5.4 Write property test for unit normalization
    - **Property 9: Unit Normalization for Comparison**
    - **Validates: Requirements 3.2**
  
  - [x] 5.5 Write property test for conflict detection
    - **Property 10: Conflict Detection and Flagging**
    - **Validates: Requirements 3.3**
  
  - [x] 5.6 Write property test for confidence score assignment
    - **Property 12: Confidence Score Assignment**
    - **Validates: Requirements 3.5**
  
  - [x] 5.7 Write unit tests for safety auditor
    - Test dual extraction with known clinical data
    - Test unit conversion edge cases (zero, very large values)
    - Test conflict detection with matching and mismatched extractions
    - _Requirements: 3.1, 3.2, 3.3_

- [ ] 6. Integrate Safety Auditor into LangGraph workflow
  - [x] 6.1 Create safety_auditor_node for LangGraph
    - Implement node that performs dual extraction and comparison
    - Update AuditState with safety_result (ComparisonResult)
    - Set requires_human_review flag based on conflicts
    - _Requirements: 3.1-3.8_
  
  - [x] 6.2 Implement conditional routing based on safety check
    - Create route_safety_check() function for conditional edge
    - Route to human review UI if conflicts detected
    - Route to continue workflow if no conflicts
    - _Requirements: 3.4, 3.7_
  
  - [x] 6.3 Update Streamlit UI for human review
    - Create review panel showing both extractions side-by-side
    - Display conflicts with discrepancy percentages
    - Add "Approve" and "Reject" buttons for human decision
    - _Requirements: 3.4_
  
  - [x] 6.4 Add safety_auditor_node to workflow graph
    - Insert node after extract_biomarkers
    - Add conditional edge: safety_check → (human_review | retrieve_guidelines)
    - _Requirements: 3.1-3.8_
  
  - [-] 6.5 Write property test for workflow routing
    - **Property 11: Workflow Routing Based on Conflicts**
    - **Validates: Requirements 3.4, 3.7**

- [ ] 7. Checkpoint - Ensure safety auditor catches conflicts
  - Test with clinical data that should produce conflicts
  - Test with clinical data that should match
  - Verify human review UI appears for conflicts
  - Verify workflow continues automatically when no conflicts
  - Ask the user if questions arise


- [ ] 8. Enhance Vector Store with metadata support
  - [ ] 8.1 Update LanceDB schema to include metadata fields
    - Add columns: source_document, page_number, section_title, guideline_version, publication_date
    - Create migration script to update existing vector store
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [ ] 8.2 Create EnhancedVectorStore class
    - Implement search_with_metadata() returning GuidelineResult objects with full metadata
    - Implement calculate_confidence_score() using inverse exponential of vector distance
    - Ensure confidence scores are in 0-100% range
    - _Requirements: 4.1-4.5_
  
  - [ ] 8.3 Update guideline ingestion to store metadata
    - Modify ingestion script to extract metadata from guideline PDFs
    - Store metadata alongside embeddings in LanceDB
    - _Requirements: 4.1-4.4_
  
  - [ ] 8.4 Write property test for metadata completeness
    - **Property 15: Guideline Metadata Completeness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
  
  - [ ] 8.5 Write property test for confidence score range
    - **Property 16: Confidence Score Range Validity**
    - **Validates: Requirements 4.5**
  
  - [ ] 8.6 Write unit tests for enhanced vector store
    - Test metadata retrieval with sample queries
    - Test confidence score calculation with various distances
    - Test backward compatibility with existing vector store
    - _Requirements: 4.1-4.5, 8.4_

- [ ] 9. Update UI to display explainable RAG results
  - [ ] 9.1 Enhance guideline display with metadata
    - Show source document, page number, section title for each guideline
    - Display confidence score with visual indicator (progress bar or color coding)
    - Add "View Source" button to show original guideline excerpt
    - _Requirements: 4.1-4.5_
  
  - [ ] 9.2 Implement guideline sorting by confidence
    - Sort recommendations in descending order by confidence score
    - Add visual separator between high/medium/low confidence results
    - _Requirements: 4.8_
  
  - [ ] 9.3 Create guideline excerpt modal
    - Implement popup/modal to display full guideline text when user clicks reference
    - Include metadata in modal header
    - _Requirements: 4.7_
  
  - [ ] 9.4 Write property test for recommendation sorting
    - **Property 17: Recommendation Sorting by Confidence**
    - **Validates: Requirements 4.8**

- [ ] 10. Checkpoint - Ensure explainable RAG works
  - Test guideline retrieval shows all metadata
  - Verify confidence scores are calculated correctly
  - Test sorting by confidence
  - Test "View Source" functionality
  - Ask the user if questions arise


- [ ] 11. Implement Digital Twin component for longitudinal analysis
  - [ ] 11.1 Create patient history database schema
    - Create patient_history table in SQLite with fields: record_id, patient_id, audit_id, biomarker_name, biomarker_value, biomarker_unit, measurement_date, confidence
    - Create indexes for efficient querying by patient_id and biomarker_name
    - _Requirements: 5.1_
  
  - [ ] 11.2 Implement DigitalTwin class with trend calculation
    - Implement load_patient_history() to query historical audit records
    - Implement calculate_trends() using scipy.stats.linregress for linear regression
    - Create TrendAnalysis dataclass with slope, intercept, r_squared, data_points
    - Flag trends with insufficient data (< 3 points) or low R² (< 0.5)
    - _Requirements: 5.1, 5.8_
  
  - [ ] 11.3 Implement prediction and threshold calculation
    - Implement predict_next_value() to extrapolate future biomarker values
    - Implement calculate_threshold_timeline() to project when critical thresholds are reached
    - Include confidence intervals for predictions
    - _Requirements: 5.3, 5.4_
  
  - [ ] 11.4 Implement intervention simulation
    - Create Intervention dataclass with type and expected_impact
    - Define INTERVENTION_IMPACTS dictionary with heuristics for common treatments
    - Implement simulate_intervention() to modify trend slope based on intervention type
    - _Requirements: 5.5_
  
  - [ ] 11.5 Write property test for trend calculation
    - **Property 18: Trend Calculation for Patient History**
    - **Validates: Requirements 5.1**
  
  - [ ] 11.6 Write property test for prediction with sufficient data
    - **Property 19: Prediction with Sufficient Data**
    - **Validates: Requirements 5.3**
  
  - [ ] 11.7 Write property test for threshold timeline calculation
    - **Property 20: Threshold Timeline Calculation**
    - **Validates: Requirements 5.4**
  
  - [ ] 11.8 Write property test for intervention simulation
    - **Property 21: Intervention Impact Simulation**
    - **Validates: Requirements 5.5**
  
  - [ ] 11.9 Write unit tests for digital twin
    - Test trend calculation with known linear data
    - Test insufficient data handling
    - Test prediction accuracy with synthetic patient histories
    - Test intervention simulation with various treatment types
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.8_

- [ ] 12. Integrate Digital Twin into workflow and UI
  - [ ] 12.1 Create digital_twin_node for LangGraph
    - Implement node that loads patient history and calculates trends
    - Update AuditState with trends dictionary
    - Handle cases with insufficient data gracefully
    - _Requirements: 5.1, 5.8_
  
  - [ ] 12.2 Create trend visualization UI with Plotly
    - Implement interactive line charts showing biomarker values over time
    - Add predicted values as dashed lines with confidence intervals
    - Add threshold lines with projected intersection dates
    - Clearly distinguish actual vs predicted values
    - _Requirements: 5.2, 5.7_
  
  - [ ] 12.3 Create intervention simulation UI
    - Add dropdown for intervention type selection
    - Add "Simulate" button to show projected impact
    - Display modified trend prediction on chart
    - _Requirements: 5.5_
  
  - [ ] 12.4 Add digital_twin_node to workflow graph
    - Insert node after audit_recist
    - Add edge: audit_recist → analyze_trends
    - _Requirements: 5.1-5.8_

- [ ] 13. Checkpoint - Ensure longitudinal analysis works
  - Test trend calculation with multi-visit patient data
  - Verify predictions are reasonable
  - Test threshold timeline calculations
  - Test intervention simulations
  - Verify insufficient data messages appear correctly
  - Ask the user if questions arise


- [ ] 14. Implement Patient Translator component
  - [x] 14.1 Create PatientTranslator class with simplification logic
    - Implement simplify_findings() using MedGemma with reading-level-specific prompts
    - Create SIMPLIFICATION_PROMPT template emphasizing 5th-grade reading level
    - Implement validate_medical_accuracy() to ensure critical terms are preserved
    - Calculate readability scores using textstat library (Flesch-Kincaid grade)
    - _Requirements: 6.1, 6.4_
  
  - [x] 14.2 Implement multi-lingual translation
    - Implement translate_to_language() supporting English, Spanish, Hindi, Marathi
    - Use MedGemma with language-specific prompts
    - Preserve medical terminology in original language (in parentheses)
    - Create TranslatedFinding dataclass
    - _Requirements: 6.2, 6.7_
  
  - [x] 14.3 Implement visual aids for severity
    - Implement add_visual_aids() to add emoji/icon indicators
    - Map severity levels: LOW → 🟢, MEDIUM → 🟡, HIGH → 🔴
    - _Requirements: 6.5_
  
  - [-] 14.4 Write property test for reading level simplification
    - **Property 23: Reading Level Simplification**
    - **Validates: Requirements 6.1**
  
  - [ ] 14.5 Write property test for medical accuracy preservation
    - **Property 24: Medical Accuracy Preservation**
    - **Validates: Requirements 6.4**
  
  - [ ] 14.6 Write property test for terminology preservation in translation
    - **Property 27: Medical Terminology Preservation in Translation**
    - **Validates: Requirements 6.7**
  
  - [ ] 14.7 Write unit tests for patient translator
    - Test simplification with complex medical findings
    - Test translation for each supported language
    - Test visual aid assignment
    - Test medical accuracy validation
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.7_

- [ ] 15. Integrate Patient Translator into workflow and UI
  - [ ] 15.1 Create patient_translator_node for LangGraph
    - Implement node that translates audit findings
    - Update AuditState with translated_findings
    - Make translation optional (conditional edge based on user request)
    - _Requirements: 6.1-6.8_
  
  - [ ] 15.2 Add language selection to Streamlit UI
    - Create language dropdown with options: English, Spanish, Hindi, Marathi
    - Add "Translate for Patient" checkbox
    - Display translated findings in separate section
    - _Requirements: 6.8_
  
  - [ ] 15.3 Add patient_translator_node to workflow graph
    - Insert node after analyze_trends
    - Add conditional edge based on translation request
    - _Requirements: 6.1-6.8_

- [ ] 16. Checkpoint - Ensure patient translation works
  - Test simplification produces appropriate reading level
  - Test translation for all four languages
  - Verify medical accuracy is preserved
  - Test visual severity indicators
  - Ask the user if questions arise


- [ ] 17. Implement Report Generator component
  - [ ] 17.1 Create ReportGenerator class with PDF generation
    - Implement generate_pdf_report() using ReportLab
    - Create PDF sections: header, patient summary, biomarkers table, audit findings, guidelines, trends, footer
    - Include severity indicators (colored circles or icons)
    - Include page numbers for guideline references
    - Include confidence scores for biomarkers and guidelines
    - Add timestamp and Sentinel version in footer
    - _Requirements: 2.1, 2.2, 2.3, 2.7_
  
  - [ ] 17.2 Implement trend visualization embedding in PDF
    - Use matplotlib to generate static charts from Plotly data
    - Embed charts in PDF report
    - Only include trends section if trend data exists
    - _Requirements: 2.8_
  
  - [ ] 17.3 Implement JSON export
    - Implement generate_json_export() following defined schema
    - Include all audit data: metadata, biomarkers, findings, guidelines, trends, safety_audit
    - Ensure backward compatibility with existing JSON fields
    - _Requirements: 2.4, 8.7_
  
  - [ ] 17.4 Implement patient-friendly PDF generation
    - Implement create_patient_pdf() with simplified language
    - Use larger fonts and more whitespace
    - Include visual severity indicators (emoji/icons)
    - Support multi-lingual content
    - _Requirements: 6.5, 6.6_
  
  - [ ] 17.5 Write property test for PDF report completeness
    - **Property 5: PDF Report Completeness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.7**
  
  - [ ] 17.6 Write property test for JSON export completeness
    - **Property 6: JSON Export Completeness**
    - **Validates: Requirements 2.4**
  
  - [ ] 17.7 Write property test for trend visualization inclusion
    - **Property 7: Trend Visualization Inclusion**
    - **Validates: Requirements 2.8**
  
  - [ ] 17.8 Write property test for patient PDF generation
    - **Property 26: Patient PDF Generation**
    - **Validates: Requirements 6.6**
  
  - [ ] 17.9 Write unit tests for report generator
    - Test PDF generation with complete audit results
    - Test PDF generation without trends
    - Test JSON export schema compliance
    - Test patient PDF with translated findings
    - _Requirements: 2.1-2.8, 6.6_

- [ ] 18. Integrate Report Generator into workflow and UI
  - [ ] 18.1 Create report_generator_node for LangGraph
    - Implement node that generates both PDF and JSON reports
    - Update AuditState with pdf_report and json_export
    - Handle errors gracefully (e.g., PDF fails but JSON succeeds)
    - _Requirements: 2.1-2.8_
  
  - [ ] 18.2 Add download buttons to Streamlit UI
    - Add "Download PDF Report" button with file download
    - Add "Download JSON Export" button with file download
    - Add "Download Patient Summary" button (if translation was requested)
    - Display success/error messages for downloads
    - _Requirements: 2.5, 2.6_
  
  - [ ] 18.3 Add report_generator_node to workflow graph
    - Insert node at end of workflow
    - Add edges: translate_patient → generate_reports, analyze_trends → generate_reports
    - _Requirements: 2.1-2.8_

- [ ] 19. Checkpoint - Ensure report generation works
  - Test PDF report generation with complete audit
  - Test JSON export with complete audit
  - Test patient PDF generation with translation
  - Verify downloads work from UI
  - Verify all required sections are present
  - Ask the user if questions arise


- [ ] 20. Implement error handling and logging
  - [ ] 20.1 Create ErrorHandler class with context-aware recovery
    - Implement handle_error() with error type routing
    - Implement specific handlers for each error category (parsing, extraction, RAG, etc.)
    - Create ErrorResponse dataclass with recovery steps
    - _Requirements: 7.3, 7.4_
  
  - [ ] 20.2 Add comprehensive logging throughout workflow
    - Add logging to all LangGraph nodes
    - Log all operations to audit_log database table
    - Include timestamps, operation types, and outcomes
    - Log conflicts and safety audit results
    - _Requirements: 7.5, 3.6_
  
  - [ ] 20.3 Implement graceful degradation
    - Wrap enhancement features in try-except blocks
    - Allow core functionality to continue if enhancements fail
    - Log warnings for non-critical failures
    - _Requirements: 7.6, 8.8_
  
  - [ ] 20.4 Write property test for descriptive error messages
    - **Property 28: Descriptive Error Messages**
    - **Validates: Requirements 7.3**
  
  - [ ] 20.5 Write property test for input validation errors
    - **Property 29: Input Validation Errors**
    - **Validates: Requirements 7.4**
  
  - [ ] 20.6 Write property test for operation logging
    - **Property 30: Operation Logging**
    - **Validates: Requirements 7.5**
  
  - [ ] 20.7 Write property test for fault isolation
    - **Property 34: Fault Isolation**
    - **Validates: Requirements 8.8**

- [ ] 21. Implement backward compatibility validation
  - [ ] 21.1 Create compatibility test suite
    - Collect existing Synthea JSON test cases
    - Run test cases through enhanced system
    - Compare outputs with pre-enhancement baseline
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 21.2 Implement feature flags
    - Create FEATURES configuration dictionary
    - Add conditional logic to enable/disable enhancements
    - Test system with all features disabled
    - _Requirements: 8.6_
  
  - [ ] 21.3 Write property test for existing functionality preservation
    - **Property 31: Existing Functionality Preservation**
    - **Validates: Requirements 8.1, 8.2, 8.3**
  
  - [ ] 21.4 Write property test for JSON schema compatibility
    - **Property 32: JSON Schema Compatibility**
    - **Validates: Requirements 8.7**
  
  - [ ] 21.5 Write property test for feature-disabled compatibility
    - **Property 33: Feature-Disabled Compatibility**
    - **Validates: Requirements 8.6**

- [ ] 22. Checkpoint - Ensure backward compatibility
  - Run all existing test cases through enhanced system
  - Verify outputs match pre-enhancement baseline
  - Test with all features disabled
  - Verify JSON exports contain all legacy fields
  - Ask the user if questions arise


- [ ] 23. Performance optimization and validation
  - [ ] 23.1 Implement caching for MedGemma embeddings
    - Create embedding cache using LRU cache or simple dict
    - Cache embeddings for repeated queries
    - Measure performance improvement
    - _Requirements: 7.1_
  
  - [ ] 23.2 Optimize LangGraph workflow execution
    - Identify opportunities for parallel node execution
    - Optimize database queries with proper indexing
    - Profile workflow to identify bottlenecks
    - _Requirements: 7.1_
  
  - [ ] 23.3 Add performance benchmarking
    - Create benchmark suite measuring end-to-end execution time
    - Test with various input sizes and formats
    - Ensure all tests complete within 30-second budget
    - _Requirements: 7.1_
  
  - [ ] 23.4 Write unit tests for performance requirements
    - Test that typical audit completes within 30 seconds
    - Test with maximum file size (50MB)
    - Test with maximum patient history (20 data points)
    - _Requirements: 7.1_

- [ ] 24. Create comprehensive integration tests
  - [ ] 24.1 Write end-to-end workflow tests
    - Test complete workflow from PDF upload to report download
    - Test complete workflow from FHIR JSON to patient translation
    - Test workflow with conflicts requiring human review
    - Test workflow with insufficient trend data
    - _Requirements: 1.1-8.8_
  
  - [ ] 24.2 Write UI integration tests
    - Test file upload for all formats
    - Test human review UI interaction
    - Test language selection and translation
    - Test report downloads
    - Test trend visualization interactions
    - _Requirements: 2.5, 2.6, 3.4, 4.6, 4.7, 5.2, 6.8_
  
  - [ ] 24.3 Write offline operation validation test
    - Verify no network calls are made during execution
    - Test with network disabled
    - _Requirements: 7.2_

- [ ] 25. Documentation and deployment preparation
  - [ ] 25.1 Update README with new features
    - Document all six enhancement areas
    - Add usage examples for each feature
    - Update installation instructions
    - _Requirements: 1.1-8.8_
  
  - [ ] 25.2 Create configuration guide
    - Document environment variables
    - Document feature flags
    - Document performance tuning options
    - _Requirements: 7.1-7.8_
  
  - [ ] 25.3 Create deployment checklist
    - List system requirements
    - List installation steps
    - List health check procedures
    - _Requirements: 7.1-7.8_
  
  - [ ] 25.4 Update requirements.txt
    - Add all new dependencies with versions
    - Test fresh installation from requirements.txt
    - _Requirements: 1.1-8.8_

- [ ] 26. Final checkpoint - Complete system validation
  - Run full test suite (unit + property + integration)
  - Verify all 34 correctness properties pass
  - Test complete workflow with all features enabled
  - Verify performance meets 30-second requirement
  - Test backward compatibility with existing data
  - Verify offline operation
  - Ask the user if questions arise

## Notes

- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout implementation
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples, edge cases, and integration points
- The implementation follows a logical progression: input → safety → RAG → trends → translation → reports
- All enhancements integrate into the existing LangGraph workflow as new or augmented nodes
- Backward compatibility is maintained throughout with feature flags and compatibility tests
- All testing tasks are required to ensure comprehensive validation and production readiness
