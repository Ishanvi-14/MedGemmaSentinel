# Test Fixtures

This directory contains sample clinical data files in various formats for testing the Sentinel multi-format input parser.

## Files

### sample_lab_report.txt
Plain text lab report with structured biomarker data. Used for testing text-based parsing.

### sample_fhir.json
HL7 FHIR Bundle containing Patient and Observation resources. Used for testing FHIR JSON parsing.

### sample_synthea.json
Synthea-generated FHIR Bundle. Used for testing backward compatibility with existing Synthea parser.

### sample_lab_results.csv
CSV file with tabular lab results. Used for testing CSV parsing with pandas.

### sample_clinical_notes.txt
Unstructured clinical notes with embedded biomarker information. Used for testing NLP-based extraction.

## Biomarker Data

All fixtures contain consistent biomarker data for patient P12345:
- Tumor Size: 3.2 cm (32 mm)
- CEA: 8.5 ng/mL
- EGFR: Positive (Exon 19 deletion)
- CA 19-9: 45 U/mL
- Date: 2024-01-15

This consistency allows for cross-format validation testing.
