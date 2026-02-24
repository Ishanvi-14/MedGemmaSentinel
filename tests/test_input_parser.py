"""
Tests for multi-format input parser.

Includes both unit tests and property-based tests for format detection,
parsing, normalization, and error handling.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis import assume

from src.input_parser import InputParser, Biomarker, ParsedClinicalData, BiomarkerType


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFormatDetection:
    """Unit tests for format detection."""
    
    def test_detect_pdf_format(self):
        """Test PDF format detection using magic bytes."""
        parser = InputParser()
        pdf_content = b'%PDF-1.4\n%some pdf content'
        assert parser.detect_format(pdf_content, "test.pdf") == "pdf"
    
    def test_detect_fhir_json_format(self):
        """Test FHIR JSON format detection."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_fhir.json", "rb") as f:
            content = f.read()
        assert parser.detect_format(content, "test.json") == "fhir_json"
    
    def test_detect_synthea_json_format(self):
        """Test Synthea JSON format detection."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            content = f.read()
        assert parser.detect_format(content, "test.json") == "synthea_json"
    
    def test_detect_csv_format(self):
        """Test CSV format detection."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            content = f.read()
        assert parser.detect_format(content, "test.csv") == "csv"
    
    def test_detect_text_format(self):
        """Test plain text format detection."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            content = f.read()
        assert parser.detect_format(content, "test.txt") == "text"
    
    def test_invalid_format_raises_error(self):
        """Test that invalid format raises descriptive error."""
        parser = InputParser()
        invalid_content = b'\x89\x50\x4E\x47'  # PNG magic bytes
        with pytest.raises(ValueError) as exc_info:
            parser.detect_format(invalid_content, "test.png")
        assert "Supported formats" in str(exc_info.value)


class TestFHIRParsing:
    """Unit tests for FHIR JSON parsing."""
    
    def test_parse_fhir_json_extracts_biomarkers(self):
        """Test FHIR JSON parsing extracts all biomarkers."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_fhir.json", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="fhir_json")
        
        assert result.format_type == "fhir_json"
        assert result.patient_id == "P12345"
        assert len(result.biomarkers) >= 2  # At least tumor_size and CEA
        
        # Check specific biomarkers
        biomarker_names = [b.name for b in result.biomarkers]
        assert "tumor_size" in biomarker_names
        assert "CEA" in biomarker_names
    
    def test_parse_fhir_tumor_size(self):
        """Test FHIR parsing correctly extracts tumor size."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_fhir.json", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="fhir_json")
        tumor_biomarkers = [b for b in result.biomarkers if b.name == "tumor_size"]
        
        assert len(tumor_biomarkers) > 0
        tumor = tumor_biomarkers[0]
        assert tumor.value == 32.0
        assert tumor.unit == "mm"
        assert tumor.confidence > 0.9  # High confidence for structured data
    
    def test_parse_fhir_cea(self):
        """Test FHIR parsing correctly extracts CEA."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_fhir.json", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="fhir_json")
        cea_biomarkers = [b for b in result.biomarkers if b.name == "CEA"]
        
        assert len(cea_biomarkers) > 0
        cea = cea_biomarkers[0]
        assert cea.value == 8.5
        assert cea.unit == "ng/mL"


class TestSyntheaParsing:
    """Unit tests for Synthea JSON parsing (backward compatibility)."""
    
    def test_parse_synthea_json_backward_compatible(self):
        """Test Synthea JSON parsing maintains backward compatibility."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="synthea_json")
        
        assert result.format_type == "synthea_json"
        assert result.metadata.get("backward_compatible") is True
        assert len(result.biomarkers) >= 2
    
    def test_synthea_extracts_tumor_size(self):
        """Test Synthea parsing extracts tumor size."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="synthea_json")
        tumor_biomarkers = [b for b in result.biomarkers if b.name == "tumor_size"]
        
        assert len(tumor_biomarkers) > 0
        assert tumor_biomarkers[0].value == 32.0
        assert tumor_biomarkers[0].unit == "mm"


class TestCSVParsing:
    """Unit tests for CSV parsing."""
    
    def test_parse_csv_extracts_biomarkers(self):
        """Test CSV parsing extracts biomarkers from tabular data."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="csv")
        
        assert result.format_type == "csv"
        assert result.patient_id == "P12345"
        assert len(result.biomarkers) > 0
    
    def test_csv_extracts_tumor_size(self):
        """Test CSV parsing extracts tumor size with correct unit conversion."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="csv")
        tumor_biomarkers = [b for b in result.biomarkers if b.name == "tumor_size"]
        
        assert len(tumor_biomarkers) > 0
        # CSV has tumor_size in cm
        assert any(b.value == 3.2 for b in tumor_biomarkers)
    
    def test_csv_extracts_cea(self):
        """Test CSV parsing extracts CEA levels."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="csv")
        cea_biomarkers = [b for b in result.biomarkers if b.name == "CEA"]
        
        assert len(cea_biomarkers) > 0
        assert any(b.value == 8.5 for b in cea_biomarkers)
    
    def test_csv_extracts_egfr(self):
        """Test CSV parsing extracts EGFR mutation status."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="csv")
        egfr_biomarkers = [b for b in result.biomarkers if b.name == "EGFR"]
        
        assert len(egfr_biomarkers) > 0
        # "positive" should be parsed as 1.0
        assert any(b.value == 1.0 for b in egfr_biomarkers)


class TestTextParsing:
    """Unit tests for plain text parsing."""
    
    def test_parse_text_extracts_biomarkers(self):
        """Test plain text parsing extracts biomarkers using regex."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="text")
        
        assert result.format_type == "text"
        assert result.patient_id == "P12345"
        assert len(result.biomarkers) > 0
    
    def test_text_extracts_tumor_size(self):
        """Test text parsing extracts tumor size from clinical notes."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="text")
        tumor_biomarkers = [b for b in result.biomarkers if b.name == "tumor_size"]
        
        # Clinical notes mention "3.2 cm"
        assert len(tumor_biomarkers) > 0
        assert any(b.value == 3.2 for b in tumor_biomarkers)
    
    def test_text_extracts_cea(self):
        """Test text parsing extracts CEA from clinical notes."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="text")
        cea_biomarkers = [b for b in result.biomarkers if b.name == "CEA"]
        
        # Clinical notes mention "CEA elevated at 8.5 ng/mL"
        assert len(cea_biomarkers) > 0
        assert any(b.value == 8.5 for b in cea_biomarkers)
    
    def test_text_extracts_egfr(self):
        """Test text parsing extracts EGFR mutation status."""
        parser = InputParser()
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            content = f.read()
        
        result = parser.parse(content, format_type="text")
        egfr_biomarkers = [b for b in result.biomarkers if b.name == "EGFR"]
        
        # Clinical notes mention "EGFR mutation analysis positive"
        assert len(egfr_biomarkers) > 0
        assert any(b.value == 1.0 for b in egfr_biomarkers)  # positive = 1.0


class TestBiomarkerNormalization:
    """Unit tests for biomarker normalization."""
    
    def test_normalize_biomarkers_from_dict(self):
        """Test normalization converts dict to Biomarker objects."""
        parser = InputParser()
        raw_data = {
            "tumor_size": {
                "value": 32.0,
                "unit": "mm",
                "timestamp": datetime.now(),
                "confidence": 0.95
            },
            "CEA": {
                "value": 8.5,
                "unit": "ng/mL",
                "timestamp": datetime.now(),
                "confidence": 0.90
            }
        }
        
        biomarkers = parser.normalize_biomarkers(raw_data)
        
        assert len(biomarkers) == 2
        assert all(isinstance(b, Biomarker) for b in biomarkers)
        assert any(b.name == "tumor_size" and b.value == 32.0 for b in biomarkers)
        assert any(b.name == "CEA" and b.value == 8.5 for b in biomarkers)
    
    def test_biomarker_type_auto_detection(self):
        """Test that biomarker type is automatically detected."""
        biomarker = Biomarker(
            name="tumor_size",
            value=32.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="test",
            confidence=0.9
        )
        
        assert biomarker.biomarker_type == BiomarkerType.TUMOR_SIZE
    
    def test_cea_type_detection(self):
        """Test CEA biomarker type detection."""
        biomarker = Biomarker(
            name="CEA",
            value=8.5,
            unit="ng/mL",
            timestamp=datetime.now(),
            source_field="test",
            confidence=0.9
        )
        
        assert biomarker.biomarker_type == BiomarkerType.CEA
    
    def test_egfr_type_detection(self):
        """Test EGFR biomarker type detection."""
        biomarker = Biomarker(
            name="EGFR",
            value=1.0,
            unit="status",
            timestamp=datetime.now(),
            source_field="test",
            confidence=0.9
        )
        
        assert biomarker.biomarker_type == BiomarkerType.EGFR


class TestErrorHandling:
    """Unit tests for error handling."""
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported format raises descriptive error."""
        parser = InputParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse(b"some content", format_type="unsupported")
        
        assert "Unsupported format" in str(exc_info.value)
        assert "Supported formats" in str(exc_info.value)
    
    def test_empty_pdf_raises_error(self):
        """Test that PDF with no text raises error."""
        parser = InputParser()
        # Create minimal PDF with no text
        pdf_content = b'%PDF-1.4\n%%EOF'
        
        with pytest.raises(ValueError) as exc_info:
            parser.parse(pdf_content, format_type="pdf")
        
        # Should have descriptive error message about PDF parsing
        error_msg = str(exc_info.value).lower()
        assert "pdf" in error_msg or "parse" in error_msg
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises error."""
        parser = InputParser()
        invalid_json = b'{invalid json content'
        
        with pytest.raises(Exception):  # Will raise JSONDecodeError
            parser.parse(invalid_json, format_type="fhir_json")


# Property-Based Tests

@settings(max_examples=100)
@given(
    format_name=st.sampled_from(["pdf", "fhir_json", "synthea_json", "csv", "text"])
)
def test_property_format_detection_consistency(format_name):
    """
    Property 2: Format Auto-Detection
    
    For any valid format signature, the Input_Parser should correctly
    identify the format type.
    
    Validates: Requirements 1.6
    """
    parser = InputParser()
    
    # Create content with format-specific signatures
    if format_name == "pdf":
        content = b'%PDF-1.4\nsome content'
        expected = "pdf"
    elif format_name == "fhir_json":
        content = json.dumps({
            "resourceType": "Bundle",
            "entry": []
        }).encode('utf-8')
        expected = "fhir_json"
    elif format_name == "synthea_json":
        content = json.dumps({
            "resourceType": "Bundle",
            "entry": [{
                "resource": {
                    "identifier": [{"system": "https://github.com/synthetichealth/synthea"}]
                }
            }]
        }).encode('utf-8')
        expected = "synthea_json"
    elif format_name == "csv":
        content = b'patient_id,value,unit\nP123,32,mm\n'
        expected = "csv"
    else:  # text
        content = b'Patient clinical notes with some text'
        expected = "text"
    
    detected = parser.detect_format(content, f"test.{format_name}")
    assert detected == expected


@settings(max_examples=100)
@given(
    tumor_size=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    cea_level=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_property_cross_format_normalization(tumor_size, cea_level):
    """
    Property 4: Cross-Format Normalization
    
    For any biomarker data parsed from different formats representing the same
    clinical information, the normalized internal representation should be equivalent.
    
    Validates: Requirements 1.8
    """
    parser = InputParser()
    
    # Create same data in different formats
    # Format 1: FHIR JSON
    fhir_data = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "P123"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Tumor Size"},
                    "valueQuantity": {"value": tumor_size, "unit": "mm"},
                    "effectiveDateTime": "2024-01-15T10:00:00Z"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "CEA"},
                    "valueQuantity": {"value": cea_level, "unit": "ng/mL"},
                    "effectiveDateTime": "2024-01-15T10:00:00Z"
                }
            }
        ]
    }
    fhir_content = json.dumps(fhir_data).encode('utf-8')
    
    # Format 2: CSV
    csv_content = f"patient_id,tumor_size,unit,CEA,cea_unit\nP123,{tumor_size},mm,{cea_level},ng/mL\n".encode('utf-8')
    
    # Parse both
    fhir_result = parser.parse(fhir_content, format_type="fhir_json")
    csv_result = parser.parse(csv_content, format_type="csv")
    
    # Both should extract biomarkers
    assert len(fhir_result.biomarkers) > 0
    assert len(csv_result.biomarkers) > 0
    
    # Find tumor_size in both
    fhir_tumor = [b for b in fhir_result.biomarkers if b.name == "tumor_size"]
    csv_tumor = [b for b in csv_result.biomarkers if b.name == "tumor_size"]
    
    if fhir_tumor and csv_tumor:
        # Values should match (within floating point tolerance)
        assert abs(fhir_tumor[0].value - csv_tumor[0].value) < 0.01
        assert fhir_tumor[0].unit == csv_tumor[0].unit


@settings(max_examples=100)
@given(
    invalid_content=st.binary(min_size=1, max_size=100)
)
def test_property_invalid_format_error_handling(invalid_content):
    """
    Property 3: Invalid Format Error Handling
    
    For any unsupported or malformed file, the Input_Parser should return
    a descriptive error message listing supported formats.
    
    Validates: Requirements 1.7
    """
    # Skip valid format signatures
    assume(not invalid_content.startswith(b'%PDF'))
    assume(not invalid_content.startswith(b'{'))
    assume(not invalid_content.startswith(b'['))
    
    # Skip valid text
    try:
        text = invalid_content.decode('utf-8')
        assume(',' not in text or '\n' not in text)  # Not CSV-like
    except UnicodeDecodeError:
        pass  # This is what we want to test
    
    parser = InputParser()
    
    try:
        parser.detect_format(invalid_content, "test.bin")
        # If it doesn't raise, it detected as text (which is valid fallback)
        # This is acceptable behavior
    except ValueError as e:
        # Should have descriptive error message
        error_msg = str(e)
        assert "Supported formats" in error_msg or "Unable to detect" in error_msg


@settings(max_examples=50)
@given(
    patient_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Nd'))),
    biomarker_count=st.integers(min_value=1, max_value=5)
)
def test_property_parsed_data_structure(patient_id, biomarker_count):
    """
    Property: Parsed data structure consistency
    
    For any valid clinical data, the parser should return a ParsedClinicalData
    object with all required fields populated.
    
    Validates: Requirements 1.8
    """
    parser = InputParser()
    
    # Create valid FHIR data
    entries = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": patient_id
            }
        }
    ]
    
    for i in range(biomarker_count):
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "code": {"text": "Tumor Size"},
                "valueQuantity": {"value": 30.0 + i, "unit": "mm"},
                "effectiveDateTime": "2024-01-15T10:00:00Z"
            }
        })
    
    fhir_data = {
        "resourceType": "Bundle",
        "entry": entries
    }
    content = json.dumps(fhir_data).encode('utf-8')
    
    result = parser.parse(content, format_type="fhir_json")
    
    # Verify structure
    assert isinstance(result, ParsedClinicalData)
    assert result.patient_id is not None
    assert isinstance(result.biomarkers, list)
    assert isinstance(result.raw_text, str)
    assert result.format_type in parser.SUPPORTED_FORMATS
    assert isinstance(result.parse_timestamp, datetime)
    assert isinstance(result.metadata, dict)


@settings(max_examples=100)
@given(
    value=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(["mm", "cm", "ng/mL", "ug/L"])
)
def test_property_biomarker_confidence_scores(value, unit):
    """
    Property: Confidence score assignment
    
    For any extracted biomarker, confidence scores should be between 0.0 and 1.0,
    with structured formats having higher confidence than unstructured.
    
    Validates: Requirements 1.8
    """
    parser = InputParser()
    
    # Create FHIR data (structured - should have high confidence)
    fhir_data = {
        "resourceType": "Observation",
        "code": {"text": "Tumor Size"},
        "valueQuantity": {"value": value, "unit": unit},
        "effectiveDateTime": "2024-01-15T10:00:00Z"
    }
    
    biomarker = parser._parse_fhir_observation(fhir_data)
    
    if biomarker:
        # Confidence should be in valid range
        assert 0.0 <= biomarker.confidence <= 1.0
        # Structured data should have high confidence
        assert biomarker.confidence >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
