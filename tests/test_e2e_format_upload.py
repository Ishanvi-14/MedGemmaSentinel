"""
End-to-end tests simulating UI upload for each format type.

These tests verify that the complete workflow from file upload through
parsing works correctly for all supported formats.
"""

import pytest
from pathlib import Path
from src.agents import SentinelOrchestrator


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestEndToEndFormatUpload:
    """End-to-end tests simulating UI file upload for each format."""
    
    def test_e2e_pdf_upload(self):
        """Test end-to-end workflow with PDF upload."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Note: We don't have a real PDF fixture, so we'll skip this test
        # In production, you would have a sample PDF lab report
        pytest.skip("PDF fixture not available - would require real PDF file")
    
    def test_e2e_fhir_json_upload(self):
        """Test end-to-end workflow with FHIR JSON upload."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Load FHIR JSON fixture
        with open(FIXTURES_DIR / "sample_fhir.json", "rb") as f:
            file_content = f.read()
        
        # Simulate UI upload
        result = graph.invoke({
            "file_content": file_content,
            "filename": "sample_fhir.json",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify successful parsing
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Should not have errors
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Should have extracted data
        assert parsed_data["format_type"] == "fhir_json"
        assert parsed_data["patient_id"] == "P12345"
        assert len(parsed_data["biomarkers"]) >= 2
        
        # Verify biomarkers were extracted
        biomarker_names = [b["name"] for b in parsed_data["biomarkers"]]
        assert "tumor_size" in biomarker_names
        assert "CEA" in biomarker_names
        
        # Verify history was updated
        assert result["history"] != ""
        # History should contain biomarker information
        assert "tumor_size" in result["history"] or "CEA" in result["history"]
    
    def test_e2e_synthea_json_upload(self):
        """Test end-to-end workflow with Synthea JSON upload (backward compatibility)."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Load Synthea JSON fixture
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            file_content = f.read()
        
        # Simulate UI upload
        result = graph.invoke({
            "file_content": file_content,
            "filename": "sample_synthea.json",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify successful parsing
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Should not have errors
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Should have extracted data
        assert parsed_data["format_type"] == "synthea_json"
        assert parsed_data["patient_id"] is not None
        assert len(parsed_data["biomarkers"]) >= 2
        
        # Verify backward compatibility flag
        assert parsed_data["metadata"].get("backward_compatible") is True
        
        # Verify history was updated
        assert result["history"] != ""
    
    def test_e2e_csv_upload(self):
        """Test end-to-end workflow with CSV upload."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Load CSV fixture
        with open(FIXTURES_DIR / "sample_lab_results.csv", "rb") as f:
            file_content = f.read()
        
        # Simulate UI upload
        result = graph.invoke({
            "file_content": file_content,
            "filename": "sample_lab_results.csv",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify successful parsing
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Should not have errors
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Should have extracted data
        assert parsed_data["format_type"] == "csv"
        assert parsed_data["patient_id"] == "P12345"
        assert len(parsed_data["biomarkers"]) >= 3
        
        # Verify biomarkers were extracted
        biomarker_names = [b["name"] for b in parsed_data["biomarkers"]]
        assert "tumor_size" in biomarker_names
        assert "CEA" in biomarker_names
        assert "EGFR" in biomarker_names
        
        # Verify history was updated
        assert result["history"] != ""
        # History should contain biomarker information
        assert "tumor_size" in result["history"] or "CEA" in result["history"]
    
    def test_e2e_text_upload(self):
        """Test end-to-end workflow with plain text upload."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Load text fixture
        with open(FIXTURES_DIR / "sample_clinical_notes.txt", "rb") as f:
            file_content = f.read()
        
        # Simulate UI upload
        result = graph.invoke({
            "file_content": file_content,
            "filename": "sample_clinical_notes.txt",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify successful parsing
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Should not have errors
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Should have extracted data
        assert parsed_data["format_type"] == "text"
        assert parsed_data["patient_id"] == "P12345"
        assert len(parsed_data["biomarkers"]) >= 1
        
        # Verify at least tumor_size was extracted
        biomarker_names = [b["name"] for b in parsed_data["biomarkers"]]
        assert "tumor_size" in biomarker_names
        
        # Verify history was updated
        assert result["history"] != ""
    
    def test_e2e_invalid_format_error(self):
        """Test end-to-end workflow with invalid format shows proper error."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Create invalid binary content
        file_content = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'  # PNG magic bytes
        
        # Simulate UI upload
        result = graph.invoke({
            "file_content": file_content,
            "filename": "invalid.png",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify error is captured
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Should have error
        assert "error" in parsed_data
        assert parsed_data["error"] is not None
        
        # Should have recovery steps
        assert "recovery_steps" in parsed_data
        assert len(parsed_data["recovery_steps"]) > 0
        
        # Error message should mention supported formats
        error_msg = parsed_data["error"]
        assert "format" in error_msg.lower() or "supported" in error_msg.lower()
    
    def test_e2e_format_detection_display(self):
        """Test that format detection works as shown in UI."""
        from src.input_parser.parser import InputParser
        
        parser = InputParser()
        
        # Test each format
        test_cases = [
            ("sample_fhir.json", "fhir_json"),
            ("sample_synthea.json", "synthea_json"),
            ("sample_lab_results.csv", "csv"),
            ("sample_clinical_notes.txt", "text"),
        ]
        
        for filename, expected_format in test_cases:
            with open(FIXTURES_DIR / filename, "rb") as f:
                file_content = f.read()
            
            detected = parser.detect_format(file_content, filename)
            assert detected == expected_format, f"Failed to detect {filename} as {expected_format}"


class TestBackwardCompatibility:
    """Test backward compatibility with existing Synthea JSON files."""
    
    def test_synthea_json_backward_compatibility(self):
        """Verify Synthea JSON processing maintains backward compatibility."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # Load existing Synthea JSON
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            file_content = f.read()
        
        # Process through new workflow
        result = graph.invoke({
            "file_content": file_content,
            "filename": "sample_synthea.json",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        })
        
        # Verify successful processing
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Verify backward compatibility metadata
        assert parsed_data["metadata"].get("backward_compatible") is True
        
        # Verify biomarkers extracted
        assert len(parsed_data["biomarkers"]) >= 2
        
        # Verify expected biomarkers present
        biomarker_names = [b["name"] for b in parsed_data["biomarkers"]]
        assert "tumor_size" in biomarker_names
        
        # Verify values match expected (from fixture)
        tumor_biomarkers = [b for b in parsed_data["biomarkers"] if b["name"] == "tumor_size"]
        assert len(tumor_biomarkers) > 0
        assert tumor_biomarkers[0]["value"] == 32.0
        assert tumor_biomarkers[0]["unit"] == "mm"
    
    def test_synthea_json_produces_same_results(self):
        """Verify enhanced system produces same results as before for Synthea JSON."""
        from src.input_parser.parser import InputParser
        
        parser = InputParser()
        
        # Load Synthea JSON
        with open(FIXTURES_DIR / "sample_synthea.json", "rb") as f:
            file_content = f.read()
        
        # Parse using new parser
        result = parser.parse(file_content, format_type="synthea_json")
        
        # Verify expected biomarkers are present with correct values
        # These values should match what the old system would extract
        biomarker_dict = {b.name: b for b in result.biomarkers}
        
        # Check tumor_size
        assert "tumor_size" in biomarker_dict
        assert biomarker_dict["tumor_size"].value == 32.0
        assert biomarker_dict["tumor_size"].unit == "mm"
        
        # Check CEA
        assert "CEA" in biomarker_dict
        assert biomarker_dict["CEA"].value == 8.5
        assert biomarker_dict["CEA"].unit == "ng/mL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
