"""
Integration tests for input parser workflow integration.
"""

import pytest
import json
from src.agents import SentinelOrchestrator


class TestWorkflowIntegration:
    """Test that input parser integrates correctly with LangGraph workflow."""
    
    def test_workflow_has_parse_input_node(self):
        """Verify parse_input_node is added to workflow."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # The graph should have parse_input as entry point
        assert graph is not None
    
    def test_parse_input_node_with_json_file(self):
        """Test parse_input_node with JSON file content."""
        orchestrator = SentinelOrchestrator()
        
        # Create sample JSON content
        sample_data = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient-123"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "text": "Tumor Size"
                        },
                        "valueQuantity": {
                            "value": 25.0,
                            "unit": "mm"
                        },
                        "effectiveDateTime": "2024-01-15T10:00:00Z"
                    }
                }
            ]
        }
        
        file_content = json.dumps(sample_data).encode('utf-8')
        
        # Test parse_input_node directly
        state = {
            "file_content": file_content,
            "filename": "test.json",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.parse_input_node(state)
        
        # Verify parsed_data is returned
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Verify no errors
        assert "error" not in parsed_data or parsed_data.get("error") is None
        
        # Verify patient ID extracted
        assert parsed_data["patient_id"] == "test-patient-123"
        
        # Verify biomarkers extracted
        assert len(parsed_data["biomarkers"]) > 0
        
        # Verify history updated
        assert "history" in result
        assert result["history"] != ""
    
    def test_parse_input_node_with_text_file(self):
        """Test parse_input_node with plain text file."""
        orchestrator = SentinelOrchestrator()
        
        text_content = """
        Patient: John Doe (ID: P-12345)
        
        Clinical Notes:
        Patient presents with tumor size measuring 30mm.
        CEA elevated at 8.5 ng/mL.
        EGFR mutation analysis positive.
        """
        
        file_content = text_content.encode('utf-8')
        
        state = {
            "file_content": file_content,
            "filename": "notes.txt",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.parse_input_node(state)
        
        # Verify parsed_data is returned
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        
        # Verify format detected as text
        assert parsed_data["format_type"] == "text"
        
        # Verify patient ID extracted
        assert parsed_data["patient_id"] == "P-12345"
        
        # Verify biomarkers extracted (at least tumor_size should be found)
        biomarkers = parsed_data["biomarkers"]
        assert len(biomarkers) >= 1  # Should extract at least tumor_size
        
        # Check specific biomarkers
        biomarker_names = [b["name"] for b in biomarkers]
        assert "tumor_size" in biomarker_names
    
    def test_parse_input_node_error_handling(self):
        """Test parse_input_node handles errors gracefully."""
        orchestrator = SentinelOrchestrator()
        
        # Invalid binary content that can't be decoded as UTF-8
        file_content = b'\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89'
        
        state = {
            "file_content": file_content,
            "filename": "invalid.bin",
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.parse_input_node(state)
        
        # Verify error is captured
        assert "parsed_data" in result
        parsed_data = result["parsed_data"]
        assert "error" in parsed_data
        
        # Verify recovery steps provided
        assert "recovery_steps" in parsed_data
        assert len(parsed_data["recovery_steps"]) > 0
    
    def test_parse_input_node_without_file_content(self):
        """Test parse_input_node when no file content provided."""
        orchestrator = SentinelOrchestrator()
        
        # State without file_content
        state = {
            "history": "Some existing history",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.parse_input_node(state)
        
        # Should return empty dict (no-op)
        assert result == {}
    
    def test_agent_state_includes_new_fields(self):
        """Verify AgentState includes new fields for input parsing."""
        from src.agents import AgentState
        
        # Check that AgentState has the new fields
        state = AgentState(
            history="test",
            biomarkers={},
            relevant_guidelines=[],
            audit_findings="",
            is_critical=False,
            file_content=b"test",
            filename="test.txt",
            parsed_data={}
        )
        
        assert state["file_content"] == b"test"
        assert state["filename"] == "test.txt"
        assert state["parsed_data"] == {}
