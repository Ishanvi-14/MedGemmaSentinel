"""
Integration tests for Safety Auditor workflow integration.
"""

import pytest
from src.agents import SentinelOrchestrator


class TestSafetyAuditorIntegration:
    """Test that Safety Auditor integrates correctly with LangGraph workflow."""
    
    def test_workflow_has_safety_auditor_node(self):
        """Verify safety_auditor_node is added to workflow."""
        orchestrator = SentinelOrchestrator()
        graph = orchestrator.build_graph()
        
        # The graph should be built successfully
        assert graph is not None
    
    def test_safety_auditor_node_with_clinical_text(self):
        """Test safety_auditor_node with clinical text."""
        orchestrator = SentinelOrchestrator()
        
        # Create state with parsed clinical data
        state = {
            "parsed_data": {
                "patient_id": "P-12345",
                "raw_text": """
                Patient presents with tumor size measuring 30mm.
                CEA level is 8.5 ng/mL.
                EGFR mutation analysis shows positive result.
                """,
                "format_type": "text",
                "biomarkers": []
            },
            "history": "Patient clinical notes",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.safety_auditor_node(state)
        
        # Verify safety_result is returned
        assert "safety_result" in result
        safety_result = result["safety_result"]
        
        # Verify no errors
        assert "error" not in safety_result or safety_result.get("error") is None
        
        # Verify structure
        assert "conflicts" in safety_result
        assert "all_biomarkers" in safety_result
        assert "requires_human_review" in safety_result
        assert "overall_confidence" in safety_result
        assert "extraction_a" in safety_result
        assert "extraction_b" in safety_result
    
    def test_safety_auditor_node_without_parsed_data(self):
        """Test safety_auditor_node when no parsed data available."""
        orchestrator = SentinelOrchestrator()
        
        # State without parsed_data
        state = {
            "history": "Some history",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.safety_auditor_node(state)
        
        # Should return empty dict (no-op)
        assert result == {}
    
    def test_safety_auditor_node_with_parsing_error(self):
        """Test safety_auditor_node when parsing had errors."""
        orchestrator = SentinelOrchestrator()
        
        # State with parsing error
        state = {
            "parsed_data": {
                "error": "Failed to parse file",
                "patient_id": "unknown",
                "biomarkers": []
            },
            "history": "",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.safety_auditor_node(state)
        
        # Should skip safety check
        assert result == {}
    
    def test_agent_state_includes_safety_fields(self):
        """Verify AgentState includes new fields for safety auditor."""
        from src.agents import AgentState
        
        # Check that AgentState has the safety fields
        state = AgentState(
            history="test",
            biomarkers={},
            relevant_guidelines=[],
            audit_findings="",
            is_critical=False,
            file_content=b"test",
            filename="test.txt",
            parsed_data={},
            safety_result={},
            human_approved=False
        )
        
        assert state["safety_result"] == {}
        assert state["human_approved"] == False
    
    def test_human_review_node_exists(self):
        """Verify human_review_node is added to workflow."""
        orchestrator = SentinelOrchestrator()
        
        # Test human_review_node directly
        state = {
            "history": "test",
            "biomarkers": {},
            "relevant_guidelines": [],
            "audit_findings": "",
            "is_critical": False
        }
        
        result = orchestrator.human_review_node(state)
        
        # Should return empty dict (placeholder)
        assert result == {}
