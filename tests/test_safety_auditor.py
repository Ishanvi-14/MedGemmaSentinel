"""
Property-based and unit tests for Safety Auditor component.

Tests dual extraction, unit normalization, conflict detection, and confidence scoring.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.safety_auditor.auditor import SafetyAuditor, Conflict, ComparisonResult
from src.input_parser.parser import Biomarker, BiomarkerType


# ============================================================================
# Property-Based Tests
# ============================================================================

# Feature: sentinel-competition-enhancements, Property 9: Unit Normalization for Comparison
@given(
    value=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    unit_pair=st.sampled_from([("mm", "cm"), ("ng/mL", "g/L"), ("ng/mL", "mg/L")])
)
@settings(max_examples=100)
def test_property_unit_normalization_equivalence(value, unit_pair):
    """
    Property 9: For any pair of biomarker values with equivalent measurements
    in different units, the Safety_Auditor should recognize them as matching
    after normalization.
    
    **Validates: Requirements 3.2**
    """
    auditor = SafetyAuditor()
    unit_a, unit_b = unit_pair
    
    # Convert value from unit_a to unit_b
    if unit_a == "mm" and unit_b == "cm":
        value_b = value / 10.0
    elif unit_a == "ng/mL" and unit_b == "g/L":
        value_b = value / 1000000.0
    elif unit_a == "ng/mL" and unit_b == "mg/L":
        value_b = value / 1000.0
    else:
        value_b = value
    
    # Normalize both
    norm_a = auditor.normalize_units(value, unit_a)
    norm_b = auditor.normalize_units(value_b, unit_b)
    
    # Assert they match after normalization (within floating point tolerance)
    assert norm_a[0] == pytest.approx(norm_b[0], rel=0.01), \
        f"Normalized values don't match: {norm_a[0]} vs {norm_b[0]}"
    assert norm_a[1] == norm_b[1], \
        f"Normalized units don't match: {norm_a[1]} vs {norm_b[1]}"


# Feature: sentinel-competition-enhancements, Property 10: Conflict Detection and Flagging
@given(
    base_value=st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    discrepancy_pct=st.floats(min_value=11.0, max_value=50.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_conflict_detection_beyond_tolerance(base_value, discrepancy_pct):
    """
    Property 10: For any pair of duplicate extractions where values differ
    beyond acceptable tolerance (10%), the Safety_Auditor should flag a
    Clinical_Conflict.
    
    **Validates: Requirements 3.3**
    """
    auditor = SafetyAuditor()
    
    # Create two biomarkers with values differing by discrepancy_pct
    value_a = base_value
    value_b = base_value * (1 + discrepancy_pct / 100.0)
    
    biomarker_a = Biomarker(
        name="tumor_size",
        value=value_a,
        unit="mm",
        timestamp=datetime.now(),
        source_field="prompt_a",
        confidence=1.0,
        biomarker_type=BiomarkerType.TUMOR_SIZE
    )
    
    biomarker_b = Biomarker(
        name="tumor_size",
        value=value_b,
        unit="mm",
        timestamp=datetime.now(),
        source_field="prompt_b",
        confidence=1.0,
        biomarker_type=BiomarkerType.TUMOR_SIZE
    )
    
    # Compare extractions
    result = auditor.compare_extractions([biomarker_a], [biomarker_b])
    
    # Should have at least one conflict
    assert len(result.conflicts) > 0, \
        f"Expected conflict for {discrepancy_pct}% discrepancy, but got none"
    
    # The conflict should be for tumor_size
    assert result.conflicts[0].biomarker_name == "tumor_size"
    
    # Discrepancy percentage should be greater than tolerance
    assert result.conflicts[0].discrepancy_percentage > auditor.TOLERANCE_PERCENTAGE


# Feature: sentinel-competition-enhancements, Property 12: Confidence Score Assignment
@given(
    value=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_confidence_score_range(value):
    """
    Property 12: For any extracted biomarker, the Safety_Auditor should assign
    a confidence score between 0.0 and 1.0.
    
    **Validates: Requirements 3.5**
    """
    auditor = SafetyAuditor()
    
    # Create two identical biomarkers (should have high confidence)
    biomarker_a = Biomarker(
        name="CEA",
        value=value,
        unit="ng/mL",
        timestamp=datetime.now(),
        source_field="prompt_a",
        confidence=1.0,
        biomarker_type=BiomarkerType.CEA
    )
    
    biomarker_b = Biomarker(
        name="CEA",
        value=value,
        unit="ng/mL",
        timestamp=datetime.now(),
        source_field="prompt_b",
        confidence=1.0,
        biomarker_type=BiomarkerType.CEA
    )
    
    # Calculate confidence
    confidence = auditor.calculate_confidence(biomarker_a, biomarker_b)
    
    # Confidence must be in valid range [0.0, 1.0]
    assert 0.0 <= confidence <= 1.0, \
        f"Confidence score {confidence} is outside valid range [0.0, 1.0]"



# ============================================================================
# Unit Tests
# ============================================================================

class TestSafetyAuditorUnitTests:
    """Unit tests for SafetyAuditor component."""
    
    def test_unit_conversion_mm_to_cm(self):
        """Test conversion from mm to cm."""
        auditor = SafetyAuditor()
        
        # 20mm should equal 2cm
        norm_mm = auditor.normalize_units(20.0, "mm")
        norm_cm = auditor.normalize_units(2.0, "cm")
        
        assert norm_mm[0] == pytest.approx(norm_cm[0], rel=0.01)
        assert norm_mm[1] == norm_cm[1] == "mm"
    
    def test_unit_conversion_ng_ml_to_g_l(self):
        """Test conversion from ng/mL to g/L."""
        auditor = SafetyAuditor()
        
        # 2,000,000 ng/mL should equal 2 g/L
        norm_ng = auditor.normalize_units(2000000.0, "ng/mL")
        norm_g = auditor.normalize_units(2.0, "g/L")
        
        assert norm_ng[0] == pytest.approx(norm_g[0], rel=0.01)
        assert norm_ng[1] == norm_g[1] == "ng/mL"
    
    def test_unit_conversion_edge_case_zero(self):
        """Test unit conversion with zero value."""
        auditor = SafetyAuditor()
        
        norm = auditor.normalize_units(0.0, "mm")
        assert norm[0] == 0.0
        assert norm[1] == "mm"
    
    def test_unit_conversion_edge_case_large_value(self):
        """Test unit conversion with very large value."""
        auditor = SafetyAuditor()
        
        # 1000mm = 100cm
        norm = auditor.normalize_units(1000.0, "mm")
        assert norm[0] == 1000.0
        assert norm[1] == "mm"
    
    def test_conflict_detection_matching_extractions(self):
        """Test that matching extractions produce no conflicts."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="tumor_size",
            value=25.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        biomarker_b = Biomarker(
            name="tumor_size",
            value=25.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        result = auditor.compare_extractions([biomarker_a], [biomarker_b])
        
        assert len(result.conflicts) == 0
        assert result.requires_human_review == False
        assert result.overall_confidence > 0.9

    
    def test_conflict_detection_mismatched_extractions(self):
        """Test that mismatched extractions produce conflicts."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="tumor_size",
            value=20.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        biomarker_b = Biomarker(
            name="tumor_size",
            value=30.0,  # 50% larger - should trigger conflict
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        result = auditor.compare_extractions([biomarker_a], [biomarker_b])
        
        assert len(result.conflicts) > 0
        assert result.requires_human_review == True
        assert result.conflicts[0].biomarker_name == "tumor_size"
    
    def test_conflict_detection_categorical_mismatch(self):
        """Test that categorical mismatches (EGFR) produce conflicts."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="EGFR",
            value=1.0,  # positive
            unit="status",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.EGFR
        )
        
        biomarker_b = Biomarker(
            name="EGFR",
            value=0.0,  # negative
            unit="status",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.EGFR
        )
        
        result = auditor.compare_extractions([biomarker_a], [biomarker_b])
        
        assert len(result.conflicts) > 0
        assert result.requires_human_review == True
        assert result.conflicts[0].biomarker_name == "EGFR"
    
    def test_confidence_calculation_perfect_match(self):
        """Test confidence calculation for perfect match."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="CEA",
            value=5.2,
            unit="ng/mL",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.CEA
        )
        
        biomarker_b = Biomarker(
            name="CEA",
            value=5.2,
            unit="ng/mL",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.CEA
        )
        
        confidence = auditor.calculate_confidence(biomarker_a, biomarker_b)
        
        assert confidence == pytest.approx(1.0, rel=0.01)
    
    def test_confidence_calculation_within_tolerance(self):
        """Test confidence calculation for values within tolerance."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="CEA",
            value=5.0,
            unit="ng/mL",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.CEA
        )
        
        biomarker_b = Biomarker(
            name="CEA",
            value=5.5,  # 10% difference
            unit="ng/mL",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.CEA
        )
        
        confidence = auditor.calculate_confidence(biomarker_a, biomarker_b)
        
        # Should have moderate confidence (around 0.5-1.0)
        assert 0.4 <= confidence <= 1.0
    
    def test_low_confidence_flagging(self):
        """Test that low confidence triggers human review."""
        auditor = SafetyAuditor()
        
        # Create biomarkers with significant discrepancy
        biomarker_a = Biomarker(
            name="tumor_size",
            value=10.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        biomarker_b = Biomarker(
            name="tumor_size",
            value=20.0,  # 100% larger
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_b",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        result = auditor.compare_extractions([biomarker_a], [biomarker_b])
        
        # Should require human review due to low confidence
        assert result.requires_human_review == True
        assert result.overall_confidence < auditor.LOW_CONFIDENCE_THRESHOLD
    
    def test_biomarker_only_in_one_extraction(self):
        """Test handling of biomarker found in only one extraction."""
        auditor = SafetyAuditor()
        
        biomarker_a = Biomarker(
            name="tumor_size",
            value=25.0,
            unit="mm",
            timestamp=datetime.now(),
            source_field="prompt_a",
            confidence=1.0,
            biomarker_type=BiomarkerType.TUMOR_SIZE
        )
        
        # Only in extraction A, not in B
        result = auditor.compare_extractions([biomarker_a], [])
        
        # Should have merged biomarker with reduced confidence
        assert len(result.all_biomarkers) == 1
        assert result.all_biomarkers[0].confidence == 0.5
        assert result.all_biomarkers[0].source_field == "extraction_a_only"
