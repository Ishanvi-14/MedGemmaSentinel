"""
Safety Auditor component for adversarial hallucination checking.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict
try:
    from langchain_community.llms import Ollama
except Exception:
    # Provide a lightweight fallback so the module can be imported in environments
    # where `langchain_community` / `Ollama` is not installed. The SafetyAuditor
    # methods will handle a missing LLM gracefully by returning empty extractions.
    class Ollama:
        def __init__(self, model: str = None, temperature: float = 0):
            self.model = model
            self.temperature = temperature

        def invoke(self, prompt: str) -> str:
            # Returning an empty JSON-like string is safer than raising at import time.
            # Callers should handle empty responses.
            return "{}"

try:
    from src.input_parser.parser import Biomarker, BiomarkerType
except ImportError:
    from input_parser.parser import Biomarker, BiomarkerType


@dataclass
class Conflict:
    biomarker_name: str
    value_a: float
    value_b: float
    unit: str
    discrepancy_percentage: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComparisonResult:
    conflicts: List[Conflict]
    all_biomarkers: List[Biomarker]
    requires_human_review: bool
    overall_confidence: float


class SafetyAuditor:
    UNIT_CONVERSIONS = {
        'mm': 1.0, 'cm': 10.0, 'm': 1000.0,
        'ng/mL': 1.0, 'ng/ml': 1.0, 'ug/L': 1.0,
        'g/L': 1000000.0, 'mg/L': 1000.0,
    }
    TOLERANCE_PERCENTAGE = 10.0
    LOW_CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self, model_name: str = "gemma3:4b"):
        self.llm = Ollama(model=model_name, temperature=0)
    
    def extract_with_prompt_a(self, clinical_data: str) -> List[Biomarker]:
        prompt = f'''Extract the following biomarkers from the clinical data: tumor size, CEA level, EGFR status.
Provide values with units in JSON format.

Clinical data:
{clinical_data}

Return ONLY a JSON object with this structure:
{{
    "tumor_size": {{"value": <number>, "unit": "<mm or cm>"}},
    "CEA": {{"value": <number>, "unit": "<ng/mL or g/L>"}},
    "EGFR": {{"value": "<positive or negative>", "unit": "status"}}
}}

If a biomarker is not found, use null for the value. Return ONLY the JSON, no other text.'''
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_extraction_response(response, "prompt_a")
        except Exception as e:
            print(f"Extraction with prompt A failed: {e}")
            return []
    
    def extract_with_prompt_b(self, clinical_data: str) -> List[Biomarker]:
        prompt = f'''You are a clinical data specialist reviewing patient records.
Identify all measurable cancer biomarkers in this patient record, including their values and measurement units.

Patient record:
{clinical_data}

Focus on:
- Tumor measurements (size in mm or cm)
- CEA levels (carcinoembryonic antigen in ng/mL or g/L)
- EGFR mutation status (positive/negative)

Return ONLY a JSON object with this structure:
{{
    "tumor_size": {{"value": <number>, "unit": "<mm or cm>"}},
    "CEA": {{"value": <number>, "unit": "<ng/mL or g/L>"}},
    "EGFR": {{"value": "<positive or negative>", "unit": "status"}}
}}

If a biomarker is not found, use null for the value. Return ONLY the JSON, no other text.'''
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_extraction_response(response, "prompt_b")
        except Exception as e:
            print(f"Extraction with prompt B failed: {e}")
            return []
    
    def _parse_extraction_response(self, response: str, source: str) -> List[Biomarker]:
        biomarkers = []
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            data = json.loads(json_match.group())
            
            if data.get("tumor_size") and data["tumor_size"].get("value") is not None:
                biomarkers.append(Biomarker(
                    name="tumor_size", value=float(data["tumor_size"]["value"]),
                    unit=data["tumor_size"]["unit"], timestamp=datetime.now(),
                    source_field=source, confidence=1.0, biomarker_type=BiomarkerType.TUMOR_SIZE
                ))
            
            if data.get("CEA") and data["CEA"].get("value") is not None:
                biomarkers.append(Biomarker(
                    name="CEA", value=float(data["CEA"]["value"]),
                    unit=data["CEA"]["unit"], timestamp=datetime.now(),
                    source_field=source, confidence=1.0, biomarker_type=BiomarkerType.CEA
                ))
            
            if data.get("EGFR") and data["EGFR"].get("value") is not None:
                egfr_value = data["EGFR"]["value"].lower()
                numeric_value = 1.0 if "positive" in egfr_value or "mutation" in egfr_value or "detected" in egfr_value else 0.0
                biomarkers.append(Biomarker(
                    name="EGFR", value=numeric_value, unit="status",
                    timestamp=datetime.now(), source_field=source,
                    confidence=1.0, biomarker_type=BiomarkerType.EGFR
                ))
        except Exception as e:
            print(f"Error parsing extraction response from {source}: {e}")
        return biomarkers
    
    def normalize_units(self, value: float, unit: str) -> Tuple[float, str]:
        unit = unit.strip()
        if unit in ['mm', 'cm', 'm']:
            return (value * self.UNIT_CONVERSIONS.get(unit, 1.0), 'mm')
        if unit in ['ng/mL', 'ng/ml', 'ug/L', 'g/L', 'mg/L']:
            return (value * self.UNIT_CONVERSIONS.get(unit, 1.0), 'ng/mL')
        if unit == 'status':
            return (value, 'status')
        return (value, unit)
    
    def calculate_confidence(self, extraction_a: Biomarker, extraction_b: Biomarker) -> float:
        norm_a_value, norm_a_unit = self.normalize_units(extraction_a.value, extraction_a.unit)
        norm_b_value, norm_b_unit = self.normalize_units(extraction_b.value, extraction_b.unit)
        
        if norm_a_unit != norm_b_unit:
            return 0.3
        if norm_a_unit == 'status':
            return 1.0 if norm_a_value == norm_b_value else 0.0
        if norm_a_value == 0 and norm_b_value == 0:
            return 1.0
        
        avg_value = (norm_a_value + norm_b_value) / 2
        if avg_value == 0:
            return 0.0
        
        discrepancy = abs(norm_a_value - norm_b_value) / avg_value * 100
        
        if discrepancy <= self.TOLERANCE_PERCENTAGE:
            confidence = 1.0 - (discrepancy / self.TOLERANCE_PERCENTAGE) * 0.5
        else:
            confidence = max(0.0, 0.5 - (discrepancy - self.TOLERANCE_PERCENTAGE) / 20.0)
        
        return confidence
    
    def compare_extractions(self, extraction_a: List[Biomarker], extraction_b: List[Biomarker]) -> ComparisonResult:
        conflicts = []
        merged_biomarkers = []
        confidence_scores = []
        
        dict_a = {b.name: b for b in extraction_a}
        dict_b = {b.name: b for b in extraction_b}
        all_names = set(dict_a.keys()) | set(dict_b.keys())
        
        for name in all_names:
            biomarker_a = dict_a.get(name)
            biomarker_b = dict_b.get(name)
            
            if biomarker_a and biomarker_b:
                confidence = self.calculate_confidence(biomarker_a, biomarker_b)
                confidence_scores.append(confidence)
                
                norm_a_value, norm_a_unit = self.normalize_units(biomarker_a.value, biomarker_a.unit)
                norm_b_value, norm_b_unit = self.normalize_units(biomarker_b.value, biomarker_b.unit)
                
                if norm_a_unit == norm_b_unit and norm_a_unit != 'status':
                    avg_value = (norm_a_value + norm_b_value) / 2
                    if avg_value > 0:
                        discrepancy_pct = abs(norm_a_value - norm_b_value) / avg_value * 100
                        if discrepancy_pct > self.TOLERANCE_PERCENTAGE:
                            conflicts.append(Conflict(
                                biomarker_name=name, value_a=norm_a_value, value_b=norm_b_value,
                                unit=norm_a_unit, discrepancy_percentage=discrepancy_pct
                            ))
                elif norm_a_unit == 'status':
                    if norm_a_value != norm_b_value:
                        conflicts.append(Conflict(
                            biomarker_name=name, value_a=norm_a_value, value_b=norm_b_value,
                            unit='status', discrepancy_percentage=100.0
                        ))
                
                avg_value = (norm_a_value + norm_b_value) / 2
                merged_biomarkers.append(Biomarker(
                    name=name, value=avg_value, unit=norm_a_unit, timestamp=datetime.now(),
                    source_field="merged", confidence=confidence, biomarker_type=biomarker_a.biomarker_type
                ))
            
            elif biomarker_a:
                confidence = 0.5
                confidence_scores.append(confidence)
                merged_biomarkers.append(Biomarker(
                    name=name, value=biomarker_a.value, unit=biomarker_a.unit, timestamp=datetime.now(),
                    source_field="extraction_a_only", confidence=confidence, biomarker_type=biomarker_a.biomarker_type
                ))
            
            elif biomarker_b:
                confidence = 0.5
                confidence_scores.append(confidence)
                merged_biomarkers.append(Biomarker(
                    name=name, value=biomarker_b.value, unit=biomarker_b.unit, timestamp=datetime.now(),
                    source_field="extraction_b_only", confidence=confidence, biomarker_type=biomarker_b.biomarker_type
                ))
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        requires_review = len(conflicts) > 0 or overall_confidence < self.LOW_CONFIDENCE_THRESHOLD
        
        return ComparisonResult(
            conflicts=conflicts, all_biomarkers=merged_biomarkers,
            requires_human_review=requires_review, overall_confidence=overall_confidence
        )
