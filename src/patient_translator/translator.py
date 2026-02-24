"""
Patient Translator component for converting technical findings to patient-friendly language.

Simplifies medical jargon to 5th-grade reading level and translates to multiple languages
while preserving medical accuracy and terminology.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
try:
    import textstat
except Exception:
    # Fallback minimal implementation for environments without textstat.
    class _TextStatFallback:
        @staticmethod
        def flesch_kincaid_grade(text: str) -> float:
            # Very simple heuristic fallback based on average sentence length.
            sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
            words = max(1, len(text.split()))
            avg_words_per_sentence = words / sentences
            # Map average words per sentence to a rough grade level
            return float(min(12, max(3, avg_words_per_sentence / 1.5)))

    textstat = _TextStatFallback()
try:
    from langchain_community.llms import Ollama
except Exception:
    class Ollama:
        def __init__(self, model: str = None, temperature: float = 0):
            self.model = model
            self.temperature = temperature

        def invoke(self, prompt: str) -> str:
            return ""


class SeverityLevel(Enum):
    """Severity levels for clinical findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Finding:
    """Represents a clinical audit finding."""
    finding_id: str
    severity: SeverityLevel
    description: str
    biomarker_name: Optional[str] = None


@dataclass
class SimplifiedFinding:
    """Represents a simplified patient-friendly finding."""
    original_text: str
    simplified_text: str
    reading_level: float
    severity_icon: str
    key_terms_preserved: List[str] = field(default_factory=list)


@dataclass
class TranslatedFinding:
    """Represents a translated finding in a target language."""
    simplified_finding: SimplifiedFinding
    language: str
    translated_text: str


class PatientTranslator:
    """
    Converts technical audit findings to patient-friendly language.
    
    Simplifies medical jargon to specified reading level, translates to multiple
    languages, and adds visual severity indicators while preserving medical accuracy.
    """
    
    # Severity to emoji mapping
    SEVERITY_ICONS = {
        SeverityLevel.LOW: "🟢",
        SeverityLevel.MEDIUM: "🟡",
        SeverityLevel.HIGH: "🔴"
    }
    
    # Supported languages
    SUPPORTED_LANGUAGES = ["en", "es", "hi", "mr"]
    
    # Language names for prompts
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "hi": "Hindi",
        "mr": "Marathi"
    }
    
    # Simplification prompt template
    SIMPLIFICATION_PROMPT = """You are a medical translator helping patients understand their test results.

Original finding: {technical_finding}

Rewrite this for a patient with a 5th-grade reading level. Rules:
1. Use simple words (avoid: "metastatic", use: "cancer spread")
2. Keep sentences short (max 15 words)
3. Preserve critical medical facts
4. Use analogies when helpful
5. Maintain accuracy - do not minimize serious findings

Simplified version:"""
    
    # Translation prompt template
    TRANSLATION_PROMPT = """Translate the following medical finding to {language}.

Original text in English: {text}

Rules:
1. Translate to {language}
2. Keep medical terms in English with translation in parentheses
3. Example: "El tumor ha crecido (tumor growth) de 2cm a 3cm"
4. Maintain accuracy and clarity
5. Use simple language appropriate for patients

Translated text:"""
    
    def __init__(self, model_name: str = "gemma3:4b", target_reading_level: int = 5):
        """
        Initialize the Patient Translator.
        
        Args:
            model_name: Ollama model name for simplification and translation
            target_reading_level: Target Flesch-Kincaid grade level (default: 5)
        """
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.target_reading_level = target_reading_level
    
    def simplify_findings(
        self, 
        audit_findings: List[Finding], 
        reading_level: int = 5
    ) -> List[SimplifiedFinding]:
        """
        Convert technical findings to specified reading level.
        
        Args:
            audit_findings: List of technical audit findings
            reading_level: Target Flesch-Kincaid grade level
            
        Returns:
            List of SimplifiedFinding objects with patient-friendly text
        """
        simplified_findings = []
        
        for finding in audit_findings:
            # Generate simplified text using MedGemma
            prompt = self.SIMPLIFICATION_PROMPT.format(
                technical_finding=finding.description
            )
            
            try:
                simplified_text = self.llm.invoke(prompt).strip()
                
                # Calculate readability score
                fk_grade = textstat.flesch_kincaid_grade(simplified_text)
                
                # Extract key medical terms from original
                key_terms = self._extract_medical_terms(finding.description)
                
                # Add visual severity indicator
                severity_icon = self.add_visual_aids(finding.severity)
                
                simplified_findings.append(SimplifiedFinding(
                    original_text=finding.description,
                    simplified_text=simplified_text,
                    reading_level=fk_grade,
                    severity_icon=severity_icon,
                    key_terms_preserved=key_terms
                ))
                
            except Exception as e:
                print(f"Error simplifying finding {finding.finding_id}: {e}")
                # Fallback: use original text with icon
                simplified_findings.append(SimplifiedFinding(
                    original_text=finding.description,
                    simplified_text=finding.description,
                    reading_level=textstat.flesch_kincaid_grade(finding.description),
                    severity_icon=self.add_visual_aids(finding.severity),
                    key_terms_preserved=[]
                ))
        
        return simplified_findings
    
    def translate_to_language(
        self, 
        simplified_findings: List[SimplifiedFinding], 
        target_language: str
    ) -> List[TranslatedFinding]:
        """
        Translate simplified findings to target language.
        
        Args:
            simplified_findings: List of simplified findings
            target_language: Target language code ("en", "es", "hi", "mr")
            
        Returns:
            List of TranslatedFinding objects
            
        Raises:
            ValueError: If target_language is not supported
        """
        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {target_language}. "
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        
        # If target is English, no translation needed
        if target_language == "en":
            return [
                TranslatedFinding(
                    simplified_finding=sf,
                    language="en",
                    translated_text=sf.simplified_text
                )
                for sf in simplified_findings
            ]
        
        translated_findings = []
        language_name = self.LANGUAGE_NAMES[target_language]
        
        for simplified_finding in simplified_findings:
            prompt = self.TRANSLATION_PROMPT.format(
                language=language_name,
                text=simplified_finding.simplified_text
            )
            
            try:
                translated_text = self.llm.invoke(prompt).strip()
                
                translated_findings.append(TranslatedFinding(
                    simplified_finding=simplified_finding,
                    language=target_language,
                    translated_text=translated_text
                ))
                
            except Exception as e:
                print(f"Error translating to {language_name}: {e}")
                # Fallback: use simplified English text
                translated_findings.append(TranslatedFinding(
                    simplified_finding=simplified_finding,
                    language=target_language,
                    translated_text=simplified_finding.simplified_text
                ))
        
        return translated_findings
    
    def add_visual_aids(self, severity: SeverityLevel) -> str:
        """
        Add emoji/icon indicators for severity levels.
        
        Args:
            severity: SeverityLevel enum value
            
        Returns:
            Emoji string for the severity level
        """
        return self.SEVERITY_ICONS.get(severity, "⚪")
    
    def validate_medical_accuracy(
        self, 
        original: Finding, 
        simplified: SimplifiedFinding
    ) -> bool:
        """
        Verify that simplification preserves medical meaning.
        
        Checks that critical medical terms from the original finding
        are preserved in the simplified version.
        
        Args:
            original: Original technical finding
            simplified: Simplified patient-friendly finding
            
        Returns:
            True if medical accuracy is preserved, False otherwise
        """
        # Extract medical terms from original
        original_terms = self._extract_medical_terms(original.description)
        
        if not original_terms:
            # No critical terms to preserve
            return True
        
        # Check if key terms appear in simplified text (case-insensitive)
        simplified_lower = simplified.simplified_text.lower()
        
        # Count how many terms are preserved
        preserved_count = sum(
            1 for term in original_terms 
            if term.lower() in simplified_lower
        )
        
        # Require at least 70% of critical terms to be preserved
        preservation_ratio = preserved_count / len(original_terms)
        
        return preservation_ratio >= 0.7
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract critical medical terms from text.
        
        Uses pattern matching to identify medical terminology that should
        be preserved during simplification.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            List of medical terms found
        """
        # Common medical term patterns
        medical_patterns = [
            r'\b(?:tumor|cancer|metastatic|metastasis|carcinoma|lesion)\b',
            r'\b(?:CEA|EGFR|biomarker|antigen)\b',
            r'\b(?:chemotherapy|radiation|immunotherapy|surgery)\b',
            r'\b(?:RECIST|NCCN|guideline)\b',
            r'\b\d+\s*(?:mm|cm|ng/mL|g/L)\b',  # Measurements
        ]
        
        terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates
