"""
Multi-format input parser for clinical data.

Supports parsing of PDF, FHIR JSON, Synthea JSON, CSV, and plain text formats.
Extracts biomarkers and normalizes them to a common representation.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum

# Make heavy dependencies optional at import time so the module can be imported
# even in minimal test environments. Functions that require these libraries
# will raise informative errors or use lightweight fallbacks when needed.
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None


class BiomarkerType(Enum):
    """Enumeration of supported biomarker types."""
    TUMOR_SIZE = "tumor_size"
    CEA = "CEA"
    EGFR = "EGFR"
    CA_19_9 = "CA_19_9"
    CUSTOM = "custom"


@dataclass
class Biomarker:
    """Normalized biomarker representation."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source_field: str
    confidence: float
    biomarker_type: Optional[BiomarkerType] = None
    
    def __post_init__(self):
        """Automatically determine biomarker type if not set."""
        if self.biomarker_type is None:
            name_upper = self.name.upper()
            if "TUMOR" in name_upper or "SIZE" in name_upper:
                self.biomarker_type = BiomarkerType.TUMOR_SIZE
            elif "CEA" in name_upper:
                self.biomarker_type = BiomarkerType.CEA
            elif "EGFR" in name_upper:
                self.biomarker_type = BiomarkerType.EGFR
            elif "CA" in name_upper and "19" in name_upper:
                self.biomarker_type = BiomarkerType.CA_19_9
            else:
                self.biomarker_type = BiomarkerType.CUSTOM


@dataclass
class ParsedClinicalData:
    """Result of parsing clinical data."""
    patient_id: str
    biomarkers: List[Biomarker]
    raw_text: str
    format_type: str
    parse_timestamp: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


class InputParser:
    """
    Multi-format clinical data parser.
    
    Detects and parses PDF, FHIR JSON, Synthea JSON, CSV, and plain text formats.
    Extracts biomarkers: tumor size (mm), CEA levels (ng/mL), EGFR mutation status.
    """
    
    SUPPORTED_FORMATS = ["pdf", "fhir_json", "synthea_json", "csv", "text"]
    
    def __init__(self):
        """Initialize the parser."""
        self.biomarker_patterns = {
            "tumor_size": [
                r"tumor\s+(?:size\s+)?(?:now\s+)?measuring\s+(\d+(?:\.\d+)?)\s*(mm|cm)",
                r"tumor\s+size[:\s]+(\d+(?:\.\d+)?)\s*(mm|cm)",
                r"size\s+tumor[:\s]+(\d+(?:\.\d+)?)\s*(mm|cm)",
                r"lesion[:\s]+(\d+(?:\.\d+)?)\s*(mm|cm)",
                r"measuring\s+(\d+(?:\.\d+)?)\s*(mm|cm)",
            ],
            "CEA": [
                r"CEA\s+(?:elevated\s+)?at\s+(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|ug/L)",
                r"CEA[:\s]+(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|ug/L)",
                r"carcinoembryonic\s+antigen[:\s]+(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|ug/L)",
            ],
            "EGFR": [
                r"EGFR[:\s]+(?:mutation\s+analysis\s+)?(positive|negative|mutation|wild[- ]?type|detected|not\s+detected)",
                r"EGFR\s+mutation\s+analysis\s+(positive|negative|detected)",
                r"epidermal\s+growth\s+factor\s+receptor[:\s]+(positive|negative|mutation|wild[- ]?type)",
            ],
        }
    
    def detect_format(self, file_content: bytes, filename: str) -> str:
        """
        Detect input format from content and filename.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Format string: "synthea_json" | "fhir_json" | "pdf" | "csv" | "text"
            
        Raises:
            ValueError: If format cannot be detected
        """
        # Check PDF magic bytes
        if file_content.startswith(b'%PDF'):
            return "pdf"
        
        # Try to parse as JSON
        try:
            data = json.loads(file_content.decode('utf-8'))
            
            # Check for FHIR structure
            if isinstance(data, dict):
                resource_type = data.get('resourceType')
                
                # FHIR Bundle with Synthea-specific structure
                if resource_type == 'Bundle' and 'entry' in data:
                    entries = data.get('entry', [])
                    if entries:
                        # Check if it's Synthea format (has specific structure)
                        first_resource = entries[0].get('resource', {})
                        if 'identifier' in first_resource and any(
                            'synthea' in str(ident.get('system', '')).lower()
                            for ident in first_resource.get('identifier', [])
                        ):
                            return "synthea_json"
                    return "fhir_json"
                
                # Standalone FHIR resource
                if resource_type in ['Patient', 'Observation', 'DiagnosticReport', 'Condition']:
                    return "fhir_json"
        
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # Check for CSV format
        try:
            text = file_content.decode('utf-8')
            lines = text.strip().split('\n')
            if len(lines) > 1 and ',' in lines[0]:
                # Check if first line looks like a header
                first_line = lines[0]
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_,\s]*$', first_line):
                    return "csv"
        except UnicodeDecodeError:
            pass
        
        # Default to plain text
        try:
            file_content.decode('utf-8')
            return "text"
        except UnicodeDecodeError:
            raise ValueError(
                f"Unable to detect format for file: {filename}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def parse(self, file_content: bytes, format_type: Optional[str] = None, 
              filename: str = "unknown") -> ParsedClinicalData:
        """
        Parse file content based on detected or specified format.
        
        Args:
            file_content: Raw file bytes
            format_type: Optional format override
            filename: Original filename for error messages
            
        Returns:
            ParsedClinicalData with normalized biomarkers
            
        Raises:
            ValueError: If format is unsupported or parsing fails
        """
        # Detect format if not specified
        if format_type is None:
            format_type = self.detect_format(file_content, filename)
        
        # Route to appropriate parser
        if format_type == "pdf":
            return self.parse_pdf(file_content)
        elif format_type == "fhir_json":
            return self.parse_fhir_json(file_content)
        elif format_type == "synthea_json":
            return self.parse_synthea_json(file_content)
        elif format_type == "csv":
            return self.parse_csv(file_content)
        elif format_type == "text":
            return self.parse_text(file_content)
        else:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def parse_pdf(self, file_content: bytes) -> ParsedClinicalData:
        """
        Parse PDF clinical report using pdfplumber.
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            ParsedClinicalData with extracted biomarkers
        """
        import io

        if pdfplumber is None:
            raise ValueError("Failed to parse PDF: pdfplumber is not installed in this environment")

        # Extract text from PDF
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
        
        if not text.strip():
            raise ValueError("PDF contains no extractable text")
        
        # Extract biomarkers using regex patterns
        biomarkers = self._extract_biomarkers_from_text(text, "pdf")
        
        # Try to extract patient ID
        patient_id = self._extract_patient_id(text)
        
        return ParsedClinicalData(
            patient_id=patient_id,
            biomarkers=biomarkers,
            raw_text=text,
            format_type="pdf",
            parse_timestamp=datetime.now(),
            metadata={"page_count": len(text.split('\n'))}
        )
    
    def parse_fhir_json(self, file_content: bytes) -> ParsedClinicalData:
        """
        Parse HL7 FHIR JSON format.
        
        Args:
            file_content: FHIR JSON bytes
            
        Returns:
            ParsedClinicalData with extracted biomarkers
        """
        data = json.loads(file_content.decode('utf-8'))
        
        biomarkers = []
        patient_id = "unknown"
        raw_text = ""
        
        # Handle Bundle resource
        if data.get('resourceType') == 'Bundle':
            entries = data.get('entry', [])
            
            for entry in entries:
                resource = entry.get('resource', {})
                resource_type = resource.get('resourceType')
                
                # Extract patient ID
                if resource_type == 'Patient':
                    patient_id = resource.get('id', 'unknown')
                
                # Extract observations
                elif resource_type == 'Observation':
                    biomarker = self._parse_fhir_observation(resource)
                    if biomarker:
                        biomarkers.append(biomarker)
                        raw_text += f"{biomarker.name}: {biomarker.value} {biomarker.unit}\n"
        
        # Handle standalone Observation
        elif data.get('resourceType') == 'Observation':
            biomarker = self._parse_fhir_observation(data)
            if biomarker:
                biomarkers.append(biomarker)
                raw_text = f"{biomarker.name}: {biomarker.value} {biomarker.unit}\n"
        
        return ParsedClinicalData(
            patient_id=patient_id,
            biomarkers=biomarkers,
            raw_text=raw_text,
            format_type="fhir_json",
            parse_timestamp=datetime.now(),
            metadata={"resource_type": data.get('resourceType')}
        )
    
    def parse_synthea_json(self, file_content: bytes) -> ParsedClinicalData:
        """
        Parse Synthea JSON format (backward compatible).
        
        Args:
            file_content: Synthea JSON bytes
            
        Returns:
            ParsedClinicalData with extracted biomarkers
        """
        # Synthea is a specific type of FHIR Bundle, so we can reuse FHIR parser
        # but mark it as synthea_json for backward compatibility tracking
        result = self.parse_fhir_json(file_content)
        result.format_type = "synthea_json"
        result.metadata["backward_compatible"] = True
        return result
    
    def parse_csv(self, file_content: bytes) -> ParsedClinicalData:
        """
        Parse CSV lab results using pandas.
        
        Args:
            file_content: CSV file bytes
            
        Returns:
            ParsedClinicalData with extracted biomarkers
        """
        import io
        import csv

        biomarkers = []
        patient_id = "unknown"
        raw_text = ""

        # If pandas is available, use it; otherwise fall back to csv.DictReader
        if pd is not None:
            df = pd.read_csv(io.BytesIO(file_content))

            # Try to find patient ID column
            patient_id_cols = ['patient_id', 'PatientID', 'patient', 'id', 'Patient']
            for col in patient_id_cols:
                if col in df.columns and not df[col].empty:
                    patient_id = str(df[col].iloc[0])
                    break
        else:
            # Fallback CSV parsing
            text = file_content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            if rows:
                cols = reader.fieldnames or []
                # Find patient id
                for pid_col in ['patient_id', 'PatientID', 'patient', 'id', 'Patient']:
                    if pid_col in cols and rows[0].get(pid_col):
                        patient_id = rows[0].get(pid_col)
                        break
            else:
                cols = []
        
        # Map common column names to biomarkers
        column_mapping = {
            'tumor_size': ['tumor_size', 'tumor size', 'size', 'lesion_size'],
            'CEA': ['CEA', 'cea', 'carcinoembryonic_antigen'],
            'EGFR': ['EGFR', 'egfr', 'egfr_status', 'mutation_status'],
        }
        
        # Check if CSV has a biomarker_name column (long format)
        if pd is not None and 'biomarker_name' in df.columns:
            # Long format: each row is a biomarker measurement
            for idx, row in df.iterrows():
                biomarker_name_raw = str(row.get('biomarker_name', '')).strip()

                # Map biomarker name
                biomarker_name = None
                if biomarker_name_raw in ['tumor_size', 'Tumor Size', 'tumor size']:
                    biomarker_name = 'tumor_size'
                elif biomarker_name_raw in ['CEA', 'cea']:
                    biomarker_name = 'CEA'
                elif biomarker_name_raw in ['EGFR', 'egfr']:
                    biomarker_name = 'EGFR'

                if not biomarker_name:
                    continue

                value = row.get('value')
                if pd.notna(value):
                    unit = str(row.get('unit', ''))

                    # Parse value
                    parsed_value = self._parse_value(str(value))

                    # Get timestamp
                    timestamp = datetime.now()
                    date_cols = ['date', 'timestamp', 'measurement_date', 'test_date']
                    for date_col in date_cols:
                        if date_col in df.columns and pd.notna(row.get(date_col)):
                            try:
                                timestamp = pd.to_datetime(row[date_col])
                                break
                            except:
                                pass

                    biomarker = Biomarker(
                        name=biomarker_name,
                        value=parsed_value,
                        unit=unit,
                        timestamp=timestamp,
                        source_field='biomarker_name',
                        confidence=0.9  # High confidence for structured CSV
                    )
                    biomarkers.append(biomarker)
                    raw_text += f"{biomarker_name}: {parsed_value} {unit}\n"
        elif pd is None:
            # Fallback long-format or generic parsing using rows list
            for row in rows:
                biomarker_name_raw = str(row.get('biomarker_name', '')).strip()
                if not biomarker_name_raw:
                    # Try wide-format detection per column mapping below
                    continue

                biomarker_name = None
                if biomarker_name_raw.lower() in ['tumor_size', 'tumor size']:
                    biomarker_name = 'tumor_size'
                elif biomarker_name_raw.lower() == 'cea':
                    biomarker_name = 'CEA'
                elif biomarker_name_raw.lower() == 'egfr':
                    biomarker_name = 'EGFR'

                if not biomarker_name:
                    continue

                value = row.get('value')
                if value is not None and value != '':
                    unit = row.get('unit', '')
                    parsed_value = self._parse_value(str(value))
                    timestamp = datetime.now()
                    for date_col in ['date', 'timestamp', 'measurement_date', 'test_date', 'test_date']:
                        if date_col in row and row.get(date_col):
                            try:
                                timestamp = datetime.fromisoformat(row.get(date_col))
                                break
                            except:
                                pass

                    biomarker = Biomarker(
                        name=biomarker_name,
                        value=parsed_value,
                        unit=unit,
                        timestamp=timestamp,
                        source_field='biomarker_name',
                        confidence=0.9
                    )
                    biomarkers.append(biomarker)
                    raw_text += f"{biomarker_name}: {parsed_value} {unit}\n"
            # If fallback parsed some biomarkers skip wide-format
            if biomarkers:
                return ParsedClinicalData(
                    patient_id=patient_id,
                    biomarkers=biomarkers,
                    raw_text=raw_text,
                    format_type="csv",
                    parse_timestamp=datetime.now(),
                    metadata={"row_count": len(rows), "columns": cols}
                )

            # Wide format: each column is a biomarker (pandas not available path)
            # Try to detect biomarker columns in header
            for biomarker_name, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in cols:
                        for row in rows:
                            value = row.get(col)
                            if value is not None and value != '':
                                unit = row.get('unit', '') if 'unit' in cols else ''
                                if biomarker_name == 'EGFR':
                                    unit = 'status'
                                parsed_value = self._parse_value(str(value))
                                timestamp = datetime.now()
                                biomarker = Biomarker(
                                    name=biomarker_name,
                                    value=parsed_value,
                                    unit=str(unit),
                                    timestamp=timestamp,
                                    source_field=col,
                                    confidence=0.9
                                )
                                biomarkers.append(biomarker)
                                raw_text += f"{biomarker_name}: {parsed_value} {unit}\n"
                        break
        else:
            # Wide format: each column is a biomarker
            # Extract biomarkers from columns
            for biomarker_name, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        for idx, row in df.iterrows():
                            value = row[col]
                            if pd.notna(value):
                                # Extract unit if present
                                unit = ""
                                if biomarker_name == 'tumor_size':
                                    unit = row.get('unit', row.get('tumor_unit', 'mm'))
                                elif biomarker_name == 'CEA':
                                    unit = row.get('unit', row.get('cea_unit', 'ng/mL'))
                                elif biomarker_name == 'EGFR':
                                    unit = "status"

                                # Parse value
                                parsed_value = self._parse_value(str(value))

                                # Get timestamp
                                timestamp = datetime.now()
                                date_cols = ['date', 'timestamp', 'measurement_date', 'test_date']
                                for date_col in date_cols:
                                    if date_col in df.columns and pd.notna(row.get(date_col)):
                                        try:
                                            timestamp = pd.to_datetime(row[date_col])
                                            break
                                        except:
                                            pass

                                biomarker = Biomarker(
                                    name=biomarker_name,
                                    value=parsed_value,
                                    unit=str(unit),
                                    timestamp=timestamp,
                                    source_field=col,
                                    confidence=0.9  # High confidence for structured CSV
                                )
                                biomarkers.append(biomarker)
                                raw_text += f"{biomarker_name}: {parsed_value} {unit}\n"
                        break
        
        return ParsedClinicalData(
            patient_id=patient_id,
            biomarkers=biomarkers,
            raw_text=raw_text,
            format_type="csv",
            parse_timestamp=datetime.now(),
            metadata={"row_count": len(df), "columns": list(df.columns)}
        )
    
    def parse_text(self, file_content: bytes) -> ParsedClinicalData:
        """
        Parse plain text clinical notes using regex patterns.
        
        For production use, this would call MedGemma/Ollama for NLP extraction.
        This implementation uses regex patterns for testing purposes.
        
        Args:
            file_content: Plain text bytes
            
        Returns:
            ParsedClinicalData with extracted biomarkers
        """
        text = file_content.decode('utf-8')
        
        # Extract biomarkers using regex patterns
        biomarkers = self._extract_biomarkers_from_text(text, "text")
        
        # Extract patient ID
        patient_id = self._extract_patient_id(text)
        
        return ParsedClinicalData(
            patient_id=patient_id,
            biomarkers=biomarkers,
            raw_text=text,
            format_type="text",
            parse_timestamp=datetime.now(),
            metadata={"char_count": len(text)}
        )
    
    def normalize_biomarkers(self, raw_data: dict) -> List[Biomarker]:
        """
        Convert format-specific data to common biomarker representation.
        
        Args:
            raw_data: Dictionary with format-specific biomarker data
            
        Returns:
            List of normalized Biomarker objects
        """
        biomarkers = []
        
        for key, value in raw_data.items():
            if isinstance(value, dict) and 'value' in value:
                biomarker = Biomarker(
                    name=key,
                    value=float(value['value']),
                    unit=value.get('unit', ''),
                    timestamp=value.get('timestamp', datetime.now()),
                    source_field=key,
                    confidence=value.get('confidence', 0.8)
                )
                biomarkers.append(biomarker)
        
        return biomarkers
    
    # Helper methods
    
    def _parse_fhir_observation(self, observation: dict) -> Optional[Biomarker]:
        """Parse a FHIR Observation resource into a Biomarker."""
        # Get code text from various possible locations
        code_obj = observation.get('code', {})
        code_text = code_obj.get('text', '')
        
        # If no text, try to get from coding display
        if not code_text:
            codings = code_obj.get('coding', [])
            if codings:
                code_text = codings[0].get('display', '')
        
        # Check if this is a biomarker we care about
        code_text_lower = code_text.lower()
        biomarker_name = None
        
        if 'tumor' in code_text_lower or 'size' in code_text_lower:
            biomarker_name = 'tumor_size'
        elif 'cea' in code_text_lower or 'carcinoembryonic' in code_text_lower:
            biomarker_name = 'CEA'
        elif 'egfr' in code_text_lower or 'epidermal growth' in code_text_lower:
            biomarker_name = 'EGFR'
        
        if not biomarker_name:
            return None
        
        # Extract value
        value_quantity = observation.get('valueQuantity', {})
        value_string = observation.get('valueString', '')
        value_codeable = observation.get('valueCodeableConcept', {})
        
        if value_quantity:
            value = float(value_quantity.get('value', 0))
            unit = value_quantity.get('unit', '')
        elif value_codeable:
            # Handle coded values (like EGFR mutation status)
            text = value_codeable.get('text', '').lower()
            if 'positive' in text or 'mutation' in text or 'detected' in text:
                value = 1.0
            else:
                value = 0.0
            unit = "status"
        elif value_string:
            value = self._parse_value(value_string)
            unit = "status"
        else:
            return None
        
        # Extract timestamp
        timestamp_str = observation.get('effectiveDateTime', observation.get('issued', ''))
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
        
        return Biomarker(
            name=biomarker_name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            source_field=code_text,
            confidence=0.95  # High confidence for structured FHIR data
        )
    
    def _extract_biomarkers_from_text(self, text: str, source: str) -> List[Biomarker]:
        """Extract biomarkers from text using regex patterns."""
        biomarkers = []
        text_lower = text.lower()
        
        for biomarker_name, patterns in self.biomarker_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if biomarker_name == "EGFR":
                        # EGFR is categorical
                        status = match.group(1).lower()
                        value = 1.0 if 'positive' in status or 'mutation' in status or 'detected' in status else 0.0
                        unit = "status"
                    else:
                        # Numeric biomarkers
                        value = float(match.group(1))
                        unit = match.group(2) if len(match.groups()) > 1 else ""
                    
                    biomarker = Biomarker(
                        name=biomarker_name,
                        value=value,
                        unit=unit,
                        timestamp=datetime.now(),
                        source_field=f"{source}_regex",
                        confidence=0.7  # Lower confidence for regex extraction
                    )
                    biomarkers.append(biomarker)
                    break  # Only take first match per pattern
        
        return biomarkers
    
    def _extract_patient_id(self, text: str) -> str:
        """Extract patient ID from text."""
        patterns = [
            r"patient[:\s]+[A-Za-z\s]+\(ID[:\s]+([A-Z0-9-]+)\)",  # Patient: Name (ID: P12345)
            r"patient\s+id[:\s]+([A-Z0-9-]+)",
            r"patient[:\s]+([A-Z0-9-]+)",
            r"id[:\s]+([A-Z0-9-]+)",
            r"mrn[:\s]+([A-Z0-9-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _parse_value(self, value_str: str) -> float:
        """Parse numeric value from string."""
        # Handle categorical values
        value_lower = value_str.lower()
        if any(word in value_lower for word in ['positive', 'detected', 'mutation', 'yes']):
            return 1.0
        if any(word in value_lower for word in ['negative', 'not detected', 'wild', 'no']):
            return 0.0
        
        # Extract numeric value
        match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        if match:
            return float(match.group(1))
        
        return 0.0
