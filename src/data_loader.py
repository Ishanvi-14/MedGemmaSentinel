import json
import io
from typing import Union

class MedicalDataLoader:
    """
    Handles medical data ingestion from local paths or Streamlit buffers.
    """
    
    def parse_synthea_record(self, data_input) -> str:
        """
        Parses Synthea JSON. Handles both file paths (str) and 
        Streamlit UploadedFile (file-like) objects.
        """
        try:
            # Check if input is a string (path) or a file-like object (Streamlit buffer)
            if hasattr(data_input, 'read'):
                # It's an UploadedFile/BytesIO object
                data = json.load(data_input)
                # Reset buffer position for safety in case of multiple reads
                data_input.seek(0)
            elif isinstance(data_input, (str, bytes)):
                # It's a file path
                with open(data_input, 'r') as f:
                    data = json.load(f)
            else:
                return "Error: Unsupported input type. Expected path or file buffer."
        except Exception as e:
            return f"Error loading clinical JSON: {str(e)}"
        
        narrative = "Patient Clinical History Summary:\n"
        # Extract entries safely
        entries = data.get('entry', [])
        if not entries:
            return narrative + "No clinical entries found in this record."

        for entry in entries:
            resource = entry.get('resource', {})
            res_type = resource.get('resourceType')
            
            if res_type == "Observation":
                code = resource.get('code', {}).get('text', 'Observation')
                val = resource.get('valueQuantity', {}).get('value', 'N/A')
                unit = resource.get('valueQuantity', {}).get('unit', '')
                narrative += f"- {code}: {val} {unit}\n"
            
            elif res_type == "MedicationRequest":
                med = resource.get('medicationCodeableConcept', {}).get('text', 'Medication')
                narrative += f"- Medication prescribed: {med}\n"
                
        return narrative