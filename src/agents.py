import json
import sys
import os
from typing import TypedDict, List
try:
    from langgraph.graph import StateGraph, END
except Exception:
    # Minimal fallbacks so module import doesn't fail when langgraph is missing.
    class END:
        pass

    class StateGraph:
        def __init__(self, state_type):
            raise RuntimeError("langgraph is not installed; StateGraph functionality is unavailable")
try:
    from langchain_community.llms import Ollama
except Exception:
    # Provide a lightweight fallback to avoid ImportError during module import.
    class Ollama:
        def __init__(self, model: str = None, temperature: float = 0):
            self.model = model
            self.temperature = temperature

        def invoke(self, prompt: str) -> str:
            return "{}"

# Add current directory to path to ensure absolute imports work on Windows
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Robust import handling for local package structure
try:
    from src.vector_store import GuidelineVectorStore
except ImportError:
    try:
        from vector_store import GuidelineVectorStore
    except ImportError:
        # Fallback if names were changed in earlier versions
        from src.vector_store import GuidelineStore as GuidelineVectorStore

# 1. State Definition (Aligned with streamlit_app.py)
class AgentState(TypedDict):
    history: str
    biomarkers: dict
    relevant_guidelines: List[str]
    audit_findings: str  
    is_critical: bool
    # New fields for input parsing
    file_content: bytes
    filename: str
    parsed_data: dict  # ParsedClinicalData as dict
    # New fields for safety auditor
    safety_result: dict  # ComparisonResult as dict
    human_approved: bool  # Whether human approved conflicts

class SentinelOrchestrator:
    """
    The main orchestrator for the Sentinel system.
    Handles extraction, retrieval, and clinical auditing.
    """
    def __init__(self, model_name: str = "gemma3:4b"):
        # Local Ollama instance - ensure 'ollama serve' is running
        self.llm = Ollama(model=model_name, temperature=0)
        
        try:
            # Initialize local LanceDB
            self.store = GuidelineVectorStore()
        except Exception as e:
            print(f"Warning: Guideline Store failed to init: {e}")
            self.store = None
        
        # Initialize input parser
        try:
            from src.input_parser.parser import InputParser
            self.parser = InputParser()
        except ImportError:
            try:
                from input_parser.parser import InputParser
                self.parser = InputParser()
            except ImportError:
                print("Warning: InputParser not available")
                self.parser = None
        
        # Initialize safety auditor
        try:
            from src.safety_auditor.auditor import SafetyAuditor
            self.safety_auditor = SafetyAuditor(model_name=model_name)
        except ImportError:
            try:
                from safety_auditor.auditor import SafetyAuditor
                self.safety_auditor = SafetyAuditor(model_name=model_name)
            except ImportError:
                print("Warning: SafetyAuditor not available")
                self.safety_auditor = None

    def parse_input_node(self, state: AgentState):
        """Parse input file and extract clinical data."""
        # Check if we have file content to parse
        if not state.get('file_content') or not state.get('filename'):
            # No file to parse, skip this node
            return {}
        
        if not self.parser:
            return {
                "parsed_data": {
                    "error": "InputParser not available",
                    "patient_id": "unknown",
                    "biomarkers": [],
                    "raw_text": state.get('history', ''),
                    "format_type": "unknown"
                }
            }
        
        try:
            # Parse the file
            parsed = self.parser.parse(
                file_content=state['file_content'],
                filename=state['filename']
            )
            
            # Convert ParsedClinicalData to dict for state
            parsed_dict = {
                "patient_id": parsed.patient_id,
                "biomarkers": [
                    {
                        "name": b.name,
                        "value": b.value,
                        "unit": b.unit,
                        "timestamp": b.timestamp.isoformat(),
                        "source_field": b.source_field,
                        "confidence": b.confidence,
                        "biomarker_type": b.biomarker_type.value if b.biomarker_type else None
                    }
                    for b in parsed.biomarkers
                ],
                "raw_text": parsed.raw_text,
                "format_type": parsed.format_type,
                "parse_timestamp": parsed.parse_timestamp.isoformat(),
                "metadata": parsed.metadata
            }
            
            # Update history with parsed text for downstream nodes
            return {
                "parsed_data": parsed_dict,
                "history": parsed.raw_text
            }
            
        except Exception as e:
            # Handle parsing errors gracefully
            error_msg = str(e)
            recovery_steps = []
            
            if "Unable to detect format" in error_msg or "Unsupported format" in error_msg:
                recovery_steps = [
                    "Verify file format is one of: PDF, FHIR JSON, Synthea JSON, CSV, plain text",
                    "Check that file is not corrupted",
                    "Ensure file contains clinical data with biomarkers"
                ]
            elif "Failed to parse PDF" in error_msg:
                recovery_steps = [
                    "Ensure PDF is not password-protected",
                    "Verify PDF contains extractable text (not scanned images)",
                    "Try converting PDF to text format first"
                ]
            else:
                recovery_steps = [
                    "Check file integrity",
                    "Review error details in logs",
                    "Try a different file format"
                ]
            
            return {
                "parsed_data": {
                    "error": error_msg,
                    "recovery_steps": recovery_steps,
                    "patient_id": "unknown",
                    "biomarkers": [],
                    "raw_text": "",
                    "format_type": "unknown"
                }
            }

    def safety_auditor_node(self, state: AgentState):
        """Perform dual extraction to detect conflicts and hallucinations."""
        # Skip if no parsed data or if there was a parsing error
        parsed_data = state.get('parsed_data', {})
        if not parsed_data or 'error' in parsed_data:
            return {}
        
        if not self.safety_auditor:
            return {
                "safety_result": {
                    "error": "SafetyAuditor not available",
                    "requires_human_review": False
                }
            }
        
        try:
            # Get the raw clinical text
            clinical_text = parsed_data.get('raw_text', state.get('history', ''))
            
            if not clinical_text:
                return {
                    "safety_result": {
                        "error": "No clinical text available for safety check",
                        "requires_human_review": False
                    }
                }
            
            # Perform dual extraction
            extraction_a = self.safety_auditor.extract_with_prompt_a(clinical_text)
            extraction_b = self.safety_auditor.extract_with_prompt_b(clinical_text)
            
            # Compare extractions
            comparison = self.safety_auditor.compare_extractions(extraction_a, extraction_b)
            
            # Convert ComparisonResult to dict for state
            safety_result = {
                "conflicts": [
                    {
                        "biomarker_name": c.biomarker_name,
                        "value_a": c.value_a,
                        "value_b": c.value_b,
                        "unit": c.unit,
                        "discrepancy_percentage": c.discrepancy_percentage
                    }
                    for c in comparison.conflicts
                ],
                "all_biomarkers": [
                    {
                        "name": b.name,
                        "value": b.value,
                        "unit": b.unit,
                        "confidence": b.confidence,
                        "source_field": b.source_field
                    }
                    for b in comparison.all_biomarkers
                ],
                "requires_human_review": comparison.requires_human_review,
                "overall_confidence": comparison.overall_confidence,
                "extraction_a": [
                    {
                        "name": b.name,
                        "value": b.value,
                        "unit": b.unit
                    }
                    for b in extraction_a
                ],
                "extraction_b": [
                    {
                        "name": b.name,
                        "value": b.value,
                        "unit": b.unit
                    }
                    for b in extraction_b
                ]
            }
            
            return {"safety_result": safety_result}
            
        except Exception as e:
            return {
                "safety_result": {
                    "error": f"Safety audit failed: {str(e)}",
                    "requires_human_review": True
                }
            }

    def human_review_node(self, state: AgentState):
        """Placeholder node for human review - actual review happens in UI."""
        # This node is just a placeholder - the actual human review
        # happens in the Streamlit UI via the interrupt mechanism
        return {}

    def extractor(self, state: AgentState):
        """Agent A: Extract clinical biomarkers as structured JSON."""
        prompt = f"""
        Extract clinical biomarkers (specifically Tumor size in mm, CEA levels, and EGFR mutation status) 
        as a valid JSON object from the following history:
        
        {state['history']}
        
        Return ONLY the JSON. If a value is missing, use "Not Documented".
        """
        try:
            res = self.llm.invoke(prompt)
            # Clean potential markdown formatting
            clean_res = res.replace('```json', '').replace('```', '').strip()
            biomarkers = json.loads(clean_res)
        except Exception as e:
            # Handle ConnectionError or ParsingError gracefully
            error_msg = str(e)
            if "Connection refused" in error_msg or "11434" in error_msg:
                biomarkers = {"error": "Ollama is not running. Please run 'ollama serve' in your terminal."}
            else:
                biomarkers = {"error": "Extraction failed", "details": error_msg}
            
        return {"biomarkers": biomarkers}

    def researcher(self, state: AgentState):
        """Agent B: Retrieval of relevant NCCN protocols."""
        if "error" in state.get('biomarkers', {}):
            return {"relevant_guidelines": ["Skipped due to extraction error."]}

        if not self.store:
            return {"relevant_guidelines": ["RAG Store offline. Using generic NCCN NSCLC baseline."]}
            
        query_text = f"NSCLC treatment guidelines for patient with markers: {state['biomarkers']}"
        try:
            hits = self.store.search(query_text)
            guideline_texts = hits['text'].tolist() if not (hits is None or hits.empty) else ["No matching guidelines found."]
        except Exception as e:
            guideline_texts = [f"Guideline lookup error: {str(e)}"]
            
        return {"relevant_guidelines": guideline_texts}

    def auditor(self, state: AgentState):
        """Agent C: Clinical Audit against RECIST 1.1 criteria."""
        if "error" in state.get('biomarkers', {}):
            return {"audit_findings": f"Audit aborted: {state['biomarkers']['error']}", "is_critical": True}

        prompt = f"""
        AUDIT ROLE: Analyze patient biomarkers against clinical guidelines.
        Patient Stats: {state['biomarkers']}
        Reference Guidelines: {state['relevant_guidelines']}
        
        TASK: 
        1. Check if tumor size increase meets RECIST 1.1 (>20%) for Progressive Disease.
        2. Identify if current treatment plan aligns with NCCN guidelines.
        3. Flag any discrepancies clearly.
        """
        try:
            res = self.llm.invoke(prompt)
            # Determine if findings are critical for UI highlighting
            is_crit = any(word in res.upper() for word in ["CRITICAL", "DEVIATION", "GROWTH", "RISK", "PD"])
        except Exception as e:
            res = f"Audit failed to connect to Ollama: {str(e)}"
            is_crit = True
            
        return {"audit_findings": res, "is_critical": is_crit}

    def build_graph(self):
        """
        Constructs and compiles the stateful LangGraph workflow.
        """
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("parse_input", self.parse_input_node)
        workflow.add_node("safety_auditor", self.safety_auditor_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("extract", self.extractor)
        workflow.add_node("research", self.researcher)
        workflow.add_node("audit", self.auditor)
        
        # Define conditional routing function
        def route_after_safety(state: AgentState):
            """Route to human review if conflicts detected, otherwise continue."""
            safety_result = state.get("safety_result", {})
            
            # Check if human review is required
            if safety_result.get("requires_human_review", False):
                # Check if human has already approved
                if state.get("human_approved", False):
                    return "extract"
                else:
                    return "human_review"
            else:
                return "extract"
        
        # Define Flow
        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "safety_auditor")
        
        # Conditional routing after safety auditor
        workflow.add_conditional_edges(
            "safety_auditor",
            route_after_safety,
            {
                "human_review": "human_review",
                "extract": "extract"
            }
        )
        
        # After human review, continue to extract
        workflow.add_edge("human_review", "extract")
        
        workflow.add_edge("extract", "research")
        workflow.add_edge("research", "audit")
        workflow.add_edge("audit", END)
        
        return workflow.compile()

def get_workflow():
    """
    Helper function to get a compiled workflow instance.
    """
    orchestrator = SentinelOrchestrator()
    return orchestrator.build_graph()