import json
import sys
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama

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
        workflow.add_node("extract", self.extractor)
        workflow.add_node("research", self.researcher)
        workflow.add_node("audit", self.auditor)
        
        # Define Flow
        workflow.set_entry_point("extract")
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