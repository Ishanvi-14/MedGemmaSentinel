import streamlit as st
import pandas as pd
import plotly.express as px
from src.agents import SentinelOrchestrator
from src.data_loader import MedicalDataLoader

st.set_page_config(page_title="Sentinel | MedGemma Auditor", layout="wide")

# Initialize Backend
orchestrator = SentinelOrchestrator()
loader = MedicalDataLoader()

st.title("🛡️ Sentinel: Agentic Oncology Auditor")
st.markdown("---")

# Sidebar - Evidence Trace & Files
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Patient History (JSON)", type="json")
    
    st.header("🔍 Evidence Trace")
    st.info("Agentic reasoning logs will appear here after analysis.")

# Main Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file:
        # 1. Process Data
        raw_text = loader.parse_synthea_record(uploaded_file)
        
        with st.status("Agentic Workflow in Progress...", expanded=True) as status:
            st.write("Running MedGemma Extractor...")
            # This triggers the LangGraph workflow
            graph = orchestrator.build_graph()
            result = graph.invoke({"raw_history": raw_text})
            status.update(label="Analysis Complete!", state="complete")

        # 2. Visualize Trends
        st.subheader("📈 Longitudinal Biomarker Trends")
        # Mocking a trend for visual impact
        df = pd.DataFrame({
            "Date": ["2024-01", "2024-06", "2024-12"],
            "Tumor Size (mm)": [15, 18, 24] # Should come from 'result'
        })
        fig = px.line(df, x="Date", y="Tumor Size (mm)", markers=True, title="Tumor Progression")
        st.plotly_chart(fig, use_container_width=True)

        # 3. Audit Findings
        st.subheader("🚨 Clinical Audit Report")
        if result.get("is_critical"):
            st.error(result["audit_findings"])
        else:
            st.success(result["audit_findings"])

with col2:
    st.subheader("📑 Guideline RAG Context")
    if 'result' in locals():
        for i, text in enumerate(result.get("relevant_guidelines", [])):
            with st.expander(f"NCCN Source {i+1}"):
                st.write(text)