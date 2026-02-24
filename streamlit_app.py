import streamlit as st
import pandas as pd
import plotly.express as px
from src.agents import SentinelOrchestrator
from src.data_loader import MedicalDataLoader

st.set_page_config(page_title="Sentinel | MedGemma Auditor", layout="wide")

# Initialize session state for human approval
if 'human_approved' not in st.session_state:
    st.session_state.human_approved = False
if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None

# Initialize Backend
orchestrator = SentinelOrchestrator()
loader = MedicalDataLoader()

st.title("🛡️ Sentinel: Agentic Oncology Auditor")
st.markdown("---")

# Sidebar - Evidence Trace & Files
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Patient History",
        type=["json", "pdf", "csv", "txt"],
        help="Supported formats: PDF, FHIR JSON, Synthea JSON, CSV, plain text"
    )
    
    # Display detected format if file is uploaded
    if uploaded_file:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        try:
            from src.input_parser.parser import InputParser
            parser = InputParser()
            detected_format = parser.detect_format(file_bytes, uploaded_file.name)
            
            # Format display mapping
            format_display = {
                "pdf": "📄 PDF Clinical Report",
                "fhir_json": "🏥 HL7 FHIR JSON",
                "synthea_json": "🔬 Synthea JSON",
                "csv": "📊 CSV Lab Results",
                "text": "📝 Plain Text Notes"
            }
            
            st.success(f"Detected: {format_display.get(detected_format, detected_format)}")
        except Exception as e:
            st.warning(f"Format detection: {str(e)}")
    
    st.header("🔍 Evidence Trace")
    st.info("Agentic reasoning logs will appear here after analysis.")

# Main Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file:
        # Read file content
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential re-reading
        
        with st.status("Agentic Workflow in Progress...", expanded=True) as status:
            st.write("📄 Parsing input file...")
            
            # Build workflow graph
            graph = orchestrator.build_graph()
            
            # Invoke workflow with file content
            try:
                st.write("🛡️ Running Safety Auditor (dual extraction)...")
                
                result = graph.invoke({
                    "file_content": file_bytes,
                    "filename": uploaded_file.name,
                    "history": "",  # Will be populated by parse_input_node
                    "biomarkers": {},
                    "relevant_guidelines": [],
                    "audit_findings": "",
                    "is_critical": False,
                    "human_approved": st.session_state.human_approved
                })
                
                # Check for parsing errors
                parsed_data = result.get("parsed_data", {})
                if "error" in parsed_data:
                    status.update(label="Parsing Failed", state="error")
                    st.error(f"**Parsing Error:** {parsed_data['error']}")
                    
                    if "recovery_steps" in parsed_data:
                        st.write("**Recovery Steps:**")
                        for step in parsed_data["recovery_steps"]:
                            st.write(f"- {step}")
                else:
                    status.update(label="Analysis Complete!", state="complete")
                    
                    # Display parsed data info
                    st.write(f"✅ Parsed {parsed_data.get('format_type', 'unknown')} format")
                    st.write(f"📋 Patient ID: {parsed_data.get('patient_id', 'unknown')}")
                    st.write(f"🔬 Extracted {len(parsed_data.get('biomarkers', []))} biomarkers")
                    
                    # Show safety auditor status
                    safety_result = result.get("safety_result", {})
                    if safety_result and "error" not in safety_result:
                        conflicts = safety_result.get("conflicts", [])
                        if conflicts:
                            st.write(f"⚠️ Safety Auditor: {len(conflicts)} conflict(s) detected")
                        else:
                            st.write(f"✅ Safety Auditor: No conflicts detected")
                    
                    st.write(f"🔍 Retrieved {len(result.get('relevant_guidelines', []))} guidelines")
                    st.write(f"📊 Clinical audit complete")
                    
            except Exception as e:
                status.update(label="Workflow Failed", state="error")
                st.error(f"**Workflow Error:** {str(e)}")
                result = None

        # Only proceed with visualization if we have valid results
        if result and "error" not in result.get("parsed_data", {}):
            # Display Safety Auditor Results
            safety_result = result.get("safety_result", {})
            
            if safety_result and "error" not in safety_result:
                st.subheader("🛡️ Safety Auditor - Dual Extraction Analysis")
                
                # Display overall confidence
                confidence = safety_result.get("overall_confidence", 0.0)
                confidence_pct = confidence * 100
                
                col_conf1, col_conf2 = st.columns([1, 3])
                with col_conf1:
                    st.metric("Overall Confidence", f"{confidence_pct:.1f}%")
                with col_conf2:
                    if confidence >= 0.8:
                        st.success("✅ High confidence - extractions agree")
                    elif confidence >= 0.6:
                        st.warning("⚠️ Medium confidence - minor discrepancies")
                    else:
                        st.error("🚨 Low confidence - significant conflicts detected")
                
                # Display conflicts if any
                conflicts = safety_result.get("conflicts", [])
                if conflicts:
                    st.subheader("⚠️ Detected Conflicts")
                    st.warning(f"Found {len(conflicts)} conflict(s) requiring review")
                    
                    for i, conflict in enumerate(conflicts):
                        with st.expander(f"Conflict {i+1}: {conflict['biomarker_name']}", expanded=True):
                            col_a, col_b, col_disc = st.columns(3)
                            
                            with col_a:
                                st.markdown("**Extraction A**")
                                if conflict['unit'] == 'status':
                                    value_a_display = "Positive" if conflict['value_a'] == 1.0 else "Negative"
                                    st.info(f"**{value_a_display}**")
                                else:
                                    st.info(f"**{conflict['value_a']:.2f}** {conflict['unit']}")
                            
                            with col_b:
                                st.markdown("**Extraction B**")
                                if conflict['unit'] == 'status':
                                    value_b_display = "Positive" if conflict['value_b'] == 1.0 else "Negative"
                                    st.info(f"**{value_b_display}**")
                                else:
                                    st.info(f"**{conflict['value_b']:.2f}** {conflict['unit']}")
                            
                            with col_disc:
                                st.markdown("**Discrepancy**")
                                st.error(f"**{conflict['discrepancy_percentage']:.1f}%**")
                
                # Display side-by-side comparison of all biomarkers
                st.subheader("🔬 Side-by-Side Extraction Comparison")
                
                extraction_a = safety_result.get("extraction_a", [])
                extraction_b = safety_result.get("extraction_b", [])
                
                if extraction_a or extraction_b:
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown("**🅰️ Extraction A (Direct Prompt)**")
                        if extraction_a:
                            for biomarker in extraction_a:
                                if biomarker['unit'] == 'status':
                                    value_display = "Positive" if biomarker['value'] == 1.0 else "Negative"
                                    st.write(f"- **{biomarker['name']}**: {value_display}")
                                else:
                                    st.write(f"- **{biomarker['name']}**: {biomarker['value']} {biomarker['unit']}")
                        else:
                            st.write("No biomarkers extracted")
                    
                    with col_right:
                        st.markdown("**🅱️ Extraction B (Contextual Prompt)**")
                        if extraction_b:
                            for biomarker in extraction_b:
                                if biomarker['unit'] == 'status':
                                    value_display = "Positive" if biomarker['value'] == 1.0 else "Negative"
                                    st.write(f"- **{biomarker['name']}**: {value_display}")
                                else:
                                    st.write(f"- **{biomarker['name']}**: {biomarker['value']} {biomarker['unit']}")
                        else:
                            st.write("No biomarkers extracted")
                
                # Human review panel if required
                if safety_result.get("requires_human_review", False):
                    st.subheader("👤 Human Review Required")
                    st.warning("⚠️ Conflicts or low confidence detected. Please review the extractions above.")
                    
                    # Display merged biomarkers with confidence scores
                    all_biomarkers = safety_result.get("all_biomarkers", [])
                    if all_biomarkers:
                        st.markdown("**Merged Biomarkers (with confidence scores):**")
                        for biomarker in all_biomarkers:
                            conf_pct = biomarker.get('confidence', 0.0) * 100
                            conf_color = "🟢" if conf_pct >= 80 else "🟡" if conf_pct >= 60 else "🔴"
                            
                            if biomarker['unit'] == 'status':
                                value_display = "Positive" if biomarker['value'] == 1.0 else "Negative"
                                st.write(f"{conf_color} **{biomarker['name']}**: {value_display} (confidence: {conf_pct:.1f}%)")
                            else:
                                st.write(f"{conf_color} **{biomarker['name']}**: {biomarker['value']:.2f} {biomarker['unit']} (confidence: {conf_pct:.1f}%)")
                    
                    # Approve/Continue button
                    if st.button("✅ Approve & Continue", type="primary", use_container_width=True):
                        st.session_state.human_approved = True
                        st.success("✅ Approved! Continuing with workflow...")
                        st.rerun()
                else:
                    st.success("✅ No human review required - extractions are consistent")
            
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
    if uploaded_file and result and "error" not in result.get("parsed_data", {}):
        for i, text in enumerate(result.get("relevant_guidelines", [])):
            with st.expander(f"NCCN Source {i+1}"):
                st.write(text)