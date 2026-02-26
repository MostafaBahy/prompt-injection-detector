
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")  # default to local
API_KEY = os.getenv("API_KEY", "change-me")

st.set_page_config(page_title="Prompt Injection Detector", page_icon="🛡️")
st.title("🛡️ Prompt Injection Detector")

text = st.text_area("Enter text to analyze:", height=150)

if st.button("Check Text", type="primary"):
    if text:
        with st.spinner("Analyzing..."):
            try:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                response = requests.post(f"{API_URL}/detect", headers=headers, json={"prompt": text})
                response.raise_for_status()
                result = response.json()["response"]

                label      = result.get("label",      "Unknown")
                risk_score = result.get("risk_score", 0)
                severity   = result.get("severity",   "Unknown")

                if severity == "CRITICAL":
                    st.error(f"🚨 **{label}** — Critical threat detected")
                elif severity == "HIGH":
                    st.error(f"⚠️ **{label}**")
                elif severity == "MEDIUM":
                    st.warning(f"⚡ **{label}**")
                else:  # LOW / Unknown
                    st.success(f"✅ **{label}**")

                # Metrics row — confidence replaces risk_score
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("risk_score", f"{risk_score}%")
                with col2:
                    st.metric("Severity", severity)

                # Reasoning
                if result.get("reason"):
                    st.info(f"**Reason:** {result['reason']}")

                if result.get("pattern_details") and result["pattern_details"] != "N/A":
                    st.write("**Pattern Details:**")
                    st.caption(result["pattern_details"])

                patterns = result.get("detected_patterns", [])
                has_real_patterns = patterns and patterns != ["None"]
                if has_real_patterns:
                    st.write("**Detected Patterns:**")
                    for pattern in patterns:
                        st.write(f"- {pattern}")

            except requests.exceptions.HTTPError as e:
                st.error(f"API error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text to analyze")