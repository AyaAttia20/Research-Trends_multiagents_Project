# research_trends_app.py
import streamlit as st
from research_agents import create_crew

st.set_page_config(page_title="📊 Research Trends Tracker", layout="wide")
st.title("📈 Research Trends Tracker")
st.markdown("Get insights into trending research topics, papers, and top authors using agents and LLMs.")

user_input = st.text_input("🔍 Enter a research topic:", value="generative AI")

if st.button("🚀 Run Research Agents"):
    with st.spinner("Running multi-agent system..."):
        crew = create_crew()
        final_output = crew.kickoff(inputs={"topic": user_input})

    st.success("✅ Analysis Complete!")

    sections = {
        "Fetched Papers": "📚 Papers",
        "Trends": "📈 Trending Keywords",
        "Authors": "👩\u200d🔬 Top Authors"
    }

    for section, title in sections.items():
        st.subheader(title)
        try:
            part = final_output.split(f"[{section}]")[1].split("[")[0].strip()
            for line in part.split("\n"):
                st.markdown(f"- {line.strip('- ')}")
        except IndexError:
            st.warning(f"{section} section not found in output.")
