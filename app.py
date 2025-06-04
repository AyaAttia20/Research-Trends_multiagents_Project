# Ensure pysqlite3 compatibility with LangChain
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Core imports
import streamlit as st
import requests
import warnings
import pandas as pd
import plotly.express as px
import re

# CrewAI & LangChain
from crewai import Agent, Task, Crew
from langchain.chat_models import HuggingFaceChat
from langchain.tools import Tool

warnings.filterwarnings("ignore")

# ----------------------------
# OpenAlex API Wrapper
# ----------------------------
def fetch_openalex(topic):
    url = "https://api.openalex.org/works"
    params = {
        "search": topic,
        "filter": "publication_year:2024",
        "sort": "cited_by_count:desc",
        "per-page": 10
    }

    response = requests.get(url, params=params)
    try:
        data = response.json()
    except:
        return []

    results = []
    for work in data.get("results", []):
        title = work.get("title", "Untitled")
        abstract = work.get("abstract", "⚠️ Abstract not available.")
        authors = [a["author"]["display_name"] for a in work.get("authorships", [])]
        paper_url = work.get("id", "")
        results.append({
            "title": title,
            "summary": abstract,
            "authors": authors,
            "url": paper_url
        })

    return results

def fetch_wrapper(input):
    if isinstance(input, dict):
        topic = input.get("topic") or next(iter(input.values()))
    else:
        topic = input
    return fetch_openalex(topic)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Research Trends Tracker (Free Edition)", layout="wide")
st.title("🔬 Research Trends Tracker (OpenAlex + HuggingFace Free)")

topic = st.text_input("📚 Enter a research topic", "Artificial Intelligence")
run_button = st.button("🚀 Run Analysis")

if run_button:
    with st.spinner("Running multi-agent crew with HuggingFaceChat..."):

        try:
            # Initialize HuggingFaceChat with a free, open-source model (no API key needed)
            llm = HuggingFaceChat(model="google/flan-t5-xl")  # Or try "bigscience/bloom" or others supported

        except Exception as e:
            st.error(f"❌ Failed to initialize LLM: {str(e)}")
            st.stop()

        tool = Tool(
            name="OpenAlexFetcher",
            func=fetch_wrapper,
            description="Fetches latest research papers using OpenAlex API",
            return_direct=True
        )

        fetcher = Agent(
            role="Research Fetcher",
            goal="Fetch recent research papers on a topic",
            tools=[tool],
            backstory="Expert at finding papers.",
            verbose=True,
            llm=llm
        )

        analyzer = Agent(
            role="Trend Analyzer",
            goal="Extract trending keywords",
            backstory="Skilled in NLP to detect research trends.",
            verbose=True,
            llm=llm
        )

        reporter = Agent(
            role="Author Reporter",
            goal="Identify top authors and institutions",
            backstory="Expert in mapping contributors in academia.",
            verbose=True,
            llm=llm
        )

        fetch_task = Task(
            description=f"Fetch latest 10 research papers on '{topic}'.",
            expected_output="List of papers with title, abstract, authors, and URL.",
            agent=fetcher
        )

        trend_task = Task(
            description="Extract top 10 trending keywords from titles and abstracts.",
            expected_output="List of 10 trending keywords with short explanations.",
            agent=analyzer,
            context=[fetch_task]
        )

        author_task = Task(
            description="List top 5 authors by frequency from papers. Format: Name – Institution (N papers).",
            expected_output="List of top 5 authors and institutions.",
            agent=reporter,
            context=[fetch_task]
        )

        crew = Crew(
            agents=[fetcher, analyzer, reporter],
            tasks=[fetch_task, trend_task, author_task],
            verbose=True
        )

        try:
            inputs = {"topic": topic}
            result = crew.kickoff(inputs=inputs)
            st.success("✅ Analysis Complete!")

            # Display Papers
            st.markdown("## 📚 Latest Papers")
            raw_output = str(fetch_task.output)
            try:
                papers = eval(raw_output) if raw_output.startswith("[{") else []
            except:
                papers = []

            if papers:
                for paper in papers[:10]:
                    with st.expander(f"📄 {paper['title']}"):
                        st.markdown(f"**🧠 Abstract:** {paper['summary']}")
                        st.markdown(f"**✍️ Authors:** {', '.join(paper['authors'])}")
                        st.markdown(f"[🔗 Read More]({paper['url']})")
            else:
                st.warning("⚠️ Could not parse papers.")

            # Display Trending Keywords
            st.markdown("## 📈 Trending Keywords")
            trend_text = str(trend_task.output)
            trend_lines = trend_text.strip().split("\n")
            keywords = []
            for line in trend_lines:
                match = re.match(r"[-•]\s*(.+?):", line)
                if match:
                    keywords.append(match.group(1))

            if keywords:
                df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
                fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ No keywords found to visualize.")

            # Display Authors
            st.markdown("## 👩‍🔬 Top Authors and Institutions")
            author_lines = str(author_task.output).split("\n")
            shown = 0
            for line in author_lines:
                if "–" in line and shown < 5:
                    st.markdown(f"👤 **{line.strip()}**")
                    shown += 1
            if shown == 0:
                st.info("⚠️ No authors data found.")

        except Exception as e:
            st.error(f"❌ Error during execution: {str(e)}")
