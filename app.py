import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import requests
import warnings
import pandas as pd
import re
import plotly.express as px
import feedparser
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

warnings.filterwarnings("ignore")

# -------------------------
# Semantic Scholar Fetcher
# -------------------------
def fetch_semantic_scholar(topic, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={limit}&fields=title,abstract,authors,url,year"
    response = requests.get(url)
    data = response.json()
    papers = []

    for item in data.get("data", []):
        papers.append({
            "title": item.get("title", "No title"),
            "summary": item.get("abstract", "No abstract"),
            "authors": [author.get("name") for author in item.get("authors", [])],
            "url": item.get("url", "#"),
            "year": item.get("year", "Unknown")
        })
    return papers

def semantic_scholar_wrapper(input):
    topic = input["topic"] if isinstance(input, dict) else input
    return fetch_semantic_scholar(topic)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Research Trends Tracker", layout="wide")
st.title("🔬 Research Trends Tracker")

api_key = st.text_input("🔑 Enter your OpenRouter API Key", type="password")
topic = st.text_input("📚 Enter a research topic", "Artificial Intelligence")
run_button = st.button("🚀 Run Analysis")

if run_button:
    if not api_key:
        st.error("❌ Please enter your OpenRouter API Key before running.")
    elif not api_key.startswith("sk-") and len(api_key) < 20:
        st.error("⚠️ Invalid API key format. Please double-check your OpenRouter API key.")
    else:
        with st.spinner("Running multi-agent crew..."):

            try:
                llm = ChatOpenAI(
                    model_name="mistralai/mistral-7b-instruct",
                    base_url="https://openrouter.ai/api/v1", 
                    api_key=api_key,
                    temperature=0.7
                )
            except Exception as e:
                st.error(f"❌ Failed to initialize LLM: {str(e)}")
                st.stop()

            # Tool
            fetch_tool = Tool(
                name="SemanticScholarFetcher",
                func=semantic_scholar_wrapper,
                description="Fetches recent research papers from Semantic Scholar given a topic string",
                return_direct=True
            )

            # Agents
            fetcher = Agent(
                role="Research Fetcher",
                goal="Find the latest research papers on a given topic",
                tools=[fetch_tool],
                backstory="Expert at gathering academic papers from Semantic Scholar.",
                verbose=True,
                llm=llm
            )

            analyzer = Agent(
                role="Trend Analyzer",
                goal="Analyze recent papers and extract trending keywords and hot topics",
                backstory="Skilled in text mining and NLP to extract useful trends.",
                verbose=True,
                llm=llm
            )

            reporter = Agent(
                role="Author & Institution Reporter",
                goal="Find top authors and institutions publishing in the research field",
                backstory="Specializes in identifying the key contributors in academic fields.",
                verbose=True,
                llm=llm
            )

            # Tasks
            fetch_task = Task(
                description=f"Fetch recent research papers from Semantic Scholar for the topic '{topic}'.",
                expected_output="A list of paper titles, abstracts, authors, and links.",
                agent=fetcher
            )

            trend_task = Task(
                description=(
                    f"Analyze the list of research papers (title, summary, authors) on the topic '{topic}' "
                    "and identify the top 5-10 trending keywords or research themes in bullet points."
                ),
                expected_output="A list of 5–10 trending research topics or keywords with descriptions.",
                agent=analyzer,
                context=[fetch_task]
            )

            author_task = Task(
                description=(
                    f"Based on the list of paper titles, authors, and summaries for the topic '{topic}', "
                    "identify the most frequently mentioned authors. Include affiliations if available."
                ),
                expected_output=(
                    "- A list of top 3–5 authors, their affiliations if possible, and publication counts.\n"
                    "- Format: Author Name – Institution (N papers)"
                ),
                agent=reporter,
                context=[fetch_task]
            )

            # Crew Setup
            crew = Crew(
                agents=[fetcher, analyzer, reporter],
                tasks=[fetch_task, trend_task, author_task],
                verbose=True
            )

            try:
                inputs = {"topic": topic}
                result = crew.kickoff(inputs=inputs)

                st.success("✅ Analysis Complete!")

                # 1. Display fetched papers
                st.markdown("## 📚 Research Papers")
                try:
                    papers_raw = eval(fetch_task.output.value if hasattr(fetch_task.output, "value") else fetch_task.output)
                    for paper in papers_raw[:10]:
                        st.markdown(f"📄 **[{paper['title']}]({paper['url']})**  \n📝 {paper['summary'][:300]}...  \n👨‍🔬 _Authors_: {', '.join(paper['authors'])}  \n📅 _Year_: {paper['year']}")
                        st.markdown("---")
                except Exception as e:
                    st.warning(f"Could not parse papers: {e}")

                # 2. Keyword Trends Chart
                st.markdown("## 📈 Trending Keywords")
                trend_lines = trend_task.output.value.split("\n") if hasattr(trend_task.output, "value") else trend_task.output.split("\n")
                keyword_data = []
                for line in trend_lines:
                    match = re.match(r"[-•]?\s*(.+?):", line.strip())
                    if match:
                        keyword_data.append(match.group(1).strip())

                if keyword_data:
                    df_keywords = pd.DataFrame({'Keyword': keyword_data[:10]})
                    fig = px.bar(df_keywords['Keyword'].value_counts().reset_index(), x='index', y='Keyword',
                                 labels={'index': 'Keyword', 'Keyword': 'Count'},
                                 title="Top Trending Keywords", text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⚠️ No keywords found to visualize.")

                # 3. Authors
                st.markdown("## 👩‍🔬 Top Authors and Institutions")
                author_lines = author_task.output.value.split("\n") if hasattr(author_task.output, "value") else author_task.output.split("\n")
                for line in author_lines:
                    if "–" in line:
                        st.markdown(f"👤 **{line.strip()}**")

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
