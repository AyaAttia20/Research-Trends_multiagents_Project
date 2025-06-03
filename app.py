import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import requests
import feedparser
import warnings
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

import pandas as pd
import re
import plotly.express as px

warnings.filterwarnings("ignore")

def fetch_arxiv(topic):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=10"
    response = requests.get(url)
    feed = feedparser.parse(response.text)
    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "authors": [author.name for author in entry.authors],
            "url": entry.link
        })
    return results

def arxiv_fetch_wrapper(input):
    if isinstance(input, dict):
        topic = input.get("topic") or input.get("subject_area") or next(iter(input.values()))
    else:
        topic = input
    return fetch_arxiv(topic)


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

            arxiv_tool = Tool(
                name="ArxivResearchFetcher",
                func=arxiv_fetch_wrapper,
                description="Fetches recent research papers from arXiv given a topic string or dict",
                return_direct=True
            )

            fetcher = Agent(
                role="Research Fetcher",
                goal="Find the latest research papers on a given topic",
                tools=[arxiv_tool],
                backstory="Expert at gathering academic papers using arXiv.",
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

            fetch_task = Task(
                description=f"Fetch recent research papers from arXiv for the topic '{topic}'.",
                expected_output="A list of paper titles, summaries, authors and URLs.",
                agent=fetcher
            )

            trend_task = Task(
                description=(
                    f"Analyze the list of research papers (title, summary, authors) on the topic '{topic}' "
                    "and identify the top 10 trending keywords or research themes in bullet points."
                ),
                expected_output="A list of 10 trending research topics or keywords with descriptions.",
                agent=analyzer,
                context=[fetch_task]
            )

            author_task = Task(
                description=(
                    f"Based on the list of paper titles, authors, and summaries for the topic '{topic}', "
                    "identify the most frequently mentioned authors. Include affiliations if available."
                ),
                expected_output=(
                    "- A list of top 10 authors, their affiliations, and publication counts.\n"
                    "- Format: Author Name – Institution (N papers)"
                ),
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

                # ========== 1. Papers Fetched ==========
                st.markdown("## 📚 Research Papers on Topic")

                # Get papers freshly to have URLs reliably
                papers = fetch_arxiv(topic)[:10]

                for i, paper in enumerate(papers):
                    title = paper["title"].replace("\n", " ").strip()
                    summary = paper["summary"].replace("\n", " ").strip()
                    url = paper["url"]
                    authors = ", ".join(paper["authors"])

                    with st.expander(f"📄 Paper {i+1}: {title}"):
                        st.markdown(f"**Title:** [{title}]({url})")
                        st.markdown(f"**Authors:** {authors}")
                        st.markdown(f"**Summary:** {summary}")

                # ========== 2. Trending Keywords Chart ==========
                st.markdown("## 📈 Trending Keywords in These Papers")

                # Convert output to string and parse keywords
                trend_text = str(trend_task.output)
                # Extract bullet points of form "- Keyword: description"
                trend_lines = [line.strip() for line in trend_text.split("\n") if line.strip()]
                keyword_data = []
                for line in trend_lines:
                    match = re.match(r"[-•]\s*(.+?):", line)
                    if match:
                        keyword = match.group(1)
                        keyword_data.append(keyword)
                    else:
                        # fallback: if line is just "- keyword" without colon
                        if line.startswith("-"):
                            keyword_data.append(line[1:].strip())

                keyword_data = keyword_data[:10]

                if keyword_data:
                    df_keywords = pd.DataFrame({'Keyword': keyword_data})
                    keyword_counts = df_keywords['Keyword'].value_counts().reset_index()
                    keyword_counts.columns = ['Keyword', 'Count']
                    fig = px.bar(keyword_counts, x='Keyword', y='Count', title="Top Trending Keywords", text='Count')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⚠️ No keywords found to visualize.")

                # ========== 3. Top Authors & Institutions ==========
                st.markdown("## 👩‍🔬 Top Authors and Institutions")

                author_text = str(author_task.output)
                author_lines = [line.strip() for line in author_text.split("\n") if line.strip()]
                # Only keep lines containing "–" (dash) indicating "Author – Institution"
                author_lines = [line for line in author_lines if "–" in line][:10]

                if author_lines:
                    for line in author_lines:
                        st.markdown(f"👤 **{line}**")
                else:
                    st.info("⚠️ No author information found to display.")

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
