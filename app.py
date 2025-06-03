import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import requests, feedparser, os

# Load environment variables
load_dotenv()

# Set Streamlit config
st.set_page_config(page_title="Research Trends Tracker", layout="wide")
st.title("📊 Research Trends Tracker")

# Input topic from user
topic = st.text_input("Enter research topic:", "Natural Language Processing")
api_key = st.text_input("Enter your OpenRouter API Key:", type="password")

# Function to fetch papers from arXiv
def fetch_arxiv(topic):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=10"
    response = requests.get(url)
    feed = feedparser.parse(response.text)
    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "authors": [author.name for author in entry.authors]
        })
    return results

# Wrapper to pass to Tool
def arxiv_fetch_wrapper(input):
    topic = input.get("topic") if isinstance(input, dict) else input
    return fetch_arxiv(topic)

# Function to create Crew and tasks
def create_crew(topic, api_key):
    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.7
    )

    arxiv_tool = Tool(
        name="ArxivResearchFetcher",
        func=arxiv_fetch_wrapper,
        description="Fetch recent arXiv papers by topic",
        return_direct=True
    )

    fetcher = Agent(
        role="Research Fetcher",
        goal="Find recent research papers",
        tools=[arxiv_tool],
        backstory="Expert in academic databases.",
        verbose=True,
        llm=llm
    )

    analyzer = Agent(
        role="Trend Analyzer",
        goal="Extract research trends from papers",
        verbose=True,
        backstory="Text mining and NLP specialist.",
        llm=llm
    )

    reporter = Agent(
        role="Author Reporter",
        goal="Find top authors and institutions",
        verbose=True,
        backstory="Academic contribution analyst.",
        llm=llm
    )

    fetch_task = Task(
        description=f"Fetch research papers for topic: {topic}",
        expected_output="A list of paper titles and summaries",
        agent=fetcher
    )

    trend_task = Task(
        description=f"Analyze research trends for topic: {topic} based on fetched papers.",
        expected_output="Top 5 trending keywords or themes.",
        context=[fetch_task],
        agent=analyzer
    )

    author_task = Task(
        description=f"List top authors and affiliations for topic: {topic}",
        expected_output="Top authors and their institutions.",
        context=[fetch_task],
        agent=reporter
    )

    crew = Crew(
        agents=[fetcher, analyzer, reporter],
        tasks=[fetch_task, trend_task, author_task],
        verbose=True
    )

    return crew, fetch_task, trend_task, author_task

# Execute if user submits input
if st.button("🔍 Track Research Trends"):
    if not api_key:
        st.warning("Please enter your OpenRouter API key.")
    else:
        with st.spinner("Running multi-agent research analysis..."):
            crew, fetch_task, trend_task, author_task = create_crew(topic, api_key)
            crew.kickoff()

            try:
                fetched_papers = fetch_task.output if hasattr(fetch_task, 'output') else fetch_task.result.output
                trends = trend_task.output if hasattr(trend_task, 'output') else trend_task.result.output
                authors = author_task.output if hasattr(author_task, 'output') else author_task.result.output

                st.subheader("📄 Fetched Papers")
                if isinstance(fetched_papers, list):
                    for i, paper in enumerate(fetched_papers):
                        st.markdown(f"**{i+1}. {paper['title']}**")
                        st.markdown(f"*Authors:* {', '.join(paper['authors'])}")
                        st.markdown(paper['summary'])
                        st.markdown("---")
                else:
                    st.markdown(fetched_papers)

                st.subheader("🔥 Research Trends")
                keywords_list = [line.strip("- ") for line in trends.split("\n") if line.strip().startswith("-")]
                st.markdown("\n".join([f"- {kw}" for kw in keywords_list]))

                st.subheader("🏫 Top Authors & Institutions")
                st.markdown(authors)

            except Exception as e:
                st.error(f"❌ Error: {e}")
