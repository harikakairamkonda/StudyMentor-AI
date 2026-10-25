
import os
from datetime import date, datetime
from typing import TypedDict

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langserve import add_routes


# Check API key
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY."
    )


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Input fields
class StudyInput(TypedDict):
    subjects: str
    exam_date: str
    daily_hours: str
    preparation_level: str
    question: str


# Agent state
class StudyState(StudyInput):
    days_remaining: int
    study_analysis: str
    study_plan: str
    final_answer: str


# Agent 1: Study Analysis
def study_analysis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are a friendly AI study mentor.

Subjects: {subjects}
Exam date: {exam_date}
Days remaining: {days_remaining}
Daily study hours: {daily_hours}
Preparation level: {preparation_level}
Student request: {question}

Analyze the student's preparation and prioritize subjects.

Rules:
- Use simple English.
- Give only 3 to 5 short bullet points.
- Identify important subjects and weak areas.
- Do not create a timetable here.
- Do not repeat the input details.
- Do not use Markdown headings, bold symbols, or code blocks.
""")

    response = (prompt | llm).invoke(state)
    return {"study_analysis": response.content}


# Agent 2: Study Plan
def create_study_plan(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are an expert study planner.

Subjects: {subjects}
Exam date: {exam_date}
Days remaining: {days_remaining}
Daily study hours: {daily_hours}
Preparation level: {preparation_level}
Student request: {question}

Study analysis:
{study_analysis}

Create a practical timetable that fits the time available.

Rules:
- Follow the number of days requested by the student.
- If the student explicitly requests a 7-day plan, create 7 days.
- Otherwise, plan according to the days remaining until the exam.
- Include daily topics, practice questions, revision, and breaks.
- Keep each day's plan concise.
- Use simple English.
- Do not invent a different current date.
- Do not use Markdown symbols, JSON, or code blocks.
- Do not repeat the study analysis.
""")

    response = (prompt | llm).invoke(state)
    return {"study_plan": response.content}


# Agent 3: Final Answer
def final_synthesis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are StudyMentor, a helpful AI study assistant.

Student request:
{question}

Study analysis:
{study_analysis}

Personalized study plan:
{study_plan}

Create the final answer for the student.

Rules:
- Include a short Study Analysis section.
- Include the Personalized Study Plan.
- End with 2 or 3 practical recommendations.
- Preserve the timetable duration from the study plan.
- Follow the student's explicit request.
- Do not invent dates or days remaining.
- Do not repeat the same information.
- Use simple English and concise text.
- Do not include Python code, JSON, metadata, or code blocks.
- Avoid Markdown symbols such as #, **, and backticks.
- Return only the final study advice.
""" )

    response = (prompt | llm).invoke(state)
    content = response.content

    # Handle text responses returned as content blocks
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)

    return {"final_answer": str(content)}


# Build the LangGraph workflow
workflow = StateGraph(StudyState)

workflow.add_node("study_analysis", study_analysis)
workflow.add_node("create_study_plan", create_study_plan)
workflow.add_node("final_synthesis", final_synthesis)

workflow.add_edge(START, "study_analysis")
workflow.add_edge("study_analysis", "create_study_plan")
workflow.add_edge("create_study_plan", "final_synthesis")
workflow.add_edge("final_synthesis", END)

graph = workflow.compile()


# Run all three agents
def run_studymentor(inputs: StudyInput):
    try:
        exam_date = datetime.strptime(
            inputs["exam_date"], "%Y-%m-%d"
        ).date()

        today = date.today()
        days_remaining = (exam_date - today).days

        initial_state = {
            **inputs,
            "days_remaining": days_remaining,
            "study_analysis": "",
            "study_plan": "",
            "final_answer": ""
        }

        result = graph.invoke(initial_state)

        return {
            "study_analysis": result["study_analysis"],
            "study_plan": result["study_plan"],
            "final_answer": result["final_answer"]
        }

    except ValueError as error:
        raise ValueError(
            "Invalid exam date. Use YYYY-MM-DD format."
        ) from error


# LangServe runnable
agent = RunnableLambda(run_studymentor)


# FastAPI application
app = FastAPI(
    title="StudyMentor AI Agent",
    description=(
        "An AI study mentor that analyzes preparation "
        "and creates personalized study plans."
    ),
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "StudyMentor AI Agent is running!"}


# Keep the existing Playground route
add_routes(app, agent, path="/agent")
