
import os
from typing import TypedDict

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langserve import add_routes


# Check Google API key
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY first."
    )


# Initialize Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3
)


# Define the information used by the agent
class StudyState(TypedDict):
    subjects: str
    exam_date: str
    daily_hours: str
    preparation_level: str
    question: str
    study_analysis: str
    study_plan: str
    final_answer: str


# Agent 1: Analyze student requirements
def study_analysis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly AI study mentor.

    Subjects: {subjects}
    Exam date: {exam_date}
    Daily study hours: {daily_hours}
    Preparation level: {preparation_level}
    Student question: {question}

    Analyze the student's needs and identify study priorities.
    Use simple English.
    """)

    response = (prompt | llm).invoke(state)

    return {"study_analysis": response.content}


# Agent 2: Create a personalized study plan
def create_study_plan(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert study planner.

    Student subjects: {subjects}
    Exam date: {exam_date}
    Daily study hours: {daily_hours}
    Preparation level: {preparation_level}

    Student analysis:
    {study_analysis}

    Create a practical study timetable.
    Include daily topics, revision, practice questions,
    and short breaks. Keep it beginner-friendly.
    """)

    response = (prompt | llm).invoke(state)

    return {"study_plan": response.content}


# Agent 3: Prepare the final response
def final_synthesis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
    You are StudyMentor, a helpful AI study assistant.

    Student question:
    {question}

    Study analysis:
    {study_analysis}

    Study plan:
    {study_plan}

    Combine these into a clear, organized answer.
    Use headings and simple English.
    """)

    response = (prompt | llm).invoke(state)

    return {"final_answer": response.content}


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


# Create the FastAPI application
app = FastAPI(
    title="StudyMentor AI Agent",
    description="An AI agent that analyzes study needs and creates personalized study plans.",
    version="1.0.0"
)


# Home page
@app.get("/")
def home():
    return {
        "message": "StudyMentor AI Agent is running!"
    }


# Expose the agent through an API endpoint
add_routes(app, graph, path="/agent")
