
import os
from typing import TypedDict

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY first.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3
)

class StudyInput(TypedDict):
    subjects: str
    exam_date: str
    daily_hours: str
    preparation_level: str
    question: str

class StudyState(StudyInput):
    study_analysis: str
    study_plan: str
    final_answer: str

def study_analysis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are a friendly AI study mentor.
Subjects: {subjects}
Exam date: {exam_date}
Daily study hours: {daily_hours}
Preparation level: {preparation_level}
Student question: {question}

Analyze the student's preparation and prioritize subjects.
Use simple English. Calculate the time remaining until the exam
based on today's date, and never assume the exam is years away.
""")
    response = (prompt | llm).invoke(state)
    return {"study_analysis": response.content}

def create_study_plan(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are an expert study planner.
Subjects: {subjects}
Exam date: {exam_date}
Daily study hours: {daily_hours}
Preparation level: {preparation_level}
Today's date: 2026-10-24

Study analysis:
{study_analysis}

Create a practical timetable that fits the time remaining until
the exam. Include topics, revision, practice questions and breaks.
Use simple English. Do not claim there are years remaining if the
exam is in November 2026.
""")
    response = (prompt | llm).invoke(state)
    return {"study_plan": response.content}

def final_synthesis(state: StudyState):
    prompt = ChatPromptTemplate.from_template("""
You are StudyMentor, a helpful AI study assistant.
Student question: {question}
Study analysis:
{study_analysis}
Study plan:
{study_plan}

Combine these into a clear answer with these headings:
1. Study Analysis
2. Personalized Study Plan
3. Final Recommendations

Use simple English.
""")
    response = (prompt | llm).invoke(state)
    return {"final_answer": response.content}

workflow = StateGraph(StudyState)
workflow.add_node("study_analysis", study_analysis)
workflow.add_node("create_study_plan", create_study_plan)
workflow.add_node("final_synthesis", final_synthesis)

workflow.add_edge(START, "study_analysis")
workflow.add_edge("study_analysis", "create_study_plan")
workflow.add_edge("create_study_plan", "final_synthesis")
workflow.add_edge("final_synthesis", END)

graph = workflow.compile()

def run_studymentor(inputs: StudyInput):
    initial_state = {
        **inputs,
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

agent = RunnableLambda(run_studymentor)

app = FastAPI(
    title="StudyMentor AI Agent",
    description="An AI agent that analyzes study needs and creates personalized study plans.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "StudyMentor AI Agent is running!"}

add_routes(app, agent, path="/agent")
