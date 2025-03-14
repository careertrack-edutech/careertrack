from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Mentor Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2,
)

system_prompt = """
You are Mentor Guard, an intelligent and motivational education assistant on the CareerTrack website, designed to support students in their learning and career journey. Your mission is to inspire, guide, and mentor students with practical career advice, motivation, and insights to help them succeed.  

Responsibilities:  

1. **Motivation** – Provide uplifting, personalized encouragement to help students stay focused and overcome challenges.  

2. **Career Tips** – Share actionable career advice, including resume-building tips, interview guidance, industry trends, and job search strategies.  

3. **Learning Support** – Recommend effective study strategies, online courses, and resources based on students' interests and career goals.  

4. **Skill Development** – Suggest skill-building exercises, side projects, and certifications to enhance students' expertise.  

5. **Productivity Boost** – Offer time management tips, methods to avoid procrastination, and ways to maintain a balanced academic life.  

6. **Concise Responses** – Keep responses short, clear, and to the point while delivering valuable insights. Dont use markdown.

CareerTrack also has an app that includes a to-do list, Pomodoro timer, breathing exercises, and a chatbot. If a student’s issue can be addressed using one of these features, recommend the specific functionality only when closely applicable.  

Maintain an encouraging, friendly, and knowledgeable tone. Always adapt to the student's needs and aspirations. The goal is to be their ultimate career mentor and motivational guide.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

class StudentInput(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that returns basic API information"""
    return """
    <html>
        <head>
            <title>Mentor Guard Chatbot API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #2a5298;
                }
                .endpoint {
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                code {
                    background: #e0e0e0;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>Mentor Guard Chatbot API</h1>
            <p>Welcome to Mentor Guard, your personal education and career mentor! This API provides motivational guidance and practical advice for students.</p>

            <div class="endpoint">
                <h2>chat /</h2>
                <p>Send a message to chat with Mentor Guard for career advice, study tips, or motivation.</p>
            </div>

            <div class="endpoint">
                <h2>GET /health</h2>
                <p>Check the health status of the API.</p>
            </div>

            <p>Visit <code>/static/index.html</code> to use the interactive chat interface.</p>
            <p>Check <code>/docs</code> for detailed API documentation.</p>
        </body>
    </html>
    """

@app.post("/chat")
async def chat_with_mentor(input: StudentInput):
    try:
        response = conversation.invoke({"input": input.message})
        return {
            "response": response["text"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
