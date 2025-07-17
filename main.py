from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser


app = FastAPI()

class ChatInput(BaseModel):
    question : str

#  histroy stroe 
chat_history = []

# langchain components 

llm = ChatOllama(model="llama3.1")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are an intelligent chatbot. Answer the following question."),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="question")
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

# api
@app.post("/chat/")
def chat(input:ChatInput):
    global chat_history

    try: 
        question = input.question

        response = chain.invoke({"history":chat_history,"question":[HumanMessage(question)]})

        chat_history.extend([HumanMessage(content=question),AIMessage(content=response)])

        return {"response":response}
    
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))

