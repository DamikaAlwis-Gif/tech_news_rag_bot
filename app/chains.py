from typing import Literal
from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

def get_rag_chain():

  template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Please attach the URL of the relevant news articles or sources to your answer.
    Provide answer in ponint format.
    
    original user query: {input}
    formulated query: {formatted_query}
    context: {context}
    Provide the response in this format:
    
    <The answer goes here>

    Source:
    <The title > 
    <The URL or citation goes here>
  
  """
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

  rag_prompt = PromptTemplate(
      template=template,
      input_variables=["input", "formatted_query", "context"]
  )
  rag_chain = rag_prompt | llm | StrOutputParser()
  return rag_chain


def get_synthesize_answer_chain():

  system = '''
  You are an intelligent assistant that synthesizes multiple responses into a single, coherent answer.
  You are the given the user query, the formatted query, and responses for the user query.
  Use the following rules:
    - Combine the key points from all responses.
    - Remove any redundant or repetitive information.
    - Ensure the final answer is concise and logically structured.
    - Provide a natural, flowing response.
    - Avoid explicitly mentioning individual sources unless needed, and do not refer to each response separately.
  orginal user query: {input}
  formatted query: {formatted_query}
  responses: {responses}
  Provide the response in this format:
    
  <The answer goes here>

  Sources:
  <The title > 
  <The URL or citation goes here>
    
  '''

  # Create the prompt
  prompt = PromptTemplate(
      template=system,
      input_variables=["input", "formatted_query", "responses"]
  )
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

  # Create the chain
  chain = prompt | llm | StrOutputParser()
  return chain

def get_formulated_query_chain():
  contextualize_q_system_prompt = """
  Given a chat history, chat summary, and the latest user question (which may refer to previous context), 
  your task is to reformulate the user's question into a standalone version that can be understood 
  without any reference to the chat history. 

  - If the question already makes sense on its own, return it as is.
  - Do not answer the question.
  - Only return the reformulated question, if necessary.

  Output only the reformulated question or the original if no reformulation is needed.
  user query : {input}
  chat_history : {chat_history}
  chat_summary : {summary}

  """

  prompt_genrate_q = PromptTemplate(
      template=contextualize_q_system_prompt,
      input_variables=["input", "chat_history", "summary"]
  )
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0,)

  chain = prompt_genrate_q | llm | StrOutputParser()
  return chain


def get_agent_router_chain():
  
  class RouteQuery(BaseModel):
    """Model for routing user queries."""
    # ... means this field is required
    binary_score: Literal["vector_store", "irrelevant"] = Field(
        ...,
        description=(
            "Given an original user query and chat history, choose to route it to a vectorstore or classify it as irrelevant."
        )
    )
    reason: str = Field(
        ...,
        description="Explain why you chose the answer you did. Provide your reasoning here."
    )


  # Initialize the language model with structured output
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  structured_llm_router = llm.with_structured_output(RouteQuery)

  # Define the system prompt
  system_prompt_template = """
      You are an expert at routing a user question to a vectorstore or classifying it as irrelevant.
      The vectorstore contains documents related to technology, business, and innovations with 
      an emphasis on tech news, companies, and emerging trends. Use the vectorstore for questions on these topics. 
      
      However, if the query is off-topic, irrelevant, or nonsensical (not related to technology, business, or innovation), 
      choose 'irrelevant'. For example, if the question is personal, unrelated to tech, or meaningless, 'irrelevant' would be appropriate.

      Pay careful attention to:
      1. The original user query for direct context.
      2. The chat history and summary of chat history for disambiguation or patterns in user interests.

      original user query: {input}
      chat history: {chat_history}
      summary: {summary}
  """

  route_prompt = PromptTemplate(
      template=system_prompt_template,
      input_variables=["input", "chat_history", "summary"],
  )

  # Combine the prompt with the structured language model router
  agent_router_chain = route_prompt | structured_llm_router

  return agent_router_chain


def get_grade_answer_chain():

  class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)


  structured_llm_grader = llm.with_structured_output(GradeAnswer)
  system = """
    You are a grader assessing whether an answer addresses / resolves a question or not.
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
    'No' means that the answer does not resolve the question. 
    original user query: {input}
    chat history: {chat_history}
    chat summary: {summary}
    answer: {answer}
    """
  grade_prompt = PromptTemplate(
      template=system,
      input_variables=["input", "answer", "chat_history", "summary"]
  )
  answer_grader = grade_prompt | structured_llm_grader

  return answer_grader


def get_doc_grader_chain():

  class GradeDocuments(BaseModel):
    """ check the relavence of the documents to the question"""
    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)


  stuctured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
  system = """
  You are a grader assessing relevance of a retrieved document to a user question.
  If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
  It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
  Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

  orginal user query: {input}
  formulated question: {formatted_query}
  document: {document}

  """
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          ("human",
              "Retrived document: \n\n {document} \n\n User question: {input} \n\n Formulated question: {formatted_query}"),
      ]
  )

  doc_grader = prompt | stuctured_llm_doc_grader
  return doc_grader


def get_irrelavent_resonse_chain():

  # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                          temperature=0,
                          #  google_api_key="AIzaSyDcY186rbEYMmex_jUSbY1zo6yOSiXg7Yk"
                          )
  system = """
  You are an AI chatbot that specializes in answering questions related to technology, business, and innovations.

  The user has asked a question that seems irrelevant to these topics.

  If the user has asked relevant questions before, guide them back to those topics.
  If the user asking irrelevant questions continusly, gently remind them of the bot's focus.

  Chat history: {chat_history}
  The user's question: {input}

  Respond in a helpful and professional manner.
  Suggest some relevant topics,for example AI trends, blockchain innovations, or news about tech companies.


  """

  prompt = PromptTemplate(
      template=system,
      input_variables=["input", "chat_history"]
  )

  irrelavent_resonse_chain = prompt | llm | StrOutputParser()
  return irrelavent_resonse_chain




  