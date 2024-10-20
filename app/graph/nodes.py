from langchain_community.utilities import GoogleSerperAPIWrapper
from .chains import get_doc_grader_chain, get_formulated_query_chain, get_irrelavent_resonse_chain, get_rag_chain, get_synthesize_answer_chain
from utils.doc_func import create_docs_from_search_results
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from .graph_state import GraphState
from utils.retriver import get_retriever


load_dotenv()

# node
def formulate_query(state: GraphState):

  chain = get_formulated_query_chain()
  
  chat_history = state["chat_history"]
  input = state["input"]
  summary = state.get("summary", "")

  print("Chat History:", chat_history)
  print("summary:", summary)
  
  formatted_query = chain.invoke(
      {
          "chat_history": chat_history,
          "input": input,
          "summary": summary,
      }
  )
  print("Formatted Query:", formatted_query)
  return {"formatted_query": formatted_query,
          "chat_history": [HumanMessage(state["input"])],
          "input": state["input"],
          "vector_store_documents": [],
          "web_search_results": [],
          "summary": summary, }


web_search_tool = GoogleSerperAPIWrapper(k=2)

# node to do web search
def web_search(state: GraphState):
  print("----Web Search----")

  question = state["formatted_query"]
  # pass the question to the web search tool and get the results
  results = web_search_tool.results(question)
  # create documents from the search results
  docs = create_docs_from_search_results(results)

  return {"web_search_results": docs}


def generate(state: GraphState):
  # get vector store documents
  vector_store_documents = state.get("vector_store_documents", [])
  # get web search results
  web_search_results_documents = state.get("web_search_results", [])

  # combine the documents
  documents = vector_store_documents + web_search_results_documents

  # print("Documents:", documents)

  rag_chain = get_rag_chain()

  if len(documents) >= 8:
    print("---Do multiple LLM calls---")
    mid = len(documents) // 2
    response_1 = rag_chain.invoke(
        {
            "input": state["input"],
            "formatted_query": state["formatted_query"],
            "context": documents[:mid]
        }
    )
    response_2 = rag_chain.invoke(
        {
            "input": state["input"],
            "formatted_query": state["formatted_query"],
            "context": documents[mid:]
        }
    )

    combined_response = "\n".join([response_1, response_2])
    synthesize_answer_chain = get_synthesize_answer_chain()
    genaration = synthesize_answer_chain.invoke(
        {
            "input": state["input"],
            "formatted_query": state["formatted_query"],
            "responses": combined_response
        }
    )

  else:
    print("---Do single LLM call---")
    genaration = rag_chain.invoke(
        {
            "input": state["input"],
            "formatted_query": state["formatted_query"],
            "context": documents
        }
    )

  return {"answer": genaration}


# reterive node
def retreive(state: GraphState):
  print("----Retrive----")
  question = state["formatted_query"]

  retriever = get_retriever()
  documents = retriever.invoke(question)
  return {"vector_store_documents": documents}


# node to grade documents
def grade_documents(state: GraphState):
  print("----Grade Documents----")
  filterd_docs = []
  docs = state["vector_store_documents"]
  original_question = state["input"]
  fomulated_question = state["formatted_query"]
  for doc in docs:
    doc_txt = doc.page_content
    output = get_doc_grader_chain().invoke(
        {"input": original_question, "document": doc_txt, "formatted_query": fomulated_question})
    if output.binary_score == "yes":
      print("---Grade: Document is relevant---")
      filterd_docs.append(doc)
    else:
      print("---Grade: Document is not relevant---")
      continue
  return {"vector_store_documents": filterd_docs}


# node to handle irrelavent question
def handle_irrelevant(state: GraphState):
  print("---handeling irrelavent question---")
  respone = get_irrelavent_resonse_chain().invoke(
      {
          "input": state["input"],
          "chat_history": state["chat_history"],
          "chat_summary": state["summary"]
      }
  )

  return {"answer": respone, "chat_history": [AIMessage(respone)]}


# node to store final aswer and chat history
def store_final_answer(state: GraphState):
  print("---Store Final Answer---")
  return {"chat_history": [AIMessage(state["answer"])]}


def summarize_conversation(state: GraphState):
  print("---Summarize Conversation---")
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0,
                           #  google_api_key="AIzaSyDcY186rbEYMmex_jUSbY1zo6yOSiXg7Yk"
                           )
  summary = state.get("summary", "")
  if summary:
    summary_message = """
    This is summary of the conversation to date: {summary}
    Extend the summary by taking into account the new messages above.
    """

  else:
    summary_message = "Create a summary of the conversation above."
  messages = state["chat_history"] + [HumanMessage(summary_message)]
  chain = llm | StrOutputParser()
  summary = chain.invoke(messages)

  delete_messages = [RemoveMessage(id=m.id)
                     for m in state["chat_history"][:-4]]
  
  return {"summary": summary, "chat_history": delete_messages}
