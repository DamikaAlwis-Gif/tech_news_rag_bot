from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from app.graph.edges import decide_source, decide_to_generate, grade_answer
from app.graph.graph_state import GraphState
from app.graph.nodes import formulate_query, generate, grade_documents, handle_irrelevant, retreive, store_final_answer, summarize_conversation, web_search

def create_workflow():
  memory = MemorySaver()
  workflow = StateGraph(GraphState)
  # formulate a query from the user input
  workflow.add_node("formulate_query", formulate_query)
  # retrive documents from the vector store
  workflow.add_node("retreive", retreive)
  workflow.add_node("web_search", web_search)  # web search
  workflow.add_node("generate", generate)  # generate answer
  workflow.add_node("grade_documents", grade_documents)  # grade documents
  # handle irrelavent question
  workflow.add_node("irrelevant", handle_irrelevant)
  # store final answer
  workflow.add_node("store_final_answer", store_final_answer)
  # summarize conversation
  workflow.add_node("summarize_conversation", summarize_conversation)

  workflow.add_edge(START, "formulate_query")
  workflow.add_conditional_edges("formulate_query", decide_source,
                                {
                                    "vector_store": "retreive",
                                    "irrelevant": "irrelevant"
                                }
                                )
  workflow.add_edge("irrelevant", "summarize_conversation")
  workflow.add_edge("summarize_conversation", END)
  workflow.add_edge("retreive", "grade_documents")
  workflow.add_conditional_edges("grade_documents", decide_to_generate,
                                {
                                    "generate": "generate",
                                    "web_search": "web_search"
                                }
                                )
  workflow.add_edge("web_search", "generate")
  workflow.add_conditional_edges("generate", grade_answer,
                                {
                                    "good answer": "store_final_answer",
                                    "bad answer": "web_search"
                                }
                                )
  workflow.add_edge("store_final_answer", "summarize_conversation")


  app = workflow.compile(checkpointer=memory)

  return app
