from dotenv import load_dotenv
from .chains import get_agent_router_chain, get_grade_answer_chain, get_irrelavent_resonse_chain
from .graph_state import GraphState


# Load the environment variables
load_dotenv()

def decide_source(state: GraphState) -> str:
    """Decide whether to use vector store or classify as irrelevant."""
    response = get_agent_router_chain().invoke(
        {
            "input": state["formatted_query"],
            # "chat_history": state["chat_history"],
            # "summary": state["summary"],
        }
    )

    grade = response.binary_score
    print("Reasoning:", response.reason)
    print("Grade:", grade)

    return "vector_store" if grade == "vector_store" else "irrelevant"


# grade answer edge
def grade_answer(state: GraphState):
  print("----Grade Answer----")
  response = get_grade_answer_chain().invoke(
      {
          "input": state["input"],
          "answer": state["answer"],
          "chat_history": state["chat_history"],
          "summary": state["summary"]

      }
  )
  grade = response.binary_score
  # the web search has done before
  has_done_web_search = bool(state.get("web_search_results"))
  if has_done_web_search:
    print("---Web Search has done before---")

  # if the answer is good one or the web search has done before return good answer
  if grade == "yes" or has_done_web_search:
    print("---Good Answer---")
    return "good answer"
  else:
    print("---Bad Answer---")
    return "bad answer"
  
  # edge to decide to generate or do web search


def decide_to_generate(state: GraphState):
  print("----Decide to Generate or web search----")

  filtered_docs = state.get("vector_store_documents")
  # if filterd docs are less than 3 then do web search
  if len(filtered_docs) < 3:
    print("---Decide to Web Search---")
    return "web_search"
  else:
    print("---Decide to Generate---")
    return "generate"



