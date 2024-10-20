from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .edges import decide_source, decide_to_generate, grade_answer
from .graph_state import GraphState
from .nodes import formulate_query, generate, grade_documents, handle_irrelevant, retreive, store_final_answer, summarize_conversation, web_search


memory = MemorySaver()
# Define the Workflow Nodes
workflow = StateGraph(GraphState)

# Add nodes to the workflow
# Step 1: Formulate a standalone query from user input
workflow.add_node("formulate_query", formulate_query)
# Step 2a: Retrieve documents from vector store
workflow.add_node("retreive", retreive)
# Step 2b: Web search for additional documents
workflow.add_node("web_search", web_search)
# Step 3: Generate an answer
workflow.add_node("generate", generate)
# Step 4: Grade the retrieved documents
workflow.add_node("grade_documents", grade_documents)
# Handle irrelevant docs
workflow.add_node("irrelevant", handle_irrelevant)
# Store the final answer
workflow.add_node("store_final_answer", store_final_answer)
# Summarize the conversation
workflow.add_node("summarize_conversation", summarize_conversation)

# Define Workflow Transitions (Edges)
workflow.add_edge(START, "formulate_query")  # Start at formulating the query

# Conditional transition after formulating the query
workflow.add_conditional_edges(
    "formulate_query",
    decide_source,  # Decide next action based on the query
    {
        "vector_store": "retreive",  # Retrieve from vector store if relevant
        "irrelevant": "irrelevant"   # If query is irrelevant, handle it
    }
)

# Transition for handling irrelevant queries
workflow.add_edge("irrelevant", "summarize_conversation")
workflow.add_edge("summarize_conversation", END)

# Transition for relevant queries: retrieve and grade documents
workflow.add_edge("retreive", "grade_documents")

# Conditional transition after grading documents
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,  # Decide whether to generate answer or perform web search
    {
        "generate": "generate",  # If documents are good, generate answer
        "web_search": "web_search"  # Otherwise, perform a web search
    }
)

# Transition from web search to generating the answer
workflow.add_edge("web_search", "generate")

# Conditional transition after generating the answer
workflow.add_conditional_edges(
    "generate",
    grade_answer,  # Grade the generated answer
    {
        "good answer": "store_final_answer",  # If answer is good, store it
        "bad answer": "web_search"  # If answer is bad, go back to web search
    }
)

# Final transition to store the final answer and summarize the conversation
workflow.add_edge("store_final_answer", "summarize_conversation")

# Compile the Workflow with Memory Checkpointing
app = workflow.compile(checkpointer=memory)

def create_workflow():
    return app
