import logging
from graph.workflow import create_workflow
from utils.process_json_files import get_json_files_list, load_file_content_to_vector_store, load_processed_files, save_processed_file
from config.constants import JSON_FILES_DIRECTORY, PROCESSED_FILES_PATH
import streamlit as st

def main():
  json_files = get_json_files_list(JSON_FILES_DIRECTORY)
  if not json_files:
    logging.info(f"No JSON files found in {JSON_FILES_DIRECTORY}.")

  # get the list of processed files
  processed_files = load_processed_files(PROCESSED_FILES_PATH)

  for json_file in json_files:
    if json_file not in processed_files:
      load_file_content_to_vector_store(json_file)
      save_processed_file(PROCESSED_FILES_PATH, json_file)
  config = {"configurable": {"thread_id": "abc123"}}

  app = create_workflow()

  st.title("Tec Wire AI 🤖 💬")

  # Initialize chat history
  if "messages" not in st.session_state:
      st.session_state.messages = []

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])


  # Accept user input
  if prompt := st.chat_input("Message Tec Wire AI..."):
      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(prompt)

      with st.chat_message("ai"):
        response = app.invoke(
            {
                "input": prompt
            },
            config= config,
        )
        answer = response["answer"]
        st.write(
            answer
        )
        st.session_state.messages.append(
            {"role": "ai", "content": answer})
        
        print("chat history:", response["chat_history"])
        print("summary:", response["summary"])


if __name__ == "__main__":
  main()
