import logging
from graph.workflow import create_workflow
from utils.process_json_files import get_json_files_list, load_file_content_to_vector_store, load_processed_files, save_processed_file
from config.constants import JSON_FILES_DIRECTORY, PROCESSED_FILES_PATH
import streamlit as st
import uuid

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


  app = create_workflow()


  # Set the page configuration
  st.set_page_config(page_title="Tec Wire AI", page_icon="🤖", layout="centered", initial_sidebar_state="expanded", )
  st.header('Tec Wire AI 🤖 📰')
  st.subheader('Your Daily Tec News Companion 😎')

  # Initialize chat history
  if "messages" not in st.session_state:
      st.session_state.messages = []
  if  "session_id" not in st.session_state:
     st.session_state.session_id = str(uuid.uuid4())

  # Sidebar with New Chat button and session ID
  with st.sidebar:
    session_id_container = st.empty()
    # Display the session ID
    session_id_container.write(f"**Session ID:** {st.session_state['session_id']}")

    # Add a button to start a new chat
    if st.button("New Chat"):
        # Reset chat history and generate a new session ID
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        session_id_container.write(f"**Session ID:** {st.session_state['session_id']}")
        st.success("New chat started!")
  

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])


  # Accept user input
  if prompt := st.chat_input("Message Tec Wire AI..."):
      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})
      # Display user message in chat message container
      try:
        with st.chat_message("user"):
          st.markdown(prompt)

        typing_indicator = st.empty()  
        typing_indicator.write("Tec wire is processiong your request...")

        config = {"configurable": {"thread_id": st.session_state.session_id}}
        response = app.invoke(
            {
                "input": prompt
            },
            config=config,
        )
        answer = response["answer"]

        typing_indicator.empty()  # Remove the typing indicator when done

        with st.chat_message("ai"):
            # display the AI's answer
            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "ai", "content": answer})

            # # Optional: Log chat history and summary
            # st.sidebar.write("Chat History:", response.get(
            #     "chat_history", "No history"))
            # st.sidebar.write("Summary:", response.get("summary", "No summary"))
      except Exception as e:
          st.error(f"An error occurred while processiong your request.")
        #   st.sidebar.write("Error details:", str(e))      

       
      


if __name__ == "__main__":
  main()
