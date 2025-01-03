import streamlit as st
import time
from dotenv import load_dotenv
import sys

from shelfmate.chatbot.bot import MainChatbot  # Import the chatbot class

def check_auth():
    return 'logged_in' in st.session_state and st.session_state.logged_in

# Function to simulate streaming response
def simulate_streaming(message):
    buffer = ""
    for char in message:
        buffer += char
        if char in [" ", "\n"]:
            yield buffer.strip() + ("<br>" if char == "\n" else " ")
            buffer = ""
            time.sleep(0.1 if char == "\n" else 0.05)
    if buffer:
        yield buffer

# Load environment variables
load_dotenv()

# Authentication check
if not check_auth():
    st.warning("You need to login to access the chatbot.")
    if st.button("Go to Login Page"):
        st.switch_page("pages/Login.py")
else:
    st.title("ShelfMate Chatbot")

    username = st.session_state['username']
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = 0 
    conversation_id = st.session_state["conversation_id"]

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial template message for first interaction
        initial_message = "Hello! I'm ShelfMate, your personal shopping assistant. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": initial_message})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "images/corujafofa.jpg"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Initialize the chatbot instance
        bot = MainChatbot()
        bot.user_login(username=username, conversation_id=conversation_id)

        try:
            # Process user input using the bot
            response = bot.process_user_input({"user_input": user_input})
            with st.chat_message("assistant", avatar="images/corujafofa.jpg"):
                st.markdown(response, unsafe_allow_html=True)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {str(e)}")


