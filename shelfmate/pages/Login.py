import sqlite3
import streamlit as st

# Database setup
db_path = "shelfmate.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Helper Functions
def check_if_email_exists(email):
    cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
    return cursor.fetchone() is not None

def verify_user(email, password):
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    return cursor.fetchone()

def get_username_by_email(email):
    cursor.execute("SELECT username FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    return result[0] if result else None

# Login Form
st.title("Login")

with st.form("login_form"):
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

    if submit:
        if not email or not password:
            st.error("Please fill all fields")
            st.session_state.logged_in = False
        elif not check_if_email_exists(email):
            st.error("Email not registered")
            st.session_state.logged_in = False
        else:
            # Verify user with email and password
            user = verify_user(email, password)
            if user:
                # Fetch and store the username in session state
                username = get_username_by_email(email)
                st.session_state['username'] = username
                st.session_state['user_email'] = email
                st.session_state['logged_in'] = True
                
                st.success(f"Welcome, {username}!")
                st.switch_page("pages/Chatbot.py")
            else:
                st.error("Incorrect password")
                st.session_state.logged_in = False

st.write("Don't have an account yet? Register now!")
if st.button("Register"):
    st.switch_page("pages/Register.py")
