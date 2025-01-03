import streamlit as st
import sqlite3

# Display animated title
st.markdown(
    """
    <style>
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    .animated-title {
        font-size: 45px;
        font-weight: 800;
        text-align: center;
        animation: bounce 2s infinite;
    }
    </style>
    <div class="animated-title">Read List</div>
    """,
    unsafe_allow_html=True
)

st.write("In this page you can have access to the books in your current read list!")

# Check if user is logged in
if 'username' not in st.session_state:
    st.warning("You need to log in to view your reading list.")
    if st.button("Go to Login"):
        st.switch_page("pages/Login.py")
    st.stop()

username = st.session_state['username']
st.write(f"Hello, {username}!")

# Connect to the database
conn = sqlite3.connect('shelfmate.db')
c = conn.cursor()

# Create two columns: one for filters, one for book list
col1, col2 = st.columns([1, 3])  # First column smaller (1), second one bigger (3)

# Column 1: Filters
with col1:
    st.subheader("Filters")

    # Add filter options for rating and status
    rating_filter = st.selectbox(
        "Filter by Rating",
        ["All Ratings", "1", "2", "3", "4", "5", "No Rating"]
    )

    status_filter = st.selectbox(
        "Filter by Status",
        ["All Statuses", "In Progress/Finished", "Not Finished"]
    )

# Column 2: Books Display
with col2:
    # Fetch books from the read list with filters
    query = '''
        SELECT books.title, read_list.rating, read_list.did_not_finish_flag
        FROM read_list
        JOIN books ON read_list.book_id = books.book_id
        WHERE read_list.username = ?
    '''

    # Add filter conditions to the query
    filters = [username]

    # Adjust query for rating filter, including "No Rating"
    if rating_filter != "All Ratings" and rating_filter != "No Rating":
        query += " AND read_list.rating = ?"
        filters.append(int(rating_filter))
    elif rating_filter == "No Rating":
        query += " AND read_list.rating IS NULL"
        
    # Add filter for status filter
    if status_filter != "All Statuses":
        if status_filter == "Not Finished":
            query += " AND read_list.did_not_finish_flag = 1"
        else:
            query += " AND read_list.did_not_finish_flag = 0"

    # Execute the query with filters
    c.execute(query, tuple(filters))
    read_list = c.fetchall()

    # Display the filtered books
    if read_list:
        for book in read_list:
            title, rating, did_not_finish = book

            # Handle missing rating
            if rating is None:
                rating_display = 'No Rating'
                rating_stars = ''  # No stars if there's no rating
            else:
                rating_display = f"{rating}/5"
                rating_stars = '⭐' * rating  # Show stars based on rating
            
            # Set status
            status = "Not Finished" if did_not_finish else "In Progress/Finished"
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 10px;">
                <h4>{title}</h4>
                <p><b>Rating:</b> {rating_stars} {rating_display}</p>
                <p><b>Status:</b> {status}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No books found in your reading list matching the selected filters.")
