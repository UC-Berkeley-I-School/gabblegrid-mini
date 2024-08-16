import streamlit as st
from streamlit_quill import st_quill  # Importing the rich-text editor
import sqlite3
from datetime import datetime

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS blog (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            content TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to save post content to a database
def save_post_to_db(title, content, author):
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('INSERT INTO blog (title, content, author, timestamp) VALUES (?, ?, ?, ?)', 
              (title, content, author, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Function to retrieve all posts from the database
def get_all_posts():
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('SELECT id, title, content, author, timestamp FROM blog ORDER BY timestamp DESC')
    posts = c.fetchall()
    conn.close()
    return posts

# Function to retrieve a single post by ID
def get_post_by_id(post_id):
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('SELECT title, content, author, timestamp FROM blog WHERE id = ?', (post_id,))
    post = c.fetchone()
    conn.close()
    return post

# Display the main MindSpace content
def display_mindspace():
    # Initialize the session state variables
    if 'page' not in st.session_state:
        st.session_state.page = "mindspace"

    if 'selected_post_id' not in st.session_state:
        posts = get_all_posts()
        if posts:
            st.session_state.selected_post_id = posts[0][0]  # Automatically select the latest blog post

    # Apply custom styles
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f9;
                color: grey;
            }
            h1, h2, h3, h4, h5, h6 {
                color: grey !important;
            }
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey;
            }
            .grey-text {
                color: grey;
            }
        </style>
    """, unsafe_allow_html=True)

    # Introduction section with collapsible "Read more"
    st.markdown("""
        <div class='section'>
            <p>Welcome to MindSpace! A personal space for your thoughts, readings, and more.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("Read more"):
        st.markdown("""
            <p class='grey-text'>
            Placeholder for additional information about MindSpace. You can add details or explanations here to help users understand what this section is about.
            </p>
        """, unsafe_allow_html=True)

    # Initialize the database
    init_db()

    # Create a two-column layout
    col1, col2 = st.columns([1, 3])

    with col1:
        # st.header("Published Blogs")
        st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
        posts = get_all_posts()
        if posts:
            for post in posts:
                id, title, content, author, timestamp = post
                if st.button(f"{title}", key=f"btn_{id}"):
                    st.session_state.selected_post_id = id
                    st.session_state.page = "view_post"
        else:
            st.write("No blog posts published yet.")

        # Button to navigate to the blog writing section (only for the admin)
        st.markdown("---")
        email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
        if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
            if st.button("Write a New Blog"):
                st.session_state.page = "write_blog"

    with col2:
        # Display the selected blog post
        if st.session_state.page == "view_post" and st.session_state.selected_post_id:
            post = get_post_by_id(st.session_state.selected_post_id)
            if post:
                title, content, author, timestamp = post
                st.subheader(title)
                st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
                st.write(content)
                st.markdown("---")
        elif st.session_state.page == "write_blog":
            display_write_blog()
        else:
            post = get_post_by_id(st.session_state.selected_post_id)
            if post:
                title, content, author, timestamp = post
                st.subheader(title)
                st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
                st.write(content)
                st.markdown("---")

# Display the blog writing page
def display_write_blog():
    st.title("Write a New Blog Post")

    # Check if user is authenticated and is the admin
    email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
    if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
        st.warning("You do not have permission to write a blog post.")
        return

    with st.form(key='blog_form'):
        title = st.text_input("Title")
        content = st_quill(placeholder="Write your blog post here...")
        submit_button = st.form_submit_button(label="Publish")

    if submit_button:
        if title and content and content.strip():  # Check if content is not empty
            save_post_to_db(title, content, st.session_state['user']['name'])
            st.success("Post published!")
            st.session_state.page = "mindspace"
            st.experimental_rerun()  # Rerun to show the newly added blog
        else:
            st.error("Title and content cannot be empty.")

# Main application logic
def main():
    display_mindspace()

if __name__ == '__main__':
    main()
