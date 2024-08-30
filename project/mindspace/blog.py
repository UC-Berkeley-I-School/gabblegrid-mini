import streamlit as st
import sqlite3
from datetime import datetime
from streamlit_quill import st_quill

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

def save_post_to_db(title, content, author):
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('INSERT INTO blog (title, content, author, timestamp) VALUES (?, ?, ?, ?)', 
              (title, content, author, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_all_posts():
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('SELECT id, title, content, author, timestamp FROM blog ORDER BY timestamp DESC')
    posts = c.fetchall()
    conn.close()
    return posts

def get_post_by_id(post_id):
    conn = sqlite3.connect('mindspace_blog.db')
    c = conn.cursor()
    c.execute('SELECT title, content, author, timestamp FROM blog WHERE id = ?', (post_id,))
    post = c.fetchone()
    conn.close()
    return post

def display_blog():
    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = "mindspace"
    
    if 'selected_post_id' not in st.session_state:
        posts = get_all_posts()
        if posts:
            st.session_state.selected_post_id = posts[0][0]  # Automatically select the latest blog post

    st.markdown("<h4>Blog</h4>", unsafe_allow_html=True)
    
    init_db()

    col1, col2 = st.columns([1, 3])

    with col1:
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

        st.markdown("---")
        email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
        if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
            if st.button("Write a New Blog"):
                st.session_state.page = "write_blog"

    with col2:
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

def display_write_blog():
    st.title("Write a New Blog Post")

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