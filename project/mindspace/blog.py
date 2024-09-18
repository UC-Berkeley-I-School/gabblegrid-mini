# # ############################### DEFAULT GOLD - Blue rectangle files with edit ###########################

import streamlit as st
from datetime import datetime
import os
import yaml
from streamlit_quill import st_quill

BLOG_DIR = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/02.Blog'

def save_blog_post(title, content, author, update=False, filename=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not update:
        filename = f"{timestamp}_{title.replace(' ', '_')}.md"
    filepath = os.path.join(BLOG_DIR, filename)
    
    metadata = {
        'title': title,
        'author': author,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(filepath, 'w') as f:
        f.write("---\n")
        yaml.dump(metadata, f)
        f.write("---\n\n")
        f.write(content)

def get_all_posts():
    posts = []
    for filename in os.listdir(BLOG_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(BLOG_DIR, filename)
            with open(filepath, 'r') as f:
                content = f.read()
                metadata, post_content = content.split('---\n', 2)[1:]
                metadata = yaml.safe_load(metadata)
                posts.append((filename, metadata['title'], metadata['author'], metadata['date']))
    return sorted(posts, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), reverse=True)

def get_post_by_filename(filename):
    filepath = os.path.join(BLOG_DIR, filename)
    with open(filepath, 'r') as f:
        content = f.read()
        metadata, post_content = content.split('---\n', 2)[1:]
        metadata = yaml.safe_load(metadata)
    return metadata['title'], post_content.strip(), metadata['author'], metadata['date']

def display_blog():
    if 'page' not in st.session_state:
        st.session_state.page = "mindspace"
    
    if 'selected_post' not in st.session_state:
        st.session_state.selected_post = None

    posts = get_all_posts()
    if posts and st.session_state.selected_post is None:
        st.session_state.selected_post = posts[0][0]

    col1, col2 = st.columns([1, 3], gap="large")  # Adding gap between columns

    st.markdown("""
        <style>
        [data-testid="stHorizontalBlock"] [data-testid="column"] .stButton > button {
            height: 75px !important;
            width: 200px !important;
            background-color: teal !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 10px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            color: grey !important;
            transition: all 0.3s ease !important;
        }
        .blog-title {
            font-size: 18px !important;  /* Bigger font size for blog title */
            font-weight: bold !important;
            color: grey !important;      /* White text for contrast */
        }
        .blog-date {
            font-size: 12px !important;  /* Smaller font size for the date */
            color: #555 !important;      /* Dimmed color for date */
            margin-bottom: 5px !important;  /* Space between date and button */
        }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        # st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
        posts = get_all_posts()
        if posts:
            email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None

            for i, (filename, title, author, timestamp) in enumerate(posts):
                formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%d, %b, %Y")
                
                is_selected = filename == st.session_state.selected_post
                
                # Display the published date above the button
                st.markdown(f'<div class="blog-date">Published on {formatted_timestamp}</div>', unsafe_allow_html=True)

                # Create a button for each blog title, centered inside the button
                if st.button(f"{title}", key=f"blog_{filename}_{i}"):
                    if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
                        st.session_state.page = "edit_blog"
                        st.session_state.selected_post = filename
                    else:
                        st.session_state.page = "view_post"
                        st.session_state.selected_post = filename

                # Styling for selected post
                st.markdown(f"""
                    <style>
                    [key="blog_{filename}_{i}"] > button {{
                        background-color: {'#FFC107' if is_selected else 'teal'} !important;
                    }}
                    </style>
                """, unsafe_allow_html=True)

        else:
            st.write("No blog posts published yet.")

        st.markdown("---")

        # Button for "Write a New Blog"
        email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
        if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
            if st.button("Write a New Blog", key="write_new_blog"):
                st.session_state.page = "write_blog"

    with col2:
        if st.session_state.page == "view_post" and st.session_state.selected_post:
            post = get_post_by_filename(st.session_state.selected_post)
            if post:
                title, content, author, timestamp = post
                st.markdown(f"<h2 style='color: teal;'>{title}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
                st.markdown(content)
                st.markdown("---")
        elif st.session_state.page == "edit_blog" and st.session_state.selected_post:
            display_edit_blog()
        elif st.session_state.page == "write_blog":
            display_write_blog()
        else:
            if st.session_state.selected_post:
                post = get_post_by_filename(st.session_state.selected_post)
                if post:
                    title, content, author, timestamp = post
                    st.markdown(f"<h2 style='color: teal;'>{title}</h2>", unsafe_allow_html=True)
                    st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
                    st.markdown(content)
                    st.markdown("---")

def display_write_blog():
    st.title("Write a New Blog Post")

    email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
    if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
        st.warning("You do not have permission to write a blog post.")
        return

    title = st.text_input("Title")
    content = st_quill(placeholder="Write your blog post here...", key="blog_editor")

    if st.button("Publish"):
        if title and content and content.strip():  # Check if content is not empty
            save_blog_post(title, content, st.session_state['user']['name'])
            st.success("Post published!")
            st.session_state.page = "mindspace"
            st.experimental_rerun()  # Rerun to show the newly added blog
        else:
            st.error("Title and content cannot be empty.")

def display_edit_blog():
    st.title("Edit Blog Post")
    
    # Get the current user's email to check authorization
    email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
    if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
        st.warning("You do not have permission to edit this blog post.")
        return
    
    # Get the selected blog post for editing
    if st.session_state.selected_post:
        title, content, author, timestamp = get_post_by_filename(st.session_state.selected_post)
    
        # Editable fields
        updated_title = st.text_input("Title", value=title)
        updated_content = st_quill(value=content, key="edit_blog_editor")

        if st.button("Save Changes"):
            # Save the updated blog post
            save_blog_post(updated_title, updated_content, author, update=True, filename=st.session_state.selected_post)
            st.success("Blog post updated!")
            st.session_state.page = "mindspace"
            st.experimental_rerun()  # Reload the page with the updated blog