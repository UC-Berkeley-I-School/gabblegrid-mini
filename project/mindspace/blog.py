# # ############################### DEFAULT GOLD - Blue rectangle files with edit ###########################

import streamlit as st
from datetime import datetime
import os
import yaml
from streamlit_quill import st_quill
import base64
import re
from streamlit.components.v1 import html
import re

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
    
    # Convert HTML content to Markdown if necessary
    if content.startswith('<'):
        # This is a simple HTML to Markdown conversion. You might want to use a more robust solution.
        markdown_content = content.replace('<p>', '').replace('</p>', '\n\n')
        markdown_content = re.sub(r'<img src="(.*?)" alt="(.*?)">', r'![Image](\1)', markdown_content)
    else:
        markdown_content = content

    # Adjust image paths
    markdown_content = re.sub(r'!\[([^\]]*)\]\(11\.Images/([^\)]+)\)', r'![Image](11.Images/\2)', markdown_content)
    
    with open(filepath, 'w') as f:
        f.write("---\n")
        yaml.dump(metadata, f)
        f.write("---\n\n")
        f.write(markdown_content)

def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '-', text).strip('-')

def get_all_posts():
    posts = []
    for filename in os.listdir(BLOG_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(BLOG_DIR, filename)
            with open(filepath, 'r') as f:
                content = f.read()
                metadata, post_content = content.split('---\n', 2)[1:]
                metadata = yaml.safe_load(metadata)
                slug = slugify(metadata['title'])
                posts.append((filename, metadata['title'], metadata['author'], metadata['date'], slug))
    return sorted(posts, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), reverse=True)

def get_post_by_filename(filename):
    filepath = os.path.join(BLOG_DIR, filename)
    with open(filepath, 'r') as f:
        content = f.read()
        metadata, post_content = content.split('---\n', 2)[1:]
        metadata = yaml.safe_load(metadata)
    
    # Fix image paths in post content
    post_content = re.sub(r'!\[Image\]\((.*?)\)', replace_image, post_content)

    return metadata['title'], post_content.strip(), metadata['author'], metadata['date']



def display_blog():
    if 'page' not in st.session_state:
        st.session_state.page = "mindspace"
    
    # Set the first blog post as the default selection if no post is selected
    posts = get_all_posts()
    if posts and 'selected_post' not in st.session_state:
        st.session_state.selected_post = posts[0][0]  # Select the most recent post by default
        st.session_state.page = "view_post"

    col1, col2 = st.columns([1, 3], gap="large")
    
    st.markdown("""
        <style>
            .blog-content {
                max-height: 150vh;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)
    
    with col1:
        posts = get_all_posts()
        if posts:
            email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None

            for i, (filename, title, author, timestamp, slug) in enumerate(posts):
                formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%d, %b, %Y")
                
                is_selected = filename == st.session_state.selected_post
                
                st.markdown(f'<div class="blog-date">Published on {formatted_timestamp}</div>', unsafe_allow_html=True)

                if st.button(f"{title}", key=f"blog_{filename}_{i}"):
                    if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
                        st.session_state.page = "edit_blog"
                        st.session_state.selected_post = filename
                    else:
                        st.query_params["post"] = slug
                        st.rerun()


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
                
                content = replace_image(content)
                st.markdown(f'<div class="blog-content">{content}</div>', unsafe_allow_html=True)
                st.markdown("---")
        elif st.session_state.page == "edit_blog" and st.session_state.selected_post:
            display_edit_blog()
        elif st.session_state.page == "write_blog":
            display_write_blog()
        else:
            st.write("Select a blog post to view its content.")

    # Add custom CSS for buttons
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
            font-size: 18px !important;
            font-weight: bold !important;
            color: grey !important;
        }
        .blog-date {
            font-size: 12px !important;
            color: #555 !important;
            margin-bottom: 5px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# def display_single_post(identifier):
#     posts = get_all_posts()
#     post = next((p for p in posts if p[4] == identifier or p[0].startswith(identifier)), None)
#     if post:
#         filename, title, author, timestamp, _ = post
#         post_content = get_post_by_filename(filename)
#         if post_content:
#             title, content, author, timestamp = post_content
#             st.markdown(f"<h2 style='color: teal;'>{title}</h2>", unsafe_allow_html=True)
#             st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
            
#             content = replace_image(content)
#             st.markdown(f'<div class="blog-content">{content}</div>', unsafe_allow_html=True)
#             st.markdown("---")
#     else:
#         st.error(f"Blog post not found for identifier: {identifier}")

def display_single_post(post_query_param):
    posts = get_all_posts()
    
    # Try matching the post by slug or by full filename
    post = next((p for p in posts if p[4] == post_query_param or p[0] == f"{post_query_param}.md"), None)
    
    if post:
        filename, title, author, timestamp, _ = post
        post_content = get_post_by_filename(filename)
        if post_content:
            title, content, author, timestamp = post_content
            st.markdown(f"<h2 style='color: teal;'>{title}</h2>", unsafe_allow_html=True)
            st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
            
            content = replace_image(content)
            st.markdown(f'<div class="blog-content">{content}</div>', unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.error("Blog post not found.")


def get_image_url(relative_path):
    return f"{BLOG_DIR}/{relative_path}"

def replace_image(match_or_content):
    def _replace(match):
        image_path = match.group(1)
        if image_path.startswith("../"):
            image_path = image_path[3:]  # Remove the "../" prefix
        elif image_path.startswith("/home"):
            image_path = os.path.relpath(image_path, BLOG_DIR)
        elif image_path.startswith("/static"):
            image_path = image_path[7:]  # Remove the "/static" prefix
        
        full_path = os.path.join(BLOG_DIR, image_path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            return f'<img src="data:image/png;base64,{encoded_image}" style="max-width: 100%; height: auto;">'
        else:
            return f'<p style="color: red;">Image not found: {image_path}</p>'
    
    if isinstance(match_or_content, str):
        return re.sub(r'<img src="(.*?)" alt="(.*?)">', _replace, match_or_content)
    else:
        return _replace(match_or_content)

def display_write_blog():
    st.title("Write a New Blog Post")

    email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
    if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
        st.warning("You do not have permission to write a blog post.")
        return

    title = st.text_input("Title")
    content = st_quill(placeholder="Write your blog post here...", key="blog_editor")
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Save the uploaded file
        save_path = os.path.join(BLOG_DIR, '11.Images', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} saved successfully!")
        
        # Insert image reference into content
        image_html = f'<img src="11.Images/{uploaded_file.name}" alt="{uploaded_file.name}">'
        content += f"\n\n{image_html}\n\n"  # Add newlines before and after the image
        
        # Update the Quill editor with the new content
        st.session_state.blog_editor = content

    if st.button("Publish"):
        if title and content and content.strip():  # Check if content is not empty
            save_blog_post(title, content, st.session_state['user']['name'])
            st.success("Post published!")
            st.session_state.page = "mindspace"
            st.experimental_rerun()  # Rerun to show the newly added blog
        else:
            st.error("Title and content cannot be empty.")

    # Display current content (for debugging)
    st.write("Current content:")
    st.write(content)

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

        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            # Save the uploaded file
            save_path = os.path.join(BLOG_DIR, '11.Images', uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} saved successfully!")
            
            # Insert image reference into content
            image_html = f'<img src="11.Images/{uploaded_file.name}" alt="{uploaded_file.name}">'
            updated_content += image_html

        if st.button("Save Changes"):
            # Save the updated blog post
            save_blog_post(updated_title, updated_content, author, update=True, filename=st.session_state.selected_post)
            st.success("Blog post updated!")
            st.session_state.page = "mindspace"
            st.experimental_rerun()  # Reload the page with the updated blog