################# File 1: Base Version - different shaped buttons ##################################

# import streamlit as st
# from datetime import datetime
# import os
# import yaml
# from streamlit_quill import st_quill

# BLOG_DIR = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/02.Blog'

# def save_blog_post(title, content, author):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{timestamp}_{title.replace(' ', '_')}.md"
#     filepath = os.path.join(BLOG_DIR, filename)
    
#     metadata = {
#         'title': title,
#         'author': author,
#         'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
    
#     with open(filepath, 'w') as f:
#         f.write("---\n")
#         yaml.dump(metadata, f)
#         f.write("---\n\n")
#         f.write(content)

# def get_all_posts():
#     posts = []
#     for filename in os.listdir(BLOG_DIR):
#         if filename.endswith('.md'):
#             filepath = os.path.join(BLOG_DIR, filename)
#             with open(filepath, 'r') as f:
#                 content = f.read()
#                 metadata, post_content = content.split('---\n', 2)[1:]
#                 metadata = yaml.safe_load(metadata)
#                 posts.append((filename, metadata['title'], metadata['author'], metadata['date']))
#     # Sort by date in descending order to show the most recent first
#     return sorted(posts, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), reverse=True)

# def get_post_by_filename(filename):
#     filepath = os.path.join(BLOG_DIR, filename)
#     with open(filepath, 'r') as f:
#         content = f.read()
#         metadata, post_content = content.split('---\n', 2)[1:]
#         metadata = yaml.safe_load(metadata)
#     return metadata['title'], post_content.strip(), metadata['author'], metadata['date']

# def display_blog():
#     if 'page' not in st.session_state:
#         st.session_state.page = "mindspace"
    
#     if 'selected_post' not in st.session_state:
#         st.session_state.selected_post = None

#     posts = get_all_posts()
#     if posts and st.session_state.selected_post is None:
#         st.session_state.selected_post = posts[0][0]

#     st.markdown("<h4>Blog</h4>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 3])

#     # Default blue color for blog post buttons
#     button_color = '#ADD8E6'

#     with col1:
#         st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
#         posts = get_all_posts()
#         if posts:
#             for filename, title, author, timestamp in posts:
#                 with st.container():
#                     st.markdown(f"""
#                         <div style="
#                             background-color: {button_color};
#                             padding: 10px;
#                             border-radius: 5px;
#                             margin-bottom: 10px;
#                             cursor: pointer;
#                             height: 100px;
#                             display: flex;
#                             align-items: center;
#                             justify-content: center;
#                             text-align: center;
#                         " onclick="
#                             document.dispatchEvent(new CustomEvent('streamlit:set_page_and_post', {{
#                                 detail: {{ page: 'view_post', post: '{filename}' }}
#                             }}))
#                         ">
#                             <p style="margin: 0;">{title}</p>
#                         </div>
#                     """, unsafe_allow_html=True)
#         else:
#             st.write("No blog posts published yet.")

#         st.markdown("---")

#         # Streamlit button for "Write a New Blog"
#         email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#         if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#             if st.button("Write a New Blog", key="write_new_blog", use_container_width=True):
#                 st.session_state.page = "write_blog"
        
#         # Style the button to make it look distinct and oval-shaped
#         st.markdown("""
#             <style>
#                 div.stButton > button {
#                     background-color: #98FB98 !important;
#                     color: black !important;
#                     font-weight: bold !important;
#                     border-radius: 50px !important;  /* Oval shape */
#                     height: 60px !important;  /* Adjust height for the oval shape */
#                 }
#             </style>
#         """, unsafe_allow_html=True)

#     with col2:
#         if st.session_state.page == "view_post" and st.session_state.selected_post:
#             post = get_post_by_filename(st.session_state.selected_post)
#             if post:
#                 title, content, author, timestamp = post
#                 st.subheader(title)
#                 st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
#                 st.markdown(content)
#                 st.markdown("---")
#         elif st.session_state.page == "write_blog":
#             display_write_blog()
#         else:
#             if st.session_state.selected_post:
#                 post = get_post_by_filename(st.session_state.selected_post)
#                 if post:
#                     title, content, author, timestamp = post
#                     st.subheader(title)
#                     st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
#                     st.markdown(content)
#                     st.markdown("---")

#     # Add this JavaScript to handle the custom event for other blog posts
#     st.markdown("""
#     <script>
#     document.addEventListener('streamlit:set_page_and_post', function(e) {
#         const data = e.detail;
#         window.parent.postMessage({
#             type: 'streamlit:set_state',
#             state: { page: data.page, selected_post: data.post }
#         }, '*');
#     });
#     </script>
#     """, unsafe_allow_html=True)


# def display_write_blog():
#     st.title("Write a New Blog Post")

#     email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#     if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#         st.warning("You do not have permission to write a blog post.")
#         return

#     title = st.text_input("Title")
#     content = st_quill(placeholder="Write your blog post here...", key="blog_editor")

#     if st.button("Publish"):
#         if title and content and content.strip():  # Check if content is not empty
#             save_blog_post(title, content, st.session_state['user']['name'])
#             st.success("Post published!")
#             st.session_state.page = "mindspace"
#             st.experimental_rerun()  # Rerun to show the newly added blog
#         else:
#             st.error("Title and content cannot be empty.")

#################### File 2 Combined save and edit - wrong shape ####################################################

# import streamlit as st
# from datetime import datetime
# import os
# import yaml
# from streamlit_quill import st_quill

# BLOG_DIR = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/02.Blog'

# def save_blog_post(title, content, author, update=False, filename=None):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     if not update:
#         # New post
#         filename = f"{timestamp}_{title.replace(' ', '_')}.md"
#     filepath = os.path.join(BLOG_DIR, filename)
    
#     metadata = {
#         'title': title,
#         'author': author,
#         'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
    
#     with open(filepath, 'w') as f:
#         f.write("---\n")
#         yaml.dump(metadata, f)
#         f.write("---\n\n")
#         f.write(content)

# def get_all_posts():
#     posts = []
#     for filename in os.listdir(BLOG_DIR):
#         if filename.endswith('.md'):
#             filepath = os.path.join(BLOG_DIR, filename)
#             with open(filepath, 'r') as f:
#                 content = f.read()
#                 metadata, post_content = content.split('---\n', 2)[1:]
#                 metadata = yaml.safe_load(metadata)
#                 posts.append((filename, metadata['title'], metadata['author'], metadata['date']))
#     # Sort by date in descending order to show the most recent first
#     return sorted(posts, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), reverse=True)

# def get_post_by_filename(filename):
#     filepath = os.path.join(BLOG_DIR, filename)
#     with open(filepath, 'r') as f:
#         content = f.read()
#         metadata, post_content = content.split('---\n', 2)[1:]
#         metadata = yaml.safe_load(metadata)
#     return metadata['title'], post_content.strip(), metadata['author'], metadata['date']

# def display_blog():
#     if 'page' not in st.session_state:
#         st.session_state.page = "mindspace"
    
#     if 'selected_post' not in st.session_state:
#         st.session_state.selected_post = None

#     posts = get_all_posts()
#     if posts and st.session_state.selected_post is None:
#         st.session_state.selected_post = posts[0][0]

#     col1, col2 = st.columns([1, 3])

#     # Default blue color for blog post buttons
#     button_color = '#ADD8E6'

#     with col1:
#         st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
#         posts = get_all_posts()
#         if posts:
#             email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None

#             for i, (filename, title, author, timestamp) in enumerate(posts):
#                 formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%d, %b, %Y")
                
#                 with st.container():
#                     col_left = st.columns(1)
                    
#                     # Blog post clickable container
#                     with col_left[0]:
#                         if st.button(f"{title} - Published on {formatted_timestamp}", key=f"blog_{filename}_{i}"):
#                             # If user is logged in, redirect to the edit page, otherwise view the post
#                             if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#                                 st.session_state.page = "edit_blog"
#                                 st.session_state.selected_post = filename
#                             else:
#                                 st.session_state.page = "view_post"
#                                 st.session_state.selected_post = filename

#         else:
#             st.write("No blog posts published yet.")

#         st.markdown("---")

#         # Streamlit button for "Write a New Blog"
#         email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#         if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#             if st.button("Write a New Blog", key="write_new_blog"):
#                 st.session_state.page = "write_blog"

#     with col2:
#         if st.session_state.page == "view_post" and st.session_state.selected_post:
#             post = get_post_by_filename(st.session_state.selected_post)
#             if post:
#                 title, content, author, timestamp = post
#                 st.markdown(f"<h2 style='color: blue;'>{title}</h2>", unsafe_allow_html=True)
#                 st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
#                 st.markdown(content)
#                 st.markdown("---")
#         elif st.session_state.page == "edit_blog" and st.session_state.selected_post:
#             display_edit_blog()
#         elif st.session_state.page == "write_blog":
#             display_write_blog()
#         else:
#             if st.session_state.selected_post:
#                 post = get_post_by_filename(st.session_state.selected_post)
#                 if post:
#                     title, content, author, timestamp = post
#                     st.markdown(f"<h2 style='color: teal;'>{title}</h2>", unsafe_allow_html=True)
#                     st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
#                     st.markdown(content)
#                     st.markdown("---")

#     # Add this JavaScript to handle the custom event for other blog posts
#     st.markdown("""
#     <script>
#     document.addEventListener('streamlit:set_page_and_post', function(e) {
#         const data = e.detail;
#         window.parent.postMessage({
#             type: 'streamlit:set_state',
#             state: { page: data.page, selected_post: data.post }
#         }, '*');
#     });
#     </script>
#     """, unsafe_allow_html=True)

# def display_write_blog():
#     st.title("Write a New Blog Post")

#     email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#     if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#         st.warning("You do not have permission to write a blog post.")
#         return

#     title = st.text_input("Title")
#     content = st_quill(placeholder="Write your blog post here...", key="blog_editor")

#     if st.button("Publish"):
#         if title and content and content.strip():  # Check if content is not empty
#             save_blog_post(title, content, st.session_state['user']['name'])
#             st.success("Post published!")
#             st.session_state.page = "mindspace"
#             st.experimental_rerun()  # Rerun to show the newly added blog
#         else:
#             st.error("Title and content cannot be empty.")

# def display_edit_blog():
#     st.title("Edit Blog Post")
    
#     # Get the current user's email to check authorization
#     email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#     if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#         st.warning("You do not have permission to edit this blog post.")
#         return
    
#     # Get the selected blog post for editing
#     if st.session_state.selected_post:
#         title, content, author, timestamp = get_post_by_filename(st.session_state.selected_post)
    
#         # Editable fields
#         updated_title = st.text_input("Title", value=title)
#         updated_content = st_quill(value=content, key="edit_blog_editor")

#         if st.button("Save Changes"):
#             # Save the updated blog post
#             save_blog_post(updated_title, updated_content, author, update=True, filename=st.session_state.selected_post)
#             st.success("Blog post updated!")
#             st.session_state.page = "mindspace"
#             st.experimental_rerun()  # Reload the page with the updated blog


############### File 3: New Approach - Startover with File 2 and add 'edit' WRONG SHAPE ##########################

# import streamlit as st
# from datetime import datetime
# import os
# import yaml
# from streamlit_quill import st_quill

# BLOG_DIR = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/02.Blog'

# def save_blog_post(title, content, author, filename=None):
#     if not filename:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{timestamp}_{title.replace(' ', '_')}.md"
    
#     filepath = os.path.join(BLOG_DIR, filename)
    
#     metadata = {
#         'title': title,
#         'author': author,
#         'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
    
#     with open(filepath, 'w') as f:
#         f.write("---\n")
#         yaml.dump(metadata, f)
#         f.write("---\n\n")
#         f.write(content)

# def get_all_posts():
#     posts = []
#     for filename in os.listdir(BLOG_DIR):
#         if filename.endswith('.md'):
#             filepath = os.path.join(BLOG_DIR, filename)
#             with open(filepath, 'r') as f:
#                 content = f.read()
#                 metadata, post_content = content.split('---\n', 2)[1:]
#                 metadata = yaml.safe_load(metadata)
#                 posts.append((filename, metadata['title'], metadata['author'], metadata['date']))
#     return sorted(posts, key=lambda x: datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), reverse=True)

# def get_post_by_filename(filename):
#     filepath = os.path.join(BLOG_DIR, filename)
#     with open(filepath, 'r') as f:
#         content = f.read()
#         metadata, post_content = content.split('---\n', 2)[1:]
#         metadata = yaml.safe_load(metadata)
#     return metadata['title'], post_content.strip(), metadata['author'], metadata['date']

# def display_blog():
#     if 'page' not in st.session_state:
#         st.session_state.page = "mindspace"
    
#     if 'selected_post' not in st.session_state:
#         st.session_state.selected_post = None

#     posts = get_all_posts()
#     if posts and st.session_state.selected_post is None:
#         st.session_state.selected_post = posts[0][0]

#     st.markdown("<h4>Blog</h4>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 3])

#     button_color = '#ADD8E6'

#     with col1:
#         st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
#         if posts:
#             for filename, title, author, timestamp in posts:
#                 with st.container():
#                     if st.button(title, key=filename):
#                         st.session_state.selected_post = filename
#                         st.session_state.page = "view_post"
#         else:
#             st.write("No blog posts published yet.")

#         st.markdown("---")

#         email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#         if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#             if st.button("Write a New Blog", key="write_new_blog", use_container_width=True):
#                 st.session_state.page = "write_blog"
        
#         # Modified CSS for smaller, rounded rectangle buttons
#         st.markdown("""
#             <style>
#                 div.stButton > button {
#                     background-color: #98FB98 !important;
#                     color: black !important;
#                     font-weight: bold !important;
#                     border-radius: 12px !important;  /* Rounded Rectangle */
#                     height: 40px !important;  /* Smaller height */
#                     width: 200px !important;  /* Adjusted width */
#                     margin-bottom: 5px !important;  /* Spacing between buttons */
#                 }
#             </style>
#         """, unsafe_allow_html=True)

#     with col2:
#         if st.session_state.page == "view_post" and st.session_state.selected_post:
#             post = get_post_by_filename(st.session_state.selected_post)
#             if post:
#                 title, content, author, timestamp = post
#                 st.subheader(title)
#                 st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
                
#                 # Editable blog content
#                 edited_content = st_quill(value=content, key="edit_content")
#                 if st.button("Save Changes", key="save_changes"):
#                     save_blog_post(title, edited_content, author, st.session_state.selected_post)
#                     st.success("Post updated successfully!")
#                     st.experimental_rerun()  # Rerun to show the updated blog
#         elif st.session_state.page == "write_blog":
#             display_write_blog()
#         else:
#             if st.session_state.selected_post:
#                 post = get_post_by_filename(st.session_state.selected_post)
#                 if post:
#                     title, content, author, timestamp = post
#                     st.subheader(title)
#                     st.markdown(f"**Author:** {author}  |  **Published on:** {timestamp}")
#                     st.markdown(content)
#                     st.markdown("---")


# def display_write_blog():
#     st.title("Write a New Blog Post")

#     email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
#     if email not in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
#         st.warning("You do not have permission to write a blog post.")
#         return

#     title = st.text_input("Title")
#     content = st_quill(placeholder="Write your blog post here...", key="blog_editor")

#     if st.button("Publish"):
#         if title and content and content.strip():  # Check if content is not empty
#             save_blog_post(title, content, st.session_state['user']['name'])
#             st.success("Post published!")
#             st.session_state.page = "mindspace"
#             st.experimental_rerun()  # Rerun to show the newly added blog
#         else:
#             st.error("Title and content cannot be empty.")

############################### File 4: Startover from File 2 and fix the shape bug ########
############################### buttons not clickable - however, good button shape ######

import streamlit as st
from datetime import datetime
import os
import yaml
from streamlit_quill import st_quill

BLOG_DIR = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/02.Blog'

def save_blog_post(title, content, author, update=False, filename=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not update:
        # New post
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
    # Sort by date in descending order to show the most recent first
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

    col1, col2 = st.columns([1, 3])

    # Updated button color and style
    button_color = '#4682B4'  # Steel Blue
    button_hover_color = '#5F9EA0'  # Cadet Blue

    with col1:
        st.markdown("<h5>Published Blogs</h5>", unsafe_allow_html=True)
        posts = get_all_posts()
        if posts:
            email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None

            for i, (filename, title, author, timestamp) in enumerate(posts):
                formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y")
                
                # Custom CSS for the button
                st.markdown(f"""
                    <style>
                        .blog-button-{i} {{
                            background-color: {button_color};
                            color: white;
                            border: none;
                            padding: 10px 15px;
                            margin: 5px 0;
                            border-radius: 5px;
                            cursor: pointer;
                            width: 100%;
                            text-align: left;
                            transition: background-color 0.3s;
                        }}
                        .blog-button-{i}:hover {{
                            background-color: {button_hover_color};
                        }}
                    </style>
                """, unsafe_allow_html=True)

                # Create a clickable div that looks like a button
                st.markdown(f"""
                    <div class="blog-button-{i}" 
                         onclick="
                            document.dispatchEvent(new CustomEvent('streamlit:set_page_and_post', {{
                                detail: {{ 
                                    page: '{('edit_blog' if email in ['gaurav.narasimhan@gmail.com', 'gaurav.narasimhan@berkeley.edu'] else 'view_post')}', 
                                    post: '{filename}'
                                }}
                            }}))
                         ">
                        <strong>{title}</strong><br>
                        <small>Published on {formatted_timestamp}</small>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.write("No blog posts published yet.")

        st.markdown("---")

        # Streamlit button for "Write a New Blog"
        email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
        if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
            if st.button("Write a New Blog", key="write_new_blog"):
                st.session_state.page = "write_blog"

    with col2:
        if st.session_state.page == "view_post" and st.session_state.selected_post:
            post = get_post_by_filename(st.session_state.selected_post)
            if post:
                title, content, author, timestamp = post
                st.markdown(f"<h2 style='color: blue;'>{title}</h2>", unsafe_allow_html=True)
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

    # Add this JavaScript to handle the custom event for blog posts
    st.markdown("""
    <script>
    document.addEventListener('streamlit:set_page_and_post', function(e) {
        const data = e.detail;
        window.parent.postMessage({
            type: 'streamlit:set_state',
            state: { page: data.page, selected_post: data.post }
        }, '*');
    });
    </script>
    """, unsafe_allow_html=True)

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

