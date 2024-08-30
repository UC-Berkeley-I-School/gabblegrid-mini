import streamlit as st
import yaml
from authlib.integrations.requests_client import OAuth2Session
from google_auth_oauthlib.flow import Flow
import requests
import traceback
# from mindspace import display_mindspace  # Import the MindSpace function
from mindspace_main import display_mindspace_main  # Updated import to reflect the new structure
from utils.sidebar_utils import render_sidebar  # Add this import at the top
import os
import base64
from content.home_tab import display_home_tab
# from playground.playground_main import display_playground_tab
from content.why_agents_tab import display_why_agents_tab
from content.documentation_tab import display_documentation_tab
from content.privacy_policy import display_privacy_policy
from content.terms_of_service import display_terms_of_service
from content.admin_tab import display_admin_tab
# from content.design_tab import display_design_tab
# from content.tech_tab import display_tech_tab
# from content.transformers_tab import display_transformers_tab
from content.about_us_tab import display_about_us_tab

from utils.footer import display_footer
from playground.playground_main import display_playground_tab
from models.models_main import display_models_tab

# Set the page configuration as the first Streamlit command
st.set_page_config(
    page_title="GabbleGrid",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state='collapsed'
    # initial_sidebar_state='expanded'
)

# def handle_pdf_request():
#     if 'pdf_request' in st.session_state and st.session_state.pdf_request:
#         pdf_path = st.session_state.pdf_request
#         try:
#             with open(pdf_path, "rb") as f:
#                 pdf_content = base64.b64encode(f.read()).decode('utf-8')
#             st.session_state.pdf_request = None
#             return {"content": pdf_content}
#         except Exception as e:
#             return {"error": str(e)}
#     return {"error": "No PDF request found"}

# Add cache-control headers
st.markdown("""
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
""", unsafe_allow_html=True)


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown("""
        <style>
            .no-cache {
                Cache-Control: no-cache, no-store, must-revalidate;
                Pragma: no-cache;
                Expires: 0;
            }
        </style>
    """, unsafe_allow_html=True)

def handle_github_oauth(code, client_id, client_secret, redirect_uri):
    try:
        response = requests.post(
            'https://github.com/login/oauth/access_token',
            data={
                'client_id': client_id,
                'client_secret': client_secret,
                'code': code,
                'redirect_uri': redirect_uri
            },
            headers={'Accept': 'application/json'}
        )
        response.raise_for_status()
        token_data = response.json()
        
        if 'access_token' in token_data:
            token = token_data['access_token']
            st.session_state.token = token
            user_response = requests.get(
                'https://api.github.com/user',
                headers={'Authorization': f'token {token}'}
            )
            user_response.raise_for_status()
            user_info = user_response.json()
            st.session_state.user = {'name': user_info.get('login'), 'email': user_info.get('email')}
            st.query_params.clear()
            st.rerun()
        else:
            st.error(f"Failed to obtain access token from GitHub. Response: {token_data}")
    except Exception as e:
        st.error(f"An error occurred during GitHub authentication: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

def handle_google_oauth(code, google_client_id, google_client_secret, google_redirect_uri):
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": google_client_id,
                "client_secret": google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
        state=st.session_state.get('google_oauth_state')
    )
    flow.redirect_uri = google_redirect_uri

    try:
        flow.fetch_token(code=code)
        session = flow.authorized_session()
        user_info = session.get('https://www.googleapis.com/oauth2/v2/userinfo').json()
        st.session_state.token = flow.credentials.token
        st.session_state.user = user_info
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred during Google authentication: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

def check_auth():
    if not st.session_state.get('token') or not st.session_state.get('user'):
        st.error("Please log in to access this feature.")
        return False
    return True

def main():
    load_css('utils/styles.css')

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Print environment variable to ensure it's being detected
    # st.write(f"GABBLEGRID_ENV variable: {os.environ.get('GABBLEGRID_ENV', 'Not Found')}")

    # GitHub OAuth configuration
    client_id = config['credentials']['github']['client_id']
    client_secret = config['credentials']['github']['client_secret']
    # redirect_uri = "https://gabblegrid.com"

    # Google OAuth configuration
    google_client_id = config['credentials']['google']['client_id']
    google_client_secret = config['credentials']['google']['client_secret']
    google_redirect_uri = "https://gabblegrid.com"

#################### Redirect URI for Dev Auth only ###################
    
    # Determine the environment based on a custom environment variable
    if "GABBLEGRID_ENV" in os.environ and os.environ["GABBLEGRID_ENV"] == "dev":
        redirect_uri = "https://dev.gabblegrid.com"
        google_redirect_uri = "https://dev.gabblegrid.com"
    else:
        redirect_uri = "https://gabblegrid.com"
        google_redirect_uri = "https://gabblegrid.com"

    # Debugging output
    # st.write(f"Redirect URI: {redirect_uri}")
######################################################################    

    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user' not in st.session_state:
        st.session_state.user = None

    query_params = st.query_params
    code = query_params.get('code')
    state = query_params.get('state')

    if code and not st.session_state.get('token'):
        if state and 'github.com' in state:
            handle_github_oauth(code, client_id, client_secret, redirect_uri)
        elif state:  # Assuming if there's a state and it's not GitHub, it's Google
            handle_google_oauth(code, google_client_id, google_client_secret, google_redirect_uri)
        else:
            st.error("Authentication failed. Please try again.")

    github_auth_url = ""
    google_auth_url = ""

    if not st.session_state.get('token'):
        oauth = OAuth2Session(client_id, redirect_uri=redirect_uri)
        github_auth_url, github_state = oauth.create_authorization_url(
            'https://github.com/login/oauth/authorize',
            state=f'github.com_{st.session_state.get("github_oauth_state", "")}'
        )
        st.session_state.github_oauth_state = github_state

        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": google_client_id,
                    "client_secret": google_client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
        )
        flow.redirect_uri = google_redirect_uri
        google_auth_url, google_state = flow.authorization_url(prompt='consent')
        st.session_state.google_oauth_state = google_state

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/02.Logo/20240731_Primary.png', use_column_width=True)

    with col2:
        st.markdown(f"""
            <style>
                .header-container {{
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    # justify-content: center;
                    height: 160px;
                    # min-height: 160px;
                }}
                .header-text {{
                    # color: #8ED9F1;
                    color: #374958;
                    # color: darkgrey;
                    # font-size: 46px;
                    # font-size: 36px;
                    font-size: 5.0vw;
                    font-weight: bold;
                    # margin: 0;
                    margin: 0px 100 100 100;  /* Add a top margin to avoid overlap */
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    hyphens: auto;
                    line-height: 0.9;  /* Adjust this value as needed */
                }}
                .login-buttons {{
                    display: flex;
                    gap: 10px;
                    align-items: flex-end;
                    # margin-top: 45px;
                    margin-top: 100px;
                    # margin-top: auto
                    # align-items: center;
                    # margin-top: -250px;  /* Push the buttons up by 5 pixels */
                     # margin-top: -250px;  /* Push the buttons up by 5 pixels */
                    # transform: translateY(-30px); /* Move the buttons up by 20 pixels */
                    # transform: translateY(68px); /* Move the buttons up by 20 pixels */
                    # transform: translateY(-15px); /* Move the buttons up by 20 pixels */
                    transform: translateY(-5%); /* Use percentage instead of pixels */
                }}
                .login-buttons a img {{
                    # height: 24px;
                    height: 30px;
                }}
                .welcome-message {{
                    color: #374958;
                    font-size: 20px;
                    margin-top: -5px;  /* Adjust this value as needed */
                }}
            </style>
            <div class="header-container">
                <div class="header-text">
                    Self-Healing Clouds with AI Agents
                </div>
                <div class="login-buttons">
                    <a href="{github_auth_url}" target="_self"><img src="https://img.shields.io/badge/Login%20with-GitHub-000?logo=github&logoColor=white"></a>
                    <a href="{google_auth_url}" target="_self"><img src="https://img.shields.io/badge/Login%20with-Google-4285F4?logo=google&logoColor=white"></a>
                </div>
                {f'<div class="welcome-message">Welcome, {st.session_state.user["name"]}!</div>' if st.session_state.get('token') and 'user' in st.session_state else ''}
            </div>
        """, unsafe_allow_html=True)

    # render_sidebar('playground')  # Add this line here to always render the sidebar
    
    if st.session_state.get('token') and 'user' not in st.session_state:
        st.error("User information not available. Please try logging in again.")

    if 'page' in query_params:
        page = query_params['page']
    else:
        page = "home"

    if page == "privacy_policy":
        display_privacy_policy()
    elif page == "terms_of_service":
        display_terms_of_service()
    else:

        tab1, tab2, tab3, tab4, tab5, tab6, tab10 = st.tabs([
            "Home", "Why Agents", "Playground", "Models", "MindSpace", 
            "Documentation", "Admin"
        ])

        with tab1:
            display_home_tab()
            display_footer()

        with tab2:
            display_why_agents_tab()
            display_footer()

        with tab3:
            if check_auth():
                display_playground_tab()
            else:
                st.warning("Please log in to access the Playground tab.")
            display_footer()

        with tab4:
            if check_auth():
                display_models_tab()
            else:
                st.warning("Please log in to access the Models tab.")
            display_footer()

        with tab5:
            display_mindspace_main()
            display_footer()
        
        with tab6:
            display_documentation_tab()
            display_footer()
                
        with tab10:
            # email = st.session_state.get('user', {}).get('email')
            email = st.session_state.get('user', {}).get('email') if st.session_state.get('user') else None
            if email in ["gaurav.narasimhan@gmail.com", "gaurav.narasimhan@berkeley.edu"]:
                display_admin_tab()
            else:
                st.warning("You do not have access to this tab.")
    # Add this at the end of the main function
    if st.session_state.get('pdf_request'):
        pdf_data = handle_pdf_request()
        if "content" in pdf_data:
            pdf_content = pdf_data["content"]
            pdf_base64 = f"data:application/pdf;base64,{pdf_content}"
            st.markdown(f"<iframe src='{pdf_base64}' width='100%' height='800px'></iframe>", unsafe_allow_html=True)
        else:
            st.error(pdf_data.get("error", "An unknown error occurred."))



# # Add this JavaScript snippet just before the __name__ check
# st.components.v1.html("""
# <script>
# window.addEventListener('message', function(e) {
#     if (e.data.type === 'openPDF') {
#         const url = `/get_pdf_content?path=${encodeURIComponent(e.data.path)}&name=${encodeURIComponent(e.data.name)}`;
#         window.open(url, '_blank');
#     }
# });
# </script>
# """, height=0)

if __name__ == '__main__':
    main()

    # # Add this section to handle PDF requests
    # if st.button('Get PDF Content', key='get_pdf_content'):
    #     st.json(handle_pdf_request())