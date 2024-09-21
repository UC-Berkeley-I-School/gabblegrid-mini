import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from utils.api_keys_private import GMAIL_APP_PASSWORD  # Import the password from the file

def send_email(name, email, subject, message):
    try:
        # Set up the email server
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "gaurav.narasimhan@gmail.com"  # Replace with your email
        smtp_password = GMAIL_APP_PASSWORD  # Use the password from api_keys_private.py

        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = "gaurav.narasimhan@berkeley.edu"
        msg['Subject'] = subject
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, "gaurav.narasimhan@berkeley.edu", msg.as_string())
        server.quit()
        st.success("Your message has been sent successfully!")
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}")

def display_contact_us_tab():
    # st.header("Contact Us")
    
    # Two-column layout: image on the left, form on the right
    col1, col2 = st.columns([1, 1])
    
    # Left column: display image and attribution
    with col1:
        st.image("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/12.Contact_Us/contact_us_01.gif", use_column_width=True)
        st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Right column: display the contact form
    with col2:
        # st.markdown("<h3>We'd Love to Hear From You</h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: grey;'>We'd Love to Hear From You!</h3>", unsafe_allow_html=True)
        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            subject = st.text_input("Subject")
            message = st.text_area("Message")

            # Submit button
            submit_button = st.form_submit_button("Send")

            if submit_button:
                if name and email and subject and message:
                    send_email(name, email, subject, message)
                else:
                    st.error("Please fill in all fields.")

