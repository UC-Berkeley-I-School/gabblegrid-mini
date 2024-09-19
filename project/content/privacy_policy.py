import streamlit as st

def display_privacy_policy():
    st.markdown("""
        <style>
        * {
            color: grey !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Privacy Policy")
    st.markdown("""
        <p><strong>Effective Date:</strong> July 31, 2024</p>

        <p><strong>Introduction</strong></p>
        <p>Welcome to MindMesh. We are committed to protecting your privacy and ensuring you have a positive experience using our services. This Privacy Policy outlines how we collect, use, and share your personal information when you use our cloud services, including our website and applications.</p>
        
        <p><strong>1. Information We Collect</strong></p>
        <p>We collect various types of information in connection with the services we provide, including:</p>
        
        <p><strong>a. Personal Information:</strong></p>
        <ul>
            <li>Name, email address, phone number, and other contact information.</li>
            <li>Account credentials, such as username and password.</li>
        </ul>
        
        <p><strong>b. Usage Data:</strong></p>
        <ul>
            <li>Log files, event data, and other information related to your use of our services.</li>
            <li>IP addresses, browser type, and operating system.</li>
        </ul>
        
        <p><strong>c. Device Information:</strong></p>
        <ul>
            <li>Information about the devices you use to access our services, including device type, operating system, and unique device identifiers.</li>
        </ul>
        
        <p><strong>d. Cookies and Tracking Technologies:</strong></p>
        <ul>
            <li>We use cookies and similar technologies to track user activity on our services and gather demographic information.</li>
        </ul>
        
        <p><strong>2. How We Use Your Information</strong></p>
        <p>We use the information we collect for various purposes, including:</p>
        <ul>
            <li>Providing and improving our services.</li>
            <li>Personalizing your experience.</li>
            <li>Communicating with you about your account and our services.</li>
            <li>Analyzing usage patterns and trends to enhance our services.</li>
            <li>Ensuring the security and integrity of our services.</li>
        </ul>
        
        <p><strong>3. Sharing Your Information</strong></p>
        <p>We may share your information with third parties in certain circumstances, including:</p>
        <ul>
            <li>With your consent.</li>
            <li>To comply with legal obligations.</li>
            <li>To protect and defend our rights and property.</li>
            <li>In connection with a merger, acquisition, or sale of assets.</li>
        </ul>
        
        <p><strong>4. Security</strong></p>
        <p>We take reasonable measures to protect your personal information from unauthorized access, use, or disclosure. However, no security system is completely secure, and we cannot guarantee the absolute security of your information.</p>
        
        <p><strong>5. Your Choices</strong></p>
        <p>You have certain choices regarding your personal information, including:</p>
        <ul>
            <li>Accessing, updating, or deleting your information.</li>
            <li>Opting out of certain data collection and use.</li>
            <li>Withdrawing your consent for us to use your information.</li>
        </ul>
        
        <p><strong>6. Changes to This Privacy Policy</strong></p>
        <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on our website.</p>
        
        <p><strong>7. Contact Us</strong></p>
        <p>If you have any questions about this Privacy Policy, please contact us at gaurav.narasimhan@berkeley.edu.</p>
    """, unsafe_allow_html=True)
