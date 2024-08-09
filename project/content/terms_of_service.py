import streamlit as st

def display_terms_of_service():
    st.markdown("""
        <style>
        * {
            color: grey !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Terms of Service")
    st.markdown("""
        <p><strong>Effective Date:</strong> July 31, 2024 </p>

        <p>Welcome to GabbleGrid. By using our services, you agree to comply with and be bound by the following terms and conditions.</p>

        <p><strong>1. Use of Services</strong></p>
        <p>You agree to use our services only for lawful purposes and in accordance with these Terms of Service.</p>

        <p><strong>2. User Accounts</strong></p>
        <p>To access certain features of our services, you may be required to create an account. You agree to provide accurate and complete information and to keep your account information updated.</p>

        <p><strong>3. Privacy</strong></p>
        <p>Your use of our services is also governed by our Privacy Policy, which can be found <a href='/?page=privacy_policy'>here</a>.</p>

        <p><strong>4. Intellectual Property</strong></p>
        <p>All content and materials available on our services are protected by intellectual property laws and are the property of GabbleGrid or its licensors. You may not use, reproduce, or distribute any content without our prior written permission.</p>

        <p><strong>5. Termination</strong></p>
        <p>We may terminate or suspend your access to our services at any time, without prior notice or liability, for any reason, including if you breach these Terms of Service.</p>

        <p><strong>6. Disclaimer of Warranties</strong></p>
        <p>Our services are provided "as is" and "as available" without any warranties of any kind, either express or implied.</p>

        <p><strong>7. Limitation of Liability</strong></p>
        <p>In no event shall GabbleGrid be liable for any indirect, incidental, special, consequential, or punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, or any loss of data, use, goodwill, or other intangible losses, resulting from your use of our services.</p>

        <p><strong>8. Changes to Terms of Service</strong></p>
        <p>We may update these Terms of Service from time to time. We will notify you of any changes by posting the new Terms of Service on our website.</p>

        <p><strong>9. Contact Us</strong></p>
        <p>If you have any questions about these Terms of Service, please contact us at gaurav.narasimhan@berkeley.edu.</p>
    """, unsafe_allow_html=True)
