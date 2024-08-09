import streamlit as st

def display_about_us_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2 {
                color: grey; /* Change text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>About Me</h2>
            <p>
            Hey there! I'm Gaurav Narasimhan, the founder-engineer behind GabbleGrid. I lead a team of rockstar data scientists and AI engineers who are all about building autonomous agents to predict and prevent failures in SaaS cloud services. Oh, and I'm making this an open-source platform soon—so stay tuned!
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/about_us_photo_1.png', use_column_width=True)
    st.markdown("""
        <div class='section'>
            <h2>Gaurav Narasimhan</h2>
            <p><strong>Title:</strong> Founder-Engineer</p>
            <p>
            I’m currently juggling a full-time leadership role while pursuing a graduate degree in Data Science at UC Berkeley. I'm into Natural Language Processing and love balancing my time between being a student and a leader. It's all about pushing boundaries and having fun with tech.
            </p>
            <p>
            <strong>About Me:</strong><br>
            ► Always challenging the status quo and looking for ways to improve.<br>
            ► Balancing a grad program at UC Berkeley with leading a kickass team focused on AI Autonomous Agents.<br>
            ► My capstone project at UC Berkeley is all about developing these cool Autonomous Agents, perfectly aligning with my work.<br>
            ► Finding the sweet spot between academic life and professional leadership.
            </p>
            <p>
            <strong>Technical Focus and Achievements:</strong><br>
            ► Bringing Large Language Models (LLMs) into the mix to create autonomous solutions for SaaS Engineering.<br>
            ► Steering the ship on using LLMs to boost the product features.<br>
            ► Driving innovation and making sure we stick to industry standards.<br>
            ► Working with transformers like BERT, LLaMA, MPT, Falcon, Flan, and T5. Also, having fun with Langchain and Huggingface for some cool AI applications.<br>
            ► Building Visual Document Scanning solutions with LayoutLM for various needs.<br>
            ► Creating AI chatbots to enhance Employee Experience and Engineering Productivity, hitting a 20% improvement in issue resolution.<br>
            ► Teaming up with folks across disciplines to make products more efficient and competitive.
            </p>
            <p>
            <strong>Professional Insights and Contributions:</strong><br>
            ► Constantly diving into AI and ML to apply new insights to real-world problems.<br>
            ► Sharing my knowledge through technical and strategic content within the AI community.
            </p>
            <p>
            <strong>Skills & Expertise:</strong><br>
            ► Expertise in Large Language Models (LLM), Machine Learning, and Autonomous Agents.<br>
            ► Leading teams in applying AI and ML to create impactful business solutions.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    display_about_me_tab()
