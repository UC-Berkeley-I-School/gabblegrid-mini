import streamlit as st
from pdf2image import convert_from_path

def display_why_agents_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
            }
            details summary {
                color: grey; /* Change 'read more' text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

#########################################################################################################

    # Section: Current Process
    st.markdown("""
        <div class='section'>
            <h2>Current Process</h2>
            <p>
                In the current approach to AI, tasks are often completed in a linear and isolated manner. This can be compared to typing an essay from start to finish without using backspace or making corrections along the way. The process lacks flexibility and adaptability, leading to inefficiencies and potential errors that are not addressed until the very end.
            </p>
            <p>
                This process often involves completing one step entirely before moving on to the next, with no option for revisiting or refining earlier steps. This approach limits the ability to adjust based on feedback or evolving circumstances.
            </p>
            <ul>
                <li>Step-by-step process with no revision or backtracking</li>
                <li>Errors are only discovered after all steps are completed</li>
                <li>No feedback loops or iteration during the process</li>
                <li>Minimal adaptability to changing requirements or inputs</li>
                <li>Task is completed in one go without opportunities for improvement</li>
            </ul>
            <p>
                This rigid approach can result in less effective solutions, as adjustments and optimizations cannot be made during the task execution.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Current_Process_800_2_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    #########################################################################################################
    
    # Section: Agent Process
    st.markdown("""
        <div class='section'>
            <h2>Agent Process</h2>
            <p>
                In contrast, the agent-based process is dynamic and iterative, much like writing an essay in multiple drafts. Agents constantly gather feedback, revise, and refine their actions based on new information and outcomes. This ensures that each step improves upon the previous one, leading to more efficient and accurate results.
            </p>
            <ul>
                <li>Iterative process where each step can be revisited and improved</li>
                <li>Constant feedback loops that guide refinements</li>
                <li>Flexibility to adjust to new inputs or evolving requirements</li>
                <li>Proactive error detection and resolution during the task</li>
                <li>Continuous improvement throughout the process, not just at the end</li>
            </ul>
            <p>
                Agents operate differently: they don't necessarily follow a strict, linear path. Instead, they adapt at each stage, performing tasks like researching, drafting, revising, and integrating feedback, ensuring that the final result is optimized through multiple iterations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agent_Process_800_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)


#########################################################################################################

    # Section 3: Agent Design Patterns
    st.markdown("""
        <div class='section'>
            <h2>Agent Design Patterns</h2>
            <p>
                The primary design patterns for agents involve Reflection, Tool Use, Planning, and Collaboration. Below are the key elements of each pattern, highlighting the benefits and functionality of autonomous AI agents in various workflows.
            </p>
        </div>
    """, unsafe_allow_html=True)

#########################################################################################################
    
    # Subsection: Reflection
    st.markdown("""
        <div class='section'>
            <h4 style='color: grey;'>1. Reflection</h4>
            <p>
                Reflection involves iterative refinement, allowing agents to improve over time. This process can happen through self-feedback, where agents review their own actions, or through multi-agent feedback, where multiple agents collaborate and provide feedback to each other.
            </p>
            <ul>
                <li>Iterative refinement with self-feedback for continuous improvement</li>
                <li>Multi-agent feedback for collective refinement of actions</li>
                <li>Enables agents to become more accurate and reliable with each iteration</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Reflect_800_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################    
    
    # Subsection: Tool Use
    st.markdown("""
        <div class='section'>
            <h4 style='color: grey;'>2. Tool Use</h4>
            <p>
                Tool use is an essential design pattern where agents interact with external systems through APIs and large language models (LLMs). Agents leverage these tools to perform tasks, gather information, and execute actions that go beyond the confines of their initial programming.
            </p>
            <ul>
                <li>Agents connect with massive APIs to access external systems and resources</li>
                <li>They can interact with the world outside the LLMs to gather real-time data</li>
                <li>Agents perform complex tasks autonomously, taking action based on the tools available</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Tool_Use_800_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################    
    
    # Subsection: Planning
    st.markdown("""
        <div class='section'>
            <h4 style='color: grey;'>3. Planning</h4>
            <p>
                Planning enables agents to break down complex user requests into smaller, manageable tasks. Agents can employ reasoning techniques like Chain-of-Thought (CoT) to decide the best way to accomplish a goal by selecting the right tools or models for each task.
            </p>
            <ul>
                <li>Agents use planning to fulfill complex user requests autonomously</li>
                <li>Disassemble tasks into smaller, executable components</li>
                <li>Employ reasoning techniques like Chain-of-Thought (CoT) for task optimization</li>
                <li>Assign the most suitable models or methods to each task</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Planning_800_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################    
    
    # Subsection: Collaboration
    st.markdown("""
        <div class='section'>
            <h4 style='color: grey;'>4. Collaboration</h4>
            <p>
                Collaboration design patterns focus on teamwork between multiple agents. Agents collaborate to solve tasks more efficiently, and they can pool their capabilities to achieve more complex goals. In this case, teamwork really does make the dream work.
            </p>
            <ul>
                <li>Multiple agents work together to achieve goals efficiently</li>
                <li>Leverage teamwork to accomplish tasks beyond individual capabilities</li>
                <li>Pool resources and expertise across agents to optimize outcomes</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Collaborate_800_yellow.gif', use_column_width=True)
    
    st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)


#########################################################################################################

    # Function to convert the first page of a PDF to an image and display it
    def display_first_page_of_pdf(file_path):
        """Convert the first page of a PDF to an image and display it."""
        try:
            # Convert only the first page of the PDF
            images = convert_from_path(file_path, first_page=0, last_page=1)
            # Display the first page as an image
            st.image(images[0], caption="Page 1")
        except Exception as e:
            st.error(f"Failed to load the PDF: {e}")
    
    # Custom CSS to color the download button pastel yellow
    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #FFFACD; /* Pastel yellow */
            color: black; /* Black text */
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stDownloadButton > button:hover {
            background-color: #FFEDB8; /* Darker pastel yellow on hover */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Subsection: Technical Paper in Collaboration Section
    st.markdown("""
        <div class='section'>
            <h5 style='color: grey;'>Technical Paper: More Agents Is All You Need</h5>
            <p style='color: grey;'>
                Explore the technical foundations of agent collaboration in this detailed paper. It outlines how multiple agents can be deployed effectively to work in unison, overcoming complex challenges through coordination.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Path to the PDF
    pdf_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/2024_More Agents_Is_All_You_Need.pdf'
    
    # Display only the first page of the PDF
    display_first_page_of_pdf(pdf_path)
    
    # Download button at the bottom with pastel yellow styling
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Download Technical Paper: 'More Agents Is All You Need'",
                data=file,
                file_name="2024_More_Agents_Is_All_You_Need.pdf",
                mime="application/pdf"
            )
    
    st.markdown("<hr>", unsafe_allow_html=True)

