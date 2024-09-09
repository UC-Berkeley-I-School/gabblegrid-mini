import streamlit as st

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
    
    # Section 1
    st.markdown("""
        <div class='section'>
            <h2>Why Agents?</h2>
            <p>
                In the current approach to AI, tasks are often completed in a linear and isolated manner. For example, typing an essay from start to finish without using backspace.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The traditional AI process involves tackling tasks in a step-by-step fashion, without the flexibility to adapt and refine as needed. This can lead to inefficiencies and suboptimal results. In contrast, the agent process is more dynamic and iterative. It involves writing an essay with multiple stages of drafting, researching, revising, and incorporating feedback. This allows for a more thorough and polished final product.
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_11.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 2
    st.markdown("""
        <div class='section'>
            <h2>Design Patterns</h2>
            <p>
                The primary design patterns for agents involve Reflection, Tool Use, Planning, and Collaboration.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    <strong>Reflection:</strong> Iterative refinement with self-feedback and multi-agent feedback allows agents to continuously improve their performance.<br>
                    <strong>Tool Use:</strong> Agents connect with large language models (LLMs) via massive APIs, interact with the external world, and perform tasks and actions.<br>
                    <strong>Planning:</strong> Agents elicit reasoning in LLMs using techniques like Chain-of-Thought, autonomously fulfill complex user requests, disassemble tasks, and assign suitable models to the tasks.<br>
                    <strong>Collaboration:</strong> Multiple agents collaborate effectively to achieve goals, leveraging teamwork to enhance overall performance.
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_13.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 3
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Disk I/O Errors</h2>
            <p>
                A team of agents is responsible for managing disk I/O errors. The process involves multiple agents working together to identify, analyze, and resolve the issue.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    <ul>
                        <li><strong>Detection Agent:</strong> Identifies a recurring disk I/O error in the logs.</li>
                        <li><strong>Analysis Agent:</strong> Confirms a failing disk on Node 42.</li>
                        <li><strong>Action Agent:</strong> Schedules an immediate disk replacement and workload migration.</li>
                        <li><strong>Execution Agent:</strong> Coordinates with the data center team to replace the disk and ensures no data loss.</li>
                        <li><strong>Notification Agent:</strong> Sends an email to administrators and updates the monitoring dashboard with the resolution status.</li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_14.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 4
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Identification</h2>
            <p>
                Identifying disk I/O errors involves navigating a complex environment with diverse hardware, configurations, and specialized detection models.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The complexity of identifying disk I/O errors includes:
                    <ul>
                        <li><strong>Heterogeneous Environment:</strong>
                            <ul>
                                <li><strong>Diverse Hardware:</strong> Data centers may contain disk hardware from various manufacturers, generations, and purposes, making uniform error detection challenging.</li>
                                <li><strong>Different Configurations:</strong> Variations in RAID setups, file systems, and storage architectures add layers of complexity.</li>
                            </ul>
                        </li>
                        <li><strong>Specialized Detection Models:</strong>
                            <ul>
                                <li><strong>Anomaly Detection Models:</strong> Multiple categories of anomaly detection models are required to filter out non-disk I/O errors.</li>
                                <li><strong>Fine-Tuning:</strong> Each category of models must be fine-tuned to account for the specific characteristics of different disk hardware and configurations.</li>
                            </ul>
                        </li>
                        <li><strong>Complex Error Landscape:</strong>
                            <ul>
                                <li><strong>Filtering and Classification:</strong> Initial filtering to identify disk I/O errors from a sea of diverse error logs.</li>
                                <li><strong>Detailed Analysis:</strong> Subsequent detailed analysis to pinpoint the exact nature and cause of the disk I/O error, considering factors like error rates, message patterns, and operational context.</li>
                            </ul>
                        </li>
                        <li><strong>Insights from Research:</strong>
                            <ul>
                                <li><strong>Anomaly Detection in Logs:</strong> Using entropy and statistical methods to identify alerts (Oliner et al., 2008).</li>
                                <li><strong>Diverse Alert Categories:</strong> Discovery of new alert types and refinement of tagging processes to enhance detection accuracy (Oliner & Stearley, 2007).</li>
                            </ul>
                        </li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_15.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 5
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Identification</h2>
            <p>
                Identifying disk I/O errors involves understanding various aspects of the error environment and applying specialized models for accurate detection.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The complexity of identifying disk I/O errors includes:
                    <ul>
                        <li><strong>Hardware Diversity:</strong> Different types and generations of disk hardware from multiple vendors. For example, disks from Seagate, Western Digital, and Samsung; SSDs and HDDs; various RAID configurations.</li>
                        <li><strong>Anomaly Detection Models:</strong> Need for specialized models to filter out non-disk I/O errors and focus on disk-related issues. For example, models for CPU errors, memory errors, network errors, and specific models for disk I/O errors.</li>
                        <li><strong>Fine-Tuning Requirements:</strong> Each detection model must be adjusted to account for the specific characteristics of the hardware. For example, fine-tuning model parameters based on disk manufacturer, firmware version, and operational context.</li>
                        <li><strong>Detailed Analysis:</strong> Further analysis to understand the exact nature and cause of the disk I/O error. For example, analyzing log patterns, error rates, and contextual information to differentiate between transient errors and persistent hardware failures.</li>
                        <li><strong>Contextual Factors:</strong> Operational context such as workload type, system uptime, and maintenance history that affect error detection. For example, consideration of factors like recent firmware updates, scheduled maintenance, and workload spikes that might influence disk performance and error manifestation.</li>
                        <li><strong>Research Insights:</strong> Applying insights from previous studies to enhance error detection methodologies. For example, using entropy-based methods to detect anomalies (Nodeinfo), and improving detection through refined alert categories and statistical analysis (Oliner et al., 2008).</li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_16.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 5
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Resolution</h2>
            <p>
                Resolving disk I/O errors involves multiple resolution paths, dynamic decision-making, and effective coordination and communication.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The complexity of resolving disk I/O errors includes:
                    <ul>
                        <li><strong>Multiple Resolution Paths:</strong>
                            <ul>
                                <li><strong>Potential Actions:</strong> Numerous potential actions can be taken to resolve a disk I/O error, from simple software fixes to complex hardware replacements.</li>
                                <li><strong>False Positives:</strong> Consideration of false positives and verification before taking action is crucial.</li>
                            </ul>
                        </li>
                        <li><strong>Dynamic Decision-Making:</strong>
                            <ul>
                                <li><strong>Context-Sensitive Actions:</strong> Actions must be tailored based on the specific context, such as the type of error, operational status, and hardware involved.</li>
                                <li><strong>Risk Assessment:</strong> Evaluating the risk and impact of each action, including potential downtime and data loss.</li>
                            </ul>
                        </li>
                        <li><strong>Coordination and Communication:</strong>
                            <ul>
                                <li><strong>Execution Coordination:</strong> Coordinating with multiple teams and systems to execute the chosen action seamlessly.</li>
                                <li><strong>Communication:</strong> Keeping all stakeholders informed through automated notifications and updates.</li>
                            </ul>
                        </li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_17.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 7
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Resolution</h2>
            <p>
                Resolving disk I/O errors involves a series of potential actions and considerations to effectively address the issue.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The resolution process includes:
                    <ul>
                        <li><strong>Disk Health Check:</strong> Running diagnostics to confirm if the error is transient or indicative of a failing disk.</li>
                        <li><strong>Reboot Node:</strong> Clearing temporary issues with a node reboot, useful for transient errors but may cause downtime.</li>
                        <li><strong>File System Check:</strong> Running file system checks to resolve logical errors without addressing hardware issues.</li>
                        <li><strong>Adjusting Disk I/O Timeout:</strong> Providing temporary fixes for high load issues by adjusting timeout settings.</li>
                        <li><strong>Firmware Update:</strong> Resolving known issues by updating disk firmware, requiring coordination with maintenance schedules.</li>
                        <li><strong>Data Backup:</strong> Ensuring data protection before hardware interventions.</li>
                        <li><strong>Migrating Workloads:</strong> Minimizing impact during disk maintenance by temporarily migrating workloads to other nodes.</li>
                        <li><strong>Disk Replacement:</strong> Physically replacing the faulty disk, requiring coordination with the data center team.</li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_18.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 8
    st.markdown("""
        <div class='section'>
            <h2>Contextual Example - Resolution</h2>
            <p>
                Resolving disk I/O errors involves multiple resolution paths, dynamic decision-making, and effective coordination and communication.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The complexity of resolving disk I/O errors includes:
                    <ul>
                        <li><strong>Hardware Diversity:</strong> Different types and generations of disk hardware from multiple vendors. For example, disks from Seagate, Western Digital, and Samsung; SSDs and HDDs; various RAID configurations.</li>
                        <li><strong>Anomaly Detection Models:</strong> Need for specialized models to filter out non-disk I/O errors and focus on disk-related issues. For example, models for CPU errors, memory errors, network errors, and specific models for disk I/O errors.</li>
                        <li><strong>Fine-Tuning Requirements:</strong> Each detection model must be adjusted to account for the specific characteristics of the hardware. For example, fine-tuning model parameters based on disk manufacturer, firmware version, and operational context.</li>
                        <li><strong>Detailed Analysis:</strong> Further analysis to understand the exact nature and cause of the disk I/O error. For example, analyzing log patterns, error rates, and contextual information to differentiate between transient errors and persistent hardware failures.</li>
                        <li><strong>Contextual Factors:</strong> Operational context such as workload type, system uptime, and maintenance history that affect error detection. For example, consideration of factors like recent firmware updates, scheduled maintenance, and workload spikes that might influence disk performance and error manifestation.</li>
                        <li><strong>Research Insights:</strong> Applying insights from previous studies to enhance error detection methodologies. For example, using entropy-based methods to detect anomalies (Nodeinfo), and improving detection through refined alert categories and statistical analysis (Oliner et al., 2008).</li>
                    </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_19.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 9
    st.markdown("""
        <div class='section'>
            <h2>Product Vision</h2>
            <p>
                The Agent system acts as the brain of both the Chaos Lab and production environments, integrating various components to ensure robust performance and reliability.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The Agent system integrates multiple components to manage and optimize operations:
                    <ul>
                        <li><strong>Language Models:</strong> Enable the agents to understand and process complex instructions and data.</li>
                        <li><strong>Autonomous Agents:</strong> Operate independently to manage tasks and make decisions based on real-time data.</li>
                        <li><strong>Anomaly Detection Models:</strong> Identify and respond to irregularities and potential issues in the system.</li>
                        <li><strong>Data Sources:</strong> 
                            <ul>
                                <li><strong>Hardware Data:</strong> Includes metrics like CPU usage, memory, and storage I/O (SLI-1 and SLI-2).</li>
                                <li><strong>Software Data:</strong> Comprises event logs and security logs (SLI-3).</li>
                                <li><strong>External Data:</strong> Integrates external factors such as weather data.</li>
                            </ul>
                        </li>
                        <li><strong>Server/Pod:</strong> The central unit where all data is processed and decisions are executed.</li>
                    </ul>
                    This integrated system enables proactive management and optimization, ensuring high performance and reliability in both experimental and production environments.
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_20.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 10
    st.markdown("""
        <div class='section'>
            <h2>Product Scope - MVP</h2>
            <p>
                The initial implementation of the Agent system focuses on a Minimum Viable Product (MVP) that demonstrates the core functionalities and benefits.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                    The MVP scope includes:
                    <ol>
                        <li><strong>One Server, One Event Log:</strong> Start with a single server and a single event log to establish baseline functionality.</li>
                        <li><strong>Develop Anomaly Detection Model (Supervised):</strong> Create and train an anomaly detection model using supervised learning techniques to identify and classify anomalies.</li>
                        <li><strong>Use Agent System for Final Decision:</strong> Leverage the agent system to make final decisions based on the outputs of the anomaly detection model and other inputs.</li>
                        <li><strong>Use Agent System for Remedial Action on Server:</strong> Implement automated remedial actions on the server based on the agent system's decisions, ensuring issues are addressed promptly and effectively.</li>
                    </ol>
                    This phased approach ensures a controlled rollout, allowing for incremental improvements and validation of the system's capabilities in a real-world environment.
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,40,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_21.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)