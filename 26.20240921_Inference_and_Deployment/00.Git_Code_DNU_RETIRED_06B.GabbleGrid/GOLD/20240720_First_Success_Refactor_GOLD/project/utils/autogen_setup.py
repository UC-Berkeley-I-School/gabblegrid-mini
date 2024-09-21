import json
import streamlit as st
import autogen

class TrackableAssistantAgent(autogen.AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        unique_message = (sender.name, message.get('content', '') if isinstance(message, dict) else message)
        if unique_message not in st.session_state['seen_messages']:
            st.session_state['seen_messages'].add(unique_message)
            if st.session_state['show_chat_content']:
                with st.chat_message(sender.name):
                    if isinstance(message, dict) and 'content' in message:
                        try:
                            json_content = json.loads(message['content'])
                            formatted_content = json.dumps(json_content, indent=2)
                            st.markdown(f"**{sender.name}**:\n```json\n{formatted_content}\n```")
                        except json.JSONDecodeError:
                            st.markdown(f"**{sender.name}**:\n{message['content']}")
                    else:
                        st.markdown(f"**{sender.name}**:\n{message}")
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(autogen.UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        unique_message = (sender.name, message.get('content', '') if isinstance(message, dict) else message)
        if unique_message not in st.session_state['seen_messages']:
            st.session_state['seen_messages'].add(unique_message)
            if st.session_state['show_chat_content']:
                with st.chat_message(sender.name):
                    if isinstance(message, dict) and 'content' in message:
                        try:
                            json_content = json.loads(message['content'])
                            formatted_content = json.dumps(json_content, indent=2)
                            st.markdown(f"**{sender.name}**:\n```json\n{formatted_content}\n```")
                        except json.JSONDecodeError:
                            st.markdown(f"**{sender.name}**:\n{message['content']}")
                    else:
                        st.markdown(f"**{sender.name}**:\n{message}")
        return super()._process_received_message(message, sender, silent)
