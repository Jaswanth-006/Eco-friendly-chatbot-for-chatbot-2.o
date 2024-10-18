import streamlit as st
import base64
import os
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

# Initialize the Groq and Pinecone clients with your API keys
client = Groq(
    api_key="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",
)
pc = Pinecone(api_key="f8632be9-0aad-417d-a0df-9c45c9f3505f")
index = pc.Index("quickstart")

# Store API key in session state for chat
st.session_state.api_key = "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"

# Only show the API key input if the key is not already set
if not st.session_state.api_key:
    # Ask the user's API key if it doesn't exist
    api_key = st.text_input("Enter API Key", type="password")
    
    # Store the API key in the session state once provided
    if api_key:
        st.session_state.api_key = api_key
        st.rerun()  # Refresh the app once the key is entered to remove the input field
else:
    # If the API key exists, show the chat app
    st.title("Eco-friendly Chatbot")

    # Initialize the chat message list in session state if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.groq_chat_messages = [{"role": "system", "content": "You are a helpful assistant. The user will ask about an item, and you will respond with eco-friendly versions or alternatives."}]
        st.session_state.chat_messages = []
        
    # Display previous chat messages
    for messages in st.session_state.chat_messages:
        if messages["role"] in ["user", "assistant"]:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])
    
    # Define a function to simulate chat interaction with eco-friendly context
    def get_chat():
        # Get the latest user input and make it eco-friendly
        user_query = f"eco-friendly {st.session_state.chat_messages[-1]['content']}"
        
        # Generate embedding with the modified query
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[user_query],  # Modify the query to include "eco-friendly"
            parameters={
                "input_type": "query"
            }
        )
        results = index.query(
            namespace="ns1",
            vector=embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        
        context = ""
        for result in results.matches:
            if result['score'] > 0.8:
                context += result['metadata']['text']
        
        # Update the Groq chat messages to include the context
        st.session_state.groq_chat_messages[-1]["content"] = f"User Query: {user_query} \n Retrieved Content (optional): {context}"
        
        # Call the Groq API with the modified eco-friendly query
        chat_completion = client.chat.completions.create(
            messages=st.session_state.groq_chat_messages,
            model="llama3-8b-8192",
        )
        
        return chat_completion.choices[0].message.content

    # Handle user input
    if prompt := st.chat_input("Ask about an item and get eco-friendly alternatives!"):
        # Prepend "eco-friendly" to the user's input before sending it to the assistant
        ecofriendly_prompt = f"eco-friendly {prompt}"

        # Display user message
        with st.chat_message("user"):
            st.markdown(ecofriendly_prompt)
        
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.groq_chat_messages.append({"role": "user", "content": ecofriendly_prompt})
        
        # Get the assistant's eco-friendly response
        with st.spinner("Getting eco-friendly suggestions..."):
            response = get_chat()
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add user message and assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.groq_chat_messages.append({"role": "assistant", "content": response})
