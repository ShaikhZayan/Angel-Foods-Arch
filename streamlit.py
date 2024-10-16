import streamlit as st
import requests
import streamlit.components.v1 as components

st.title("Angel Foods")
st.markdown('')
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

# Ensure the prompt isn't empty or just whitespace
    if prompt.strip():
        # Show spinner while waiting for the model's response
        with st.spinner("Generating response..."):
            def get_response_from_api(prompt: str):
                # Define your API endpoint URL with query parameter
                # api_url = f"http://localhost:8001/main-model?query={prompt}"
                
                api_url = f"https://enhanced-redbird-artistic.ngrok-free.app/main-model?query={prompt}"
            
                # Define headers (optional, modify as needed)
                headers = {
                "accept": "application/json"
                }
            
                try:
                    # Make the POST request to your FastAPI backend
                    response = requests.post(api_url, headers=headers)
                
                    # Check if the request was successful
                    if response.status_code == 200:
                        # Access the 'result' field from the API response
                        return response.json().get("result", "No response available.")
                    else:
                        return f"Error: {response.status_code} - Unable to fetch the response from the API."
                
                except Exception as e:
                    return f"Exception occurred: {str(e)}"
        
            # Fetch the response from your AI API
            response = get_response_from_api(prompt)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(response)

# Sidebar success message
st.sidebar.success("Select a page above")