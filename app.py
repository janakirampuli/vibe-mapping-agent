import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv

from recommandation import ProductRecommender
from system_prompt import get_system_prompt

from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# --------------------------
# Config
# --------------------------
load_dotenv()
st.set_page_config(page_title="Vibe Mapping Agent", layout="centered")

st.title("🛍️ Vibe-Based Fashion Chatbot")
st.caption("Find outfits that match your vibe ✨")

# Choose LLM source
# model_choice = st.sidebar.radio("LLM Backend", ["OpenAI", "Ollama"])
ollama_model = st.sidebar.text_input("Ollama Model", value="phi4:14b")

# if model_choice == "OpenAI":
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         openai_api_key="YOUR_OPENAI_API_KEY",
#         openai_api_base="https://api.openai.com/v1",
#         temperature=0.7
#     )
# else:
llm = ChatOllama(model=ollama_model, temperature=0.7, think=False)

# --------------------------
# Load Data and Initialize Recommender
# --------------------------
@st.cache_resource
def initialize_recommender():
    """Initialize recommender with caching to prevent recomputation."""
    return ProductRecommender('Apparels_shared.xlsx')

# Initialize recommender with caching
recommender = initialize_recommender()

catalog_df = pd.read_excel('Apparels_shared.xlsx')

# --------------------------
# System Prompt
# --------------------------
SYSTEM_PROMPT_STRING = get_system_prompt()

# --------------------------
# Helper Functions
# --------------------------
def map_vibe_to_attributes(vibe_query, followup_response=None, current_state=None, followup_count=0, last_next_query=None):
    base_prompt = f"User query: {vibe_query}"
    if followup_response:
        base_prompt += f"\nAdditional info: {followup_response}"
    if current_state:
        base_prompt += f"\nCurrent state: {json.dumps(current_state)}"
    base_prompt += f"\nNumber of follow-up questions asked so far: {followup_count}"
    if last_next_query:
        base_prompt += f"\nLast follow-up question asked: {last_next_query}"
        base_prompt += "\nIMPORTANT: Do NOT repeat the last follow-up question. Ask about different missing fields."
    if followup_count >= 2:
        base_prompt += "\nNote: You have asked enough questions. Set next_query to null and proceed with recommendations."

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT_STRING),
        HumanMessage(content=base_prompt)
    ])
    chain = prompt | llm
    response = chain.invoke({})
    return response.content

def recommend_products(structured_attrs):
    return recommender.recommend_products(structured_attrs, k=6, final_k=3)

def clean_json_response(response_text):
    """Clean and validate JSON response"""
    try:
        # Remove markdown formatting
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            # Remove first and last lines if they contain ```
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines)
        
        # Try parsing as-is first
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            # Fix common issues
            fixed = cleaned.replace(' | ', '", "').replace('|', '", "')
            # Handle specific problematic patterns
            fixed = fixed.replace('"top" | "dress" | "skirt" | "pants"', 'null')
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

# --------------------------
# Session State
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_json" not in st.session_state:
    st.session_state.conversation_json = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "vibe_seed" not in st.session_state:
    st.session_state.vibe_seed = None

if "followup_count" not in st.session_state:
    st.session_state.followup_count = 0

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = ["Debug console"]

# --------------------------
# Display Chat History
# --------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display the main content
            if message["content"]:
                st.markdown(message["content"])
            
            # Display dataframe if present
            if "dataframe" in message and message["dataframe"] is not None:
                st.dataframe(message["dataframe"])
        else:
            st.markdown(message["content"])

# --------------------------
# Chat Input and Processing
# --------------------------
placeholder = "Describe the vibe you're going for..."

if prompt := st.chat_input(placeholder):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the message
    with st.chat_message("assistant"):
        with st.spinner("Mapping your vibe..."):
            last_next_query = st.session_state.get("pending_query")
            if st.session_state.pending_query:
                updated = map_vibe_to_attributes(
                    vibe_query=st.session_state.vibe_seed,
                    followup_response=prompt,
                    current_state=st.session_state.conversation_json,
                    followup_count=st.session_state.followup_count,
                    last_next_query=last_next_query
                )
            else:
                st.session_state.vibe_seed = prompt
                st.session_state.followup_count = 0  # Reset counter for new conversation
                updated = map_vibe_to_attributes(prompt, followup_count=0, last_next_query=None)

            # Clean and parse JSON
            parsed_json = clean_json_response(updated)
            
            if parsed_json:
                st.session_state.conversation_json = parsed_json

                debug_content = f"**Json formed:**\n```"
                debug_content += "{\n"
                debug_content += json.dumps(parsed_json, indent=2)
                debug_content += "\n"
                # Instead of printing in chat, show in a debug console expander
                if "debug_logs" not in st.session_state:
                    st.session_state.debug_logs = []
                st.session_state.debug_logs.append(debug_content)

                # Prepare assistant response content
                assistant_content = ""
                dataframe_content = None
                

                next_query = parsed_json.get("next_query")
                if next_query and st.session_state.followup_count <= 3:
                    followup_msg = f"{next_query}"
                    st.markdown(followup_msg)
                    assistant_content = followup_msg
                    st.session_state.pending_query = next_query
                    st.session_state.followup_count += 1
                else:
                    completion_msg = "Here are my top recommendations:"
                    st.session_state.pending_query = None
                    
                    # Show product recommendations
                    final_json = parsed_json.copy()
                    final_json.pop("next_query", None)
                    top_recs = recommend_products(final_json)
                

                    # Generate LLM summary for top recommendations
                    summary_prompt = f"""
You are a helpful fashion assistant. The user described their vibe and preferences as follows:
{json.dumps(final_json, indent=2)}

Here are the top 3 recommended products:
{top_recs.to_markdown(index=False)}

Write a friendly, concise summary (2-4 sentences) explaining why these products are a great fit for the user's vibe and preferences. Highlight any key matches (e.g., fit, fabric, color, occasion, etc.).
"""
                    summary_chain = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are a helpful fashion assistant."),
                        HumanMessage(content=summary_prompt)
                    ]) | llm
                    summary_response = summary_chain.invoke({})
                    summary_text = summary_response.content if hasattr(summary_response, 'content') else summary_response
                    
                    st.markdown(completion_msg)
                    st.dataframe(top_recs)
                    st.markdown(f"**Why these picks?**\n\n{summary_text}")

                    
                    assistant_content = completion_msg + "\n\n" + summary_text
                    dataframe_content = top_recs
                    
                    # Reset for next conversation
                    st.session_state.followup_count = 0

                # Add complete message to history (this will be displayed in chat history later)
                assistant_message = {
                    "role": "assistant", 
                    "content": assistant_content,
                    "dataframe": dataframe_content
                }
                st.session_state.messages.append(assistant_message)


            else:
                error_msg = f"Failed to parse model response. Please try again.\n\n**Raw response:**\n``````"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "dataframe": None
                })

with st.sidebar.expander("Debug Console", expanded=True):
    for log in st.session_state.debug_logs[-10:]:
        st.markdown(log)
