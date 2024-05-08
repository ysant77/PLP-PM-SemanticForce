# Native Libraries
import uuid
import json
import ast
import os
import shutil
import sys

# Third-party Libraries
import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
from datetime import datetime

# Python Libraries
sys.path.insert(1, 'Backend')

from chatbot_inference import chatbot_setup, chatbot_infer


home_dir = os.getcwd()

st.set_page_config(page_title="📌 Semantic Force", layout="wide")


st.title('📌 Welcome to Semantic Force!')
st.subheader("Start anytime by asking your question below")
st.write("This bot is taught to align to financial data and insights of various companies. Feed either 10K or annual report to it and ask questions to seek your answers quickly. Try asking the latest news from it too!")

@st.cache_resource()
def load_model():
    return chatbot_setup()
# chatbot_model, chatbot_tokenizer, llama_llm = chatbot_setup()

if 'question_state' not in st.session_state:
    st.session_state.question_state = False

if 'fbk' not in st.session_state:
    st.session_state.fbk = str(uuid.uuid4())


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def display_answer():
    for entry in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(entry["question"])
        with st.chat_message("ai"):
            if (entry["command"] == 'news_fetch_and_summarize'):
                display_columns_news(entry["answer"])
            elif (entry["command"] == 'summarize_10k'):
                display_columns_10k(entry["answer"])
            else:
                st.write(entry["answer"])


        # Do not display the feedback field since we are still about to ask the user.
        if 'feedback' not in entry:
            continue

        # If there is no feedback show N/A
        if "feedback" in entry:
            st.write(f"Feedback: {entry['feedback']}")
        else:
            st.write("Feedback: N/A")


def create_answer(question):
    """Add question/answer to history."""
    # Don't save to history if question is None, need this since streamlit reruns to get the feedback.
    if question is None:
        return

    # ********* Send to Chatbot and get output here *********
    chatbot_model, chatbot_tokenizer, llama_llm = load_model()
    try:
        if save_path is not None:
            print(f'save_path: {save_path}')
            answer, command = chatbot_infer(question, chatbot_model, chatbot_tokenizer, llama_llm, save_path)
        else:
            answer, command = chatbot_infer(question, chatbot_model, chatbot_tokenizer, llama_llm)
    except:
        answer, command = chatbot_infer(question, chatbot_model, chatbot_tokenizer, llama_llm)


    print(f'command: {command}')
 
    st.session_state.chat_history.append({
        "question": question,
        "answer":  answer,
        "command": command,
    })


def fbcb(response):
    """Update the history with feedback.
    
    The question and answer are already saved in history.
    Now we will add the feedback in that history entry.
    """
    last_entry = st.session_state.chat_history[-1]  # get the last entry
    last_entry.update({'feedback': response})  # update the last entry
    st.session_state.chat_history[-1] = last_entry  # replace the last entry
    # display_answer()  # display hist

    # Create a new feedback by changing the key of feedback component.
    st.session_state.fbk = str(uuid.uuid4())

def display_columns_news(data_path):
    """
    Displays three columns with news information from an Excel file.

    Args:
        data_path: Path to the Excel file containing the news data.
    """

    # Read data
    df = pd.read_csv(data_path)

    # Extract data from specific columns and rows
    Title = df['title'][0:3].tolist()
    #   date =  df['datetime_str'] = df['datetime'].dt.strftime("%Y-%m-%d")
    df['date'] += ' 2024'
    date = df['date'].apply(lambda x:datetime.strptime(x, "%b %d %Y"))
    Summary = df['Summary'][0:3].apply(lambda x: x.replace("$", "\$")).tolist() # To overcome the Markdown formatting issue with dollar sign
    links = df['link'][0:3].tolist()

    news_container = st.container()
    with news_container:
        st.markdown('Here are some of the latest news surrounding the company.')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(Title[0])
            st.caption(date[0])
            st.markdown(Summary[0]) 
            st.link_button("Read more", links[0])

        with col2:
            st.subheader(Title[1])
            st.caption(date[1])
            st.markdown(Summary[1]) 
            st.link_button("Read more", links[1])

        with col3:
            st.subheader(Title[2])
            st.caption(date[2])
            st.markdown(Summary[2]) 
            st.link_button("Read more", links[2])
    
    return news_container

def display_columns_10k(data_path):

    # Read data
    print(f'data_path:{data_path}')
    df = pd.read_csv(data_path)

    summary_arr = df['Summary'].apply(lambda x: x.replace("$", "\$")).tolist() # To overcome the Markdown formatting issue with dollar sign

    news_container = st.container()
    with news_container:
        st.markdown('The 10K report is summarized as follows:')
        for (item_num, summary) in zip(df['Item Number'], summary_arr):
            st.subheader(item_num)
            st.markdown(summary)
            st.markdown('####')
    
    return news_container




with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"], key="upload")
    # Make sure chat input area is positioned below this container 

    # Conditional to handle file upload
    if uploaded_file is not None:
        # Get the full path to save the file (modify as needed) 
        save_path = os.path.join("Backend/PDF_files", uploaded_file.name)

        # Open the file in write-binary mode 
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write the file contents
        st.success("File saved successfully!")
    
    if st.button('Save Chat History'):
        try:
            history_filename = os.path.join(home_dir, f"Backend/data/chat_history/chat_history_{datetime.today().strftime('%d_%b_%Y_%HH_%MM_%SS')}.csv")
            shutil.copyfile('test_feedback.csv', history_filename)
            st.write(f'Chat History saved under {history_filename} 👍')
        except:
            st.write('Start talking to the chatbot first!')


# Starts here.
question = st.chat_input(placeholder="Type your question here")
if question:
    st.session_state.question_state = True


if st.session_state.question_state:
    create_answer(question)
    display_answer()

    # Pressing a button in feedback reruns the code.
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional]",
        align="flex-start",
        key=st.session_state.fbk,
        on_submit=fbcb,
    )

    test_df = pd.DataFrame(st.session_state.chat_history)
    test_df.dropna(inplace=True,how='any')
    try:
        test_df['feedback']=test_df['feedback'].apply(lambda x:x['score'].replace('👍','1').replace('👎','0')) 
    except:
        pass
    test_df.to_csv('test_feedback.csv',index=False)