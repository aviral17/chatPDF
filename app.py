import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  #for creating chunks of characters for get_text_chunks() pdf text
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS   #FAISS is like PineCone or Chroma or its just like a Database that allows us to store all of Numerical 0,1 bytes of our text chunks
                                            # We will be storing all the generated embeddings in our own machine instead of any cloud or anything like that as of NOW, It will be erased once we close the application
from langchain.chat_models import ChatOpenAI                                            
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain # Allows us to chat with our text of vectorstore and have some memory to it    
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub                                      

def get_pdf_text(pdf_docs):
    text=""
    # There can be multiple PDFs so, looping through each pdf and initializing one PDF reader object for each PDF as  `pdf_reader`
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) #initializing with pdf object, this function creates pdf objects that has pages, so we are also going through the loop of pages below
        for page in pdf_reader.pages:
            text += page.extract_text()  # extracts raw text from the respective pages of pdf

    return text


    # 1000 characters, as 1000 may end up with like upto word `.... ni`, and next character set = `ce. what about.....`, so we can start this next character set 200 words before so it contains all the words and so their meanings, etc,  so we kept chunk_overlap=200 200 characters before for the next characterset
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len   
    )
        #len from python
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(mode_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Creating Database locally
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversaton_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, #our language model
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversaton_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) #NOTE: that this st.session_state.... already remember our previous questions so the bot will know the context better 
    st.session_state.chat_history = response['chat_history'] # chat_history array of response object

    # Below the MSG will be replaced by the userinput and bot response output for the respective templates
    # chathistory is an array contains our and bot's response, so the EVEN position is our question/query and ODD position is Bot's response, remember to use st.write(...) like for any doubt, like st.write(response) 
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()  #Through this, langchain will be able to access our key from .env file, remember naming convention inside .env as per langchain
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # css and html templates we imported from htmlTemplates.py
   

# Never start using conversation or chat_history without initializing it to NONE if they dont exist already
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):  #while Processing
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                
                #create vector store ----> for text_chunks to bytes
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                conversation = get_conversation_chain(vectorstore) #All it does is it takes the history of the conversation and returns the next element of the conversation,
                               # we gonna be using it in the entire Application lateron 
                               # NOTE: If we click `submit` button of Streamlit, its gonna reload the entire thing and so our conversation history gone, and it will `reinitialize` some variables (above variables), So for it to not happen like this, if we want some variables to be persistent over time, so we can do st.session_state.<variable_name> , here its st.session_state.conversation and that way the variable is linked to the session state of the application and the application knows that this varibale is not supposed to be `re-initialized`  
    
    


if __name__ == '__main__':
    main()        