import os
from openai import OpenAI
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from llama_parse import LlamaParse

### Parameter Data ###
PDF_FOLDER = "data"
VECTOR_FOLDER = "vector"
OPEN_AI_MODEL = "gpt-3.5-turbo"
Supported_Orgs = ['Amazon','Apple','CitiGroup', 'FedEx', 'Ford','Google', 'Microsoft', 'Pepsi', 'Tesla','Walmart']
######################

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_supported_orgs_names():
    """
    Provide the supported organizations list
    :return: String of comma seperated values.
    """
    supported_orgs = [f.split("_")[0] for f in os.listdir(VECTOR_FOLDER)]
    return ", ".join(supported_orgs)

def get_create_context_from_chat(chat_data):
    """
    Provide the chat history with last 1000 characters
    :chat_data: Input list of streamlit chat sessions
    :return: String of 1000 character chat history.
    """
    chat_text ="chat history : "
    for chat in chat_data:
        chat_text += chat["role"] + ":" + chat["content"] + " "
    if len(chat_text) > 1000: #Set only 1000 characters
        chat_text = chat_text[-1000:]
    return chat_text


def split_questions_to_multiple(query):
    """
    Split a question into multiple sub-questions
    :query: Input question
    :return: String of multiple questions
    """
    global client
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a question, your task is to split it into multiple sub-questions. The answer should be comma-separated text"
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.5,
        max_tokens=512,
        top_p=0.5
    )

    return response.choices[0].message.content.replace("\n",",")

def fix_sentence_grammer_and_spelling(query):
    """
    Fix the grammar and spelling of input query
    :query: Input question
    :return: Gramattically fixed string.
    """
    global client
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statements, and your task is to convert them to standard English.Do not give me any information about procedures and service features that are not mentioned in the PROVIDED CONTEXT."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.2,
        max_tokens=512,
        top_p=0.2
    )
    return response.choices[0].message.content


def create_folder(folder_name):
    """
    Create a folder
    :folder_name: Folder path to create
    """
    os.makedirs(folder_name, exist_ok=True)

def get_org_names(query):
    """
    Get the organizations name in a string
    :query: Input question string
    :return: List of organizations
    """
    global client
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statement, perform NER and find business organization names present, return as comma seperated values, return None if no organizations found. " },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.2,
        max_tokens=128,
        top_p=0.2
    )
    found_orgs = response.choices[0].message.content.strip()
    if found_orgs == "None":
        return None
    else:
        found_org_name=[]
        orgs_splitted = found_orgs.split(",")
        for org_chk in orgs_splitted:
            found_match =False
            for original_org in Supported_Orgs:
                if original_org.upper() in org_chk.replace(" ","").upper():
                    found_org_name.append(original_org)
                    found_match = True
                    break
            if not found_match:
                found_org_name.append(org_chk)

        if len(found_org_name) > 0:
            return list(set(found_org_name))
        else:
            return None


        return [f.strip().replace(" ","") for f in found_orgs.split(",")]



def pdf_not_processed():
    """
    Get the list of pdf's that are not converted to vector.
    :return: List of pdf that need to be converted.
    """
    pdf_files = [f.replace(".pdf","") for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    vector_folder = [f for f in os.listdir(VECTOR_FOLDER)]
    remaining_pdf = [f+".pdf" for f in list(set(pdf_files) - set(vector_folder))]
    return remaining_pdf


def get_vector_map(organization):
    """
    Get the vector map of an organization.
    :organization: Organization name whose vector map is needed
    :return: Vector of the organization.
    """
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vector_folders = [f for f in os.listdir(VECTOR_FOLDER)]
    for fldr in vector_folders:
        if organization.upper() in fldr.split("_")[0].upper():
            current_vector = FAISS.load_local(os.path.join(VECTOR_FOLDER,fldr), embeddings,allow_dangerous_deserialization=True)
            return current_vector
    return None


def pdf_reader_vectorize():
    """
    Convert a pdf to vector embedding and save the vector. Llama parser is used to extract text.
    Use FAISS create vector db using OPENAI embedding data.
    """
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    pdf_files = pdf_not_processed() #[f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    parser = LlamaParse(api_key=st.secrets["LLAMA_API_KEY"],result_type="markdown")
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER,pdf)
        markdown_text = parser.load_data(pdf_path)[0].text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)

        texts = text_splitter.split_text(markdown_text)
        markdown_text
        document_vector = FAISS.from_texts(texts, embeddings)

        document_vector_path = os.path.join(VECTOR_FOLDER,pdf.replace(".pdf",""))
        document_vector.save_local(document_vector_path)


def chat_bot_start_process():
    """
    Start the streamlit chat. Get query from user and output the answer.
    """
    global client
    st.title("Financial Report Analyser")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = OPEN_AI_MODEL

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_used_orgs" not in st.session_state:
        st.session_state["last_used_orgs"] = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        good_query = str(fix_sentence_grammer_and_spelling(prompt))
        st.session_state.messages.append({"role": "user", "content": prompt})
        orgs_found = get_org_names(good_query)
        current_vector = []
        current_org=[]
        answer=None
        if orgs_found is not None:
            for org in orgs_found:
                if len(org)>20:
                    continue
                loaded_vector = get_vector_map(org)
                if loaded_vector is None:
                    answer = "The details of org " + org + " is not found. I can provide the details of following orgs : " + str(get_supported_orgs_names())
                else:
                    current_vector.append(loaded_vector)
                    current_org.append(org)

            if len(current_vector)>0:
                st.session_state["last_used_orgs"] = orgs_found

        elif st.session_state["last_used_orgs"] is not None:
            for org in st.session_state["last_used_orgs"]:
                loaded_vector = get_vector_map(org)
                current_vector.append(loaded_vector)
                current_org.append(org)
        else:
            answer = "Hi, please provide the  question and name of Organization for getting the details. I can provide the details of following orgs : " + str(get_supported_orgs_names())
            st.session_state["last_used_orgs"] = None

        if len(current_vector) > 0:
            prompt = PromptTemplate.from_template(
                """
                Use the following pieces of context to answer the question at the end. If you 
                don't know the answer, just say that you don't know, don't try to make up an 
                answer. The answer should be formatted correctly.

                {context}

                Question: {question}
                Helpful Answer:
                """
            )
            chain =  load_qa_chain(ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"],model=OPEN_AI_MODEL,temperature=0), chain_type="stuff", prompt=prompt) #, streaming=True
            docs = []
            # if len(current_vector)>1:
            #     #adjusted_query = split_questions_to_multiple(good_query)
            #     adjusted_query = good_query
            # else:
            #     adjusted_query = good_query
            adjusted_query = good_query

            for pos,selected_vector in enumerate(current_vector):
                current_docs=[]
                if adjusted_query is not None and selected_vector is not None:
                    current_docs = selected_vector.similarity_search(adjusted_query) #selected_vector.similarity_search_with_score(adjusted_query)#
                if len(current_vector)>1:
                    if len(current_docs) > 2: #If multiple org details is needed.
                        for doc_nm in range(0, len(current_docs)):
                            current_docs[doc_nm].page_content = "Details of " + current_org[pos] + ":" + current_docs[doc_nm].page_content
                        docs.extend(current_docs)
                    else:
                        docs.extend(current_docs)
                else:
                    docs.extend(current_docs)
            if len(docs)>0:
                docs.extend([Document(page_content=get_create_context_from_chat(st.session_state.messages))])
                answer = chain.run(input_documents=docs, question=adjusted_query)
                #answer = fix_sentence_grammer_and_spelling(answer)
                answer =  answer.replace("$","\\$")
            else:
                answer = "Sorry, Cannot find the answer."

        with st.chat_message("assistant"):
            if answer is None:
                answer= "Can you please rephrase the question? I can provide the details of following orgs : " + str(get_supported_orgs_names())
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)



if __name__ == "__main__":
    create_folder(VECTOR_FOLDER)

    pdf_number = len([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])
    vector_number =len([f for f in os.listdir(VECTOR_FOLDER)])

    if pdf_number!= vector_number:
        print("Start vectorizing")
        pdf_reader_vectorize()
        
    chat_bot_start_process()
