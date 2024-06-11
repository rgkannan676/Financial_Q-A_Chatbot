import os
from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

PDF_FOLDER = "data"
VECTOR_FOLDER = "vector"
OPEN_AI_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

LAST_LOADED_ORG = None

def get_supported_orgs_names():
    supported_orgs = [f.split("_")[0] for f in os.listdir(VECTOR_FOLDER)]
    return ", ".join(supported_orgs)

def get_create_context_from_chat(chat_data):
    chat_text ="chat history : "
    for chat in chat_data:
        chat_text += chat["role"] + ":" + chat["content"] + " "
    if len(chat_text) > 1000: #Set only 1000 characters
        chat_text = chat_text[-1000:]
    print("chat_text : " , chat_text)
    return chat_text


def split_questions_to_multiple(query):
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
    global client
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statements, and your task is to convert them to standard English.Don't include any explanations in your responses"
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
    os.makedirs(folder_name, exist_ok=True)

def get_org_names(query):
    # global nlp
    # doc = nlp(query)
    # org_list = [x.text for x in doc.ents if x.label_ == "ORG"]
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)

    global client
    print("quey : ", query)
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statement, find business organization names present, return as comma seperated values, return None if no organizations found.Don't include any explanations in your responses."
            },
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
        return [f.strip() for f in found_orgs.split(",")]



def pdf_not_processed():
    pdf_files = [f.replace(".pdf","") for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    vector_folder = [f for f in os.listdir(VECTOR_FOLDER)]
    remaining_pdf = [f+".pdf" for f in list(set(pdf_files) - set(vector_folder))]
    return remaining_pdf


def get_vector_map(organization):
    print("Loading vector to Map")
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vector_folders = [f for f in os.listdir(VECTOR_FOLDER)]
    for fldr in vector_folders:
        if organization.upper() in fldr.split("_")[0].upper():
            current_vector = FAISS.load_local(os.path.join(VECTOR_FOLDER,fldr), embeddings,allow_dangerous_deserialization=True)
            print("Loaded vector for ", fldr.split("_")[0])
            return current_vector
    return None


def pdf_reader_vectorize():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    pdf_files = pdf_not_processed() #[f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    print(pdf_files)
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER,pdf)
        print("processing :", pdf_path)
        pdfreader = PdfReader(pdf_path)
        raw_text=""
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len)
        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len,
        # )
        texts = text_splitter.split_text(raw_text)
        document_vector = FAISS.from_texts(texts, embeddings)

        document_vector_path = os.path.join(VECTOR_FOLDER,pdf.replace(".pdf",""))
        document_vector.save_local(document_vector_path)

        print("Saved vector to :", document_vector_path)




def chat_bot_start_process():

    global client,LAST_LOADED_ORG
    
    st.title("Q&A chatbot on financial reports")

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
        st.session_state.messages.append({"role": "user", "content": good_query})
        orgs_found = get_org_names(good_query)
        print("Orgs found : ", orgs_found)
        current_vector = []
        current_org=[]
        answer=None
        if orgs_found is not None:
            for org in orgs_found:
                loaded_vector = get_vector_map(org)
                if loaded_vector is None:
                    answer = "The details of org " + org + " is not found. I can provide the details of following orgs : " + str(get_supported_orgs_names())
                    st.session_state.messages.append({"role": "assistant", "content": answer})
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
            st.session_state.messages.append({"role": "assistant", "content": answer})


        if len(current_vector) > 0:
            prompt = PromptTemplate.from_template(
                """
                Use the following pieces of context to answer the question at the end. If you 
                don't know the answer, just say that you don't know, don't try to make up an 
                answer.

                {context}

                Question: {question}
                Helpful Answer:
                """
            )
            chain =  load_qa_chain(ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"],model=OPEN_AI_MODEL,temperature=0), chain_type="stuff", prompt=prompt) #, streaming=True
            docs = []
            print("Good query = : ", good_query)
            if len(current_vector)>1:
                adjusted_query = split_questions_to_multiple(good_query)
            else:
                adjusted_query = good_query
            print("Updated query = ", adjusted_query)
            for pos,selected_vector in enumerate(current_vector):
                current_docs = selected_vector.similarity_search(adjusted_query) #selected_vector.similarity_search_with_score(adjusted_query)#
                for doc_nm in range(0,len(current_docs)):
                    current_docs[doc_nm].page_content = "Details of " + current_org[pos] + ":" + current_docs[0].page_content
                if len(current_vector)>1:
                    if len(current_docs) > 2:
                        docs.extend(current_docs[:2])
                    else:
                        docs.extend(current_docs)
                else:
                    docs.extend(current_docs)
                print("len: ", len(docs))
            if len(docs)>0:
                print("DOCS are ", docs)
                answer = chain.run(input_documents=docs, question=adjusted_query)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                docs.extend([Document(page_content=get_create_context_from_chat(st.session_state.messages))])
                print("##docs history length : ", len(docs))
            else:
                answer = "Sorry, Cannot find the answer."

        with st.chat_message("assistant"):
            st.markdown(answer)



        # with st.chat_message("assistant"):
        #     stream = client.chat.completions.create(
        #         model=st.session_state["openai_model"],
        #         messages=[
        #             {"role": m["role"], "content": m["content"]}
        #             for m in st.session_state.messages
        #         ],
        #         stream=True,
        #     )
        #     response = st.write_stream(stream)
        # st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    print("Start Processing")
    create_folder(VECTOR_FOLDER)

    pdf_number = len([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])
    vector_number =len([f for f in os.listdir(VECTOR_FOLDER)])
    if pdf_number!= vector_number:
        print("Start vectorizing")
        pdf_reader_vectorize()
    else:
        print("All documents already vectorized")

    chat_bot_start_process()
