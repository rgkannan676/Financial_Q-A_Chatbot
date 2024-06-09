import os
from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
from langchain_openai import ChatOpenAI

PDF_FOLDER = "data"
VECTOR_FOLDER = "vector"
OPEN_AI_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

LAST_LOADED_ORG = None

def fix_sentence_grammer_and_spelling(query):
    global client
    print("quey : ", query)
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statements, and your task is to convert them to standard English."
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
                "content": "You will be provided with statement, find organization names present, return as comma seperated values, return None if no organizations found."
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
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        good_query = str(fix_sentence_grammer_and_spelling(prompt))
        orgs_found = get_org_names(good_query)
        print("Orgs found : ", orgs_found)
        current_vector = None
        answer=None
        if orgs_found is not None:
            for org in orgs_found:
                loaded_vector = get_vector_map(org)
                if loaded_vector is None:
                    answer = "The details of org " + org + " is not found."
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    LAST_LOADED_ORG = org
                else:
                    if current_vector is None:
                        current_vector = loaded_vector
                    else:
                        current_vector.merge_from(loaded_vector)

        elif LAST_LOADED_ORG is not None:
            current_vector = get_vector_map(org)
        else:
            answer = "Provide Organization name whose details is needed."
            st.session_state.messages.append({"role": "assistant", "content": answer})

        if current_vector is not None:
            chain = load_qa_chain(ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"],model=OPEN_AI_MODEL), chain_type="stuff")
            docs = current_vector.similarity_search(good_query)
            print("DOCS: ", docs)
            answer = chain.run(input_documents=docs, question=good_query)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer})

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
