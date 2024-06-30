# Financial_Q-A_Chatbot #

## About ##
Q&A Chatbot capable of answering questions related to the financial reports of large public companies. This chatbot can be directly hosted in Heroku platform.
The RAG (Retrieval-Augmented Generation) approach is used. Data source (PDF) is extracted, processed and stored as a vector store. The required data for answering the question is retrieved from the vector store.The retrieved data is passed as context to LLM from which the answer is found, if present.

## Data Used ##
- 10-K Report, a comprehensive document filed annually by public companies to detail their financial performance for year 2023.
- Currently contains data of 10 orgs - 'Amazon','Apple','CitiGroup', 'FedEx', 'Ford','Google', 'Microsoft', 'Pepsi', 'Tesla' and 'Walmart' in vector store (vector folder) parsed by Llama parser. So no need to parse again and is ready to use.
- Can add more pdf to data folder, which will be converted to vector db using Llama parser. Need to add the organization name to Supported_Orgs list in main.py.

## Tools Used ##
- StreamLite : Used to create the chat web UI.
- LlamaParse :  For extracting text from PDFs.
- Open AI : For the LLM model and embeddings
- FAISS(Facebook AI Similarity Search) : For creating vector stores and checking similarity.
- LangChain - For various extraction, retrieval, and generative utilities.

## Steps To Run ##
- Open Anaconda prompt
- Install the modules in requirements.txt
- Add api keys in .streamlit/secrets.toml
- Start the Streamlit app by running 'Streamlit run main.py'

## Examples ##
![Alt text](example/ex_1.png?raw=true "ex_1")
![Alt text](example/ex_2.png?raw=true "ex_2")
