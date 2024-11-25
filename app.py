import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere  # Replace with ChatOpenAI if needed

# Load environment variables from .env
find_dotenv()
load_dotenv()

# Initialize directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "about_data")
persistent_directory = os.path.join(data_dir, "chroma_about_data_data")

# Step 1: Initialize Embeddings and Vector Store
if not os.path.exists(persistent_directory):
    print("Initializing vector store as directory does not exist.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist. Check the setup.")
    
    firecrawl_api = os.getenv("FireCrawlLoader_api")
    url = ["https://www.apple.com/"]  # Replace with your target URL
    loader = FireCrawlLoader(url, api_key=firecrawl_api, mode="crawl")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 2: Initialize Language Model
llm = ChatCohere(model="command-r")  # Replace with ChatOpenAI(model="gpt-4o") if needed

# Step 3: Job Fetching Logic
def fetch_and_extract_jobs():
    """
    Fetch job openings from a predefined webpage.
    """
    url = "https://www.workcohol.com/page-career"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator="\n")
        
        prompt = PromptTemplate(
            input_variables=["web_content"],
            template=(
                "Extract job openings from the content below. If no job openings are found, "
                "respond with 'No job openings found.'\n\nContent:\n{web_content}\n\nJob Openings:"
            )
        )
        chain = LLMChain(llm=ChatOpenAI(model="gpt-4o"), prompt=prompt)
        result = chain.run({"web_content": text_content})
        return [line.strip() for line in result.split("\n") if line.strip()]
    except requests.exceptions.RequestException as e:
        return [f"Error fetching webpage: {e}"]

# Step 4: Detect User Intent
def detect_intent(user_query):
    """
    Detects if a query is asking about job openings.
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Check if this query is about job openings. Respond 'yes' for job-related queries and 'no' otherwise:\n{query}"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"query": user_query}).strip().lower()
    return result == "yes"

# Step 5: Build Question Answering System
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Reformulate user questions to make them independent of context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer user questions about the company's services, mission, or products."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Step 6: Tools and Agent
tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Use to answer company-related questions."
    ),
    Tool(
        name="Fetch Job Openings",
        func=lambda input: fetch_and_extract_jobs(),
        description="Fetch and extract job openings from the company website."
    ),
]

agent = create_react_agent(llm=llm, tools=tools, prompt=hub.pull("hwchase17/react"))
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Step 7: Chat Loop
def main():
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if detect_intent(user_input):
            # Fetch job openings
            jobs = tools[1].func(user_input)
            print("\n--- Job Openings ---")
            for job in jobs:
                print(job)
        else:
            # Handle general queries
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print(f"AI: {response['output']}")
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=response["output"])])

if __name__ == "__main__":
    main()
