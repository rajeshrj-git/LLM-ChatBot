

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
from langchain_cohere import ChatCohere  # Change this to ChatOpenAI if needed


# Load environment variables
find_dotenv()
load_dotenv()

# Define persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "about_data")
persistent_directory = os.path.join(data_dir, "chroma_about_data_data")

# Initialize embeddings and vector store
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist. Please check the path.")
    
    firecrawl_api = os.getenv("FireCrawlLoader_api")
    url = ["https://www.apple.com/"]
    loader = FireCrawlLoader(url, api_key=firecrawl_api, mode="crawl")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize ChatOpenAI LLM
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatCohere(model="command-r")  # Replace with ChatOpenAI if needed

# Job Fetching Logic
def fetch_and_extract_jobs():
    """
    Fetches the webpage content from a predefined URL and extracts job openings.
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
                "Given the following webpage content, extract all the job openings. "
                "Job openings are typically titles or headings indicating positions available at a company. "
                "If no job openings are found, respond with 'No job openings found.'\n\n"
                "Webpage Content:\n{web_content}\n\n"
                "Extracted Job Openings:"
            )
        )
        chain = LLMChain(llm=ChatOpenAI(model="gpt-4o"), prompt=prompt)
        result = chain.run({"web_content": text_content})
        job_openings = [line.strip() for line in result.split("\n") if line.strip()]
        return job_openings if job_openings else ["No job openings found."]
    except requests.exceptions.RequestException as e:
        return [f"Error fetching webpage: {e}"]

# Intent Detection
def detect_intent(user_query):
    """
    Detects if the user query is about job openings.
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Determine if the following user query is asking about job openings or positions "
            "available at a company. Respond with 'yes' if it is related to job openings, "
            "and 'no' otherwise.\n\n"
            "User Query:\n{query}\n\n"
            "Is this about job openings? (yes or no):"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"query": user_query}).strip().lower()
    return result == "yes"

# Prompts for QA
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question about the company's details, "
    "such as its services, history, mission, or products, reformulate the question "
    "into a standalone version that can be understood without relying on the chat history. "
    "Do NOT provide an answer to the question, only rewrite or return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "Given a user question about the company's details (e.g., services, mission, products), "
    "analyze the context and provide a concise and accurate answer based solely on available information."
    "\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Tools
tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Use this tool to answer questions about the company's context.",
    ),
    Tool(
        name="Fetch and Extract Job Openings",
        func=lambda input: fetch_and_extract_jobs(),
        description="Extracts job openings from the company's webpage.",
    ),
]

# ReAct Agent
react_docstore_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_docstore_prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)

# Main Chat Loop
def main():
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        if detect_intent(query):
            # Use Job Fetching Tool
            response = tools[1].func(query)
            print("\n--- Job Openings ---")
            for job in response:
                print(job)
        else:
            # General Query Handling
            response = agent_executor.invoke({"input": query, "chat_history": chat_history})
            print(f"AI: {response['output']}")
        
        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response["output"]))

if __name__ == "__main__":
    main()

    