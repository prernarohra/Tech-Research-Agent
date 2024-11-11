# Tech-Research-Agent 🤖

Tech Research Agent is an AI-Based research assistant built with OpenAI's API, SERPAPI for web search, and Pinecone for vector database storage. It helps users quickly gather and analyze the latest tech news, research papers, and industry insights by providing fast, accurate information according to their needs.

## ✨ Features

- Automated Tech Research: Gathers and processes the latest tech news, academic research, and industry updates.
- Natural Language Understanding: Leverages OpenAI’s API to interpret user queries and deliver accurate responses.
- Vector Database: Uses Pinecone for efficient storage and retrieval of information based on semantic similarity.
- Web Search Integration: Integrates SERPAPI to pull up-to-date information directly from the web.
- Hugging Face Dataset: Utilizes a tech news dataset from Hugging Face for added contextual information.

## ⚙️ Tools and libraries

- Python
- VS code
- OpenAI API Key
- SerpAPI Key
- Pinecone Vector Database
- LangChain
- LangGraph
- HuggingFace Datasets

## 🔧 Installation

1. Clone this repository:

```bash
# Clone the repository
git clone https://github.com/prernarohra/Tech-Research-Agent.git

# Navigate to the project directory
cd Tech-Research-Agent
```

2. Create a .env file in the root directory to store your API keys securely:

```bash
touch .env
```

3. In the .env file, add the required API keys and database configurations:

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
SERPAPI_KEY=YOUR_SERPAPI_KEY
PINECONE_API_KEY=YOUR_PINECONE_API_KEY
```

4. Run the file:

```bash
python Tech_Agent.py
```

## 🚀 Usage

1. API Key Replacement:
- Replace YOUR_OPENAI_API_KEY with your OpenAI API key.
- Replace YOUR_SERPAPI_KEY with your SERPAPI key.
- Replace YOUR_PINECONE_API_KEY with your Pinecone API key.
  
2. Run the .py file: After setting up the environment, the program will be accessible.

3. Query for Information: Use the platform to ask questions or seek updates on tech news. The Hugging Face dataset enriches results by providing additional context from stored tech news articles.

## 🙌 Contributing

Contributions are welcome! Please fork this repository, create a feature branch, and submit a pull request.

## 🎀 License

This project is licensed under the MIT License.
