# ✒️ IqbalGPT: A Multilingual RAG Chatbot on Iqbal's Poetry

**IqbalGPT** is a sophisticated, multilingual Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate and contextually relevant answers about the poetry and philosophy of the great poet-philosopher, Allama Iqbal. Users can interact with the chatbot in both **English** and **Roman Urdu**.

This project was built to overcome the limitations of standard Large Language Models (LLMs), which can often "hallucinate" or provide factually incorrect information. By grounding the model's responses in a custom knowledge base, IqbalGPT ensures that its answers are derived directly from Iqbal's actual work.

---

## 🚀 Live Demo

*(It is highly recommended to replace this image with a GIF of your running application. You can use a free tool like LICEcap or ScreenToGif to record your screen.)*

![Screenshot of IqbalGPT Application](https://placehold.co/800x450/0f172a/e2e8f0?text=IqbalGPT%20Demo)

---

## ⚙️ How It Works: The RAG Architecture

This chatbot is built on a modern AI architecture known as **Retrieval-Augmented Generation (RAG)**. Instead of relying solely on the LLM's pre-trained knowledge, it first retrieves relevant information from a specialized database and then uses that information to generate a precise answer.

The entire process is orchestrated using the **LangChain** framework. Here’s a step-by-step breakdown:

### 1. Data Ingestion & Processing
* The process begins with a large, unstructured text file (`iqbal.txt`) containing Iqbal's poetry in the Urdu script.
* A Python script (`auto_process.py`) reads this file, automatically **transliterates** the Urdu script into Roman Urdu, and **translates** it into English using NLP libraries.
* This processed data is then structured and saved into a clean knowledge base file (`iqbal_knowledge_base.txt`). This automated pipeline ensures that a large corpus of text can be processed efficiently.

### 2. Embedding & Vector Storage
* When the application starts, it loads the structured text from `iqbal_knowledge_base.txt`.
* The text is split into smaller, manageable chunks.
* A powerful multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) is used to convert each chunk into a numerical representation called an **embedding**. These embeddings capture the semantic meaning of the text.
* All these embeddings are then stored in a highly efficient, local vector database using **FAISS** (Facebook AI Similarity Search). This database allows for incredibly fast similarity searches.

### 3. Retrieval
* When a user asks a question (e.g., "What is the concept of Khudi?"), the question itself is first converted into an embedding using the same sentence-transformer model.
* The application then searches the FAISS vector database to find the text chunks whose embeddings are most semantically similar to the question's embedding.
* These top-matching chunks are retrieved as the "context" for the answer.

### 4. Generation
* Finally, the retrieved context and the user's original question are passed to a powerful Large Language Model (`llama3-8b-8192`) via the fast **Groq API**.
* The LLM is given a specific prompt, instructing it to formulate an answer **based only on the provided context**.
* This final, contextually-grounded answer is then streamed back to the user in the chat interface.

This architecture ensures that the chatbot's responses are not just creative but also accurate and faithful to the source material.

---

## 🛠️ Technology Stack

* **Core Framework:** LangChain
* **LLM Provider:** Groq API (`llama3-8b-8192`)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit
* **Data Processing:** Python, `urdu-roman`, `deep-translator`

---

## 📂 Project Structure

Here is an overview of the project's file structure:

```
Urdu-English-RAG-Chatbot/
│
├── venv/                  # Virtual environment directory
├── .env                   # Stores the API key (not committed to Git)
├── .gitignore             # Specifies files to be ignored by Git
├── app.py                 # The main Streamlit application script
├── auto_process.py        # Script to automate data processing
├── iqbal.txt              # The original unstructured Urdu poetry data
├── iqbal_knowledge_base.txt # The processed and structured knowledge base
├── README.md              # Project documentation (this file)
└── requirements.txt       # List of Python dependencies
```

---

## 💻 Setup and Local Usage

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Urdu-English-RAG-Chatbot.git
cd Urdu-English-RAG-Chatbot
```

### 2. Create and Activate a Virtual Environment
It is crucial to use a virtual environment to avoid dependency conflicts.
```bash
# Create the environment
python -m venv venv

# Activate the environment (on Windows Git Bash)
source venv/Scripts/activate
```

### 3. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
* Get a free API key from [groq.com](https://groq.com/).
* Create a file named `.env` in the root of your project folder.
* Add your API key to the `.env` file like this:
    ```
    GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

### 5. Process the Data
Run the automated data processing script **once** to create the knowledge base.
```bash
python auto_process.py
```
This will read `iqbal.txt` and generate `iqbal_knowledge_base.txt`.

### 6. Run the Application
Launch the Streamlit app.
```bash
streamlit run app.py
```
The application will open in your web browser, and you can start chatting!

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the chatbot, feel free to open an issue or submit a pull request.

1.  **Fork the repository** on GitHub.
2.  **Create a new branch:** `git checkout -b feature/your-feature-name`
3.  **Make your changes** and commit them: `git commit -m 'Add some feature'`
4.  **Push to the branch:** `git push origin feature/your-feature-name`
5.  **Open a Pull Request.**

---

## 🙏 Credits & Acknowledgements

* **Project Lead & Developer:** Danish Mir
* **Core Technologies:** This project was made possible by the incredible open-source work from the teams behind [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Hugging Face](https://huggingface.co/).
* **LLM Access:** The generative AI capabilities are powered by the high-speed [Groq API](https://groq.com/).
* **Data Source:** The initial poetry data was sourced from the [Allama Iqbal Poetry Dataset](https://www.kaggle.com/datasets/hassaanali/allama-iqbal-poetry) on Kaggle.
