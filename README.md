# PDF ChatBot

## Overview
PDF ChatBot is a Streamlit-based AI chatbot that allows users to upload PDF documents, process them, and ask questions about the content. The chatbot uses Google's Generative AI models to generate embeddings and perform similarity searches on the processed text, providing relevant answers.

## Features
- Upload multiple PDF files and extract text
- Split text into manageable chunks for efficient processing
- Store text embeddings using FAISS (Facebook AI Similarity Search)
- Perform semantic search on the processed PDFs
- Provide intelligent responses using Google's Gemini AI model
- Maintain a conversation history
- Clear conversation history when needed

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: For building the user interface
- **PyPDF2**: To extract text from PDF documents
- **LangChain**: For text processing and chatbot logic
- **FAISS**: To store and retrieve text embeddings efficiently
- **Google Generative AI**: For embeddings and chatbot responses
- **dotenv**: For managing API keys securely

## Installation
### Prerequisites
Make sure you have Python installed (preferably 3.8 or later). Also, ensure you have a Google API key for Generative AI.

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/shoaibmobassir/pdf-chatbot.git
   cd pdf-chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your Google API key:
   ```sh
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

## Usage
1. Run the application:
   ```sh
   streamlit run app.py
   ```
2. Upload one or more PDF files using the sidebar.
3. Click the "Process PDFs" button to extract and store text embeddings.
4. Start asking questions in the chat input field.
5. View the conversation history and clear it when needed.

## Folder Structure
```
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── .env                   # API Key storage (not included in repo)
├── README.md              # Project documentation
└── faiss_index/           # FAISS vector store (generated at runtime)
```

## Notes
- Ensure you have a stable internet connection for API requests.
- Processing large PDFs may take some time depending on system resources.

## License
This project is licensed under the MIT License.

## Author
Shoaib Mobassir

