"# RAG_text_summerization" 

This project implements a **Retrieval-Augmented Generation (RAG)** system for summarizing and retrieving documents. It uses **ChromaDB** for document storage and retrieval, **Hugging Face embeddings** for semantic search, and the **meta-llama/Llama-3.3-70B-Instruct** model for summarization.

## Features

- **Document Loading**: Supports loading PDFs, Excel files, and CSV files.
- **Document Splitting**: Splits documents into smaller chunks for efficient processing.
- **Semantic Search**: Retrieves documents relevant to a query using ChromaDB and Hugging Face embeddings.
- **Summarization**: Summarizes retrieved documents using the `meta-llama/Llama-3.3-70B-Instruct` model.
- **Duplicate Removal**: Removes duplicate documents before summarization.
- **Similarity-Based Combination**: Combines similar documents to avoid repetitive summaries.

## Prerequisites

Before running the code, ensure you have the following installed:

1. **Python 3.8 or higher**
2. **Required Python Libraries**:
   - `langchain`
   - `langchain-community`
   - `huggingface-hub`
   - `pandas`
   - `pypdf`
   - `scikit-learn`
   - `python-dotenv`

You can install the required libraries using the following command:

```bash
pip install langchain langchain-community huggingface-hub pandas pypdf scikit-learn python-dotenv
```

3. **Hugging Face API Key**:
   - Obtain your Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens).
   - Add the API key to a `.env` file in the project directory:
     ```plaintext
     HUGGINGFACE_API_KEY=your_api_key_here
     ```

4. **Llama-3.3-70B-Instruct Model**:
   - Ensure you have access to the `meta-llama/Llama-3.3-70B-Instruct` model on Hugging Face.
   - Accept the model's terms of use on the [model page](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

## Project Structure

- **`sample_data/pdfs/`**: Folder containing PDF documents.
- **`sample_data/excels/`**: Folder containing Excel and CSV files.
- **`chroma_db/`**: Directory where ChromaDB stores document embeddings.
- **`.env`**: File containing the Hugging Face API key.
- **`main.py`**: Main script for loading, processing, and summarizing documents.

## How It Works

1. **Document Loading**:
   - The script loads PDFs, Excel files, and CSV files from the `sample_data` folder.
   - PDFs are loaded using `PyPDFLoader`, and Excel/CSV files are loaded using `pandas`.

2. **Document Splitting**:
   - Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.

3. **ChromaDB Initialization**:
   - Document embeddings are generated using Hugging Face's `all-MiniLM-L6-v2` model.
   - The embeddings are stored in ChromaDB for efficient retrieval.

4. **Querying and Retrieval**:
   - The script retrieves documents relevant to a query using ChromaDB's semantic search.

5. **Summarization**:
   - The retrieved documents are summarized using the `meta-llama/Llama-3.3-70B-Instruct` model.

6. **Duplicate Removal and Similarity-Based Combination**:
   - Duplicate documents are removed, and similar documents are combined to avoid repetitive summaries.

## Usage

1. **Set Up the Environment**:
   - Add your Hugging Face API key to the `.env` file.
   - Place your documents in the `sample_data/pdfs` and `sample_data/excels` folders.

2. **Run the Script**:
   Execute the script using the following command:
   ```bash
   python main.py
   ```

3. **Example Queries**:
   - The script will run predefined queries (e.g., `"GDP Growth"`) and print the summaries.

## Example Output

```
Retrieved 3 documents for query: 'GDP Growth'
After removing duplicates, 1 unique documents remain.
After combining similar documents, 1 documents remain.
Summary for query 'GDP Growth': "GDP growth is influenced by various factors, including external deficits and surpluses, which can signal economic imbalances. Emerging economies often run temporary deficits to access advanced technologies and improve competitiveness."
```

## Customization

- **Change the LLM**:
  To use a different model, update the `repo_id` in the `summarize_document` function:
  ```python
  llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.3-70B-Instruct", huggingfacehub_api_token=huggingface_api_key)
  ```

- **Adjust Chunk Size and Overlap**:
  Modify the `chunk_size` and `chunk_overlap` parameters in the `split_documents` function:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
  ```

- **Add More Queries**:
  Add additional queries to the `main.py` script:
  ```python
  query2 = "Economic Indicators"
  summary2 = summarize_relevant_documents(vector_store, query2, top_k=3)
  print(f"Summary for query '{query2}':", summary2)
  ```

## Troubleshooting

- **Hugging Face API Errors**:
  - Ensure your API key is valid and has access to the `meta-llama/Llama-3.3-70B-Instruct` model.
  - Check the Hugging Face API status at [Hugging Face Status](https://status.huggingface.co/).

- **Duplicate Content**:
  - If summaries are repetitive, adjust the `similarity_threshold` in the `combine_similar_documents` function.

- **PDF Parsing Warnings**:
  - Warnings like `Multiple definitions in dictionary` are from the `pypdf` library and can be ignored unless they affect the document content.


This `README.md` provides a comprehensive guide to your project. Let me know if you need further customization!

