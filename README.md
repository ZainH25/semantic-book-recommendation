📚 Semantic Book Recommendation System
AI-powered book recommender using embeddings, emotions, and semantic search.

🔗 Live Demo on HuggingFace Spaces:
👉 https://huggingface.co/spaces/zainh25/semantic-book-recommender

🚀 Overview

This project is an AI-based Semantic Book Recommendation System that recommends books based on:

User-entered natural language descriptions

Semantic similarity using vector embeddings

Emotion scores extracted from book descriptions

Category classification (Fiction, Nonfiction, Children’s Fiction)

It uses Large Language Model (LLM)-based embeddings, vector search, and sentiment classification to generate high-quality recommendations.

The final application is deployed on HuggingFace Spaces using Gradio UI.

🧠 How It Works (Architecture)
1️⃣ Data Cleaning & Preprocessing

Removed books with missing descriptions

Combined title + subtitle

Required descriptions to have at least 25 words

Created books_cleaned.csv and later enriched it with:

Category predictions

Emotion scores

Cleaned metadata

2️⃣ Semantic Embedding Using LLMs

We use:

🔹 BAAI/bge-small-en-v1.5

A state-of-the-art embedding model optimized for:

Semantic similarity

Retrieval tasks

Low latency vector search

This is loaded via:

FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


Each book description is converted into a vector representation, enabling semantic understanding.

3️⃣ Vector Database (ChromaDB)

We store embeddings using:

🔹 ChromaDB — a fast, open-source vector store ideal for semantic search.

Each document stored contains:

page_content: book description

metadata: ISBN (used to map back to book details)

At inference time:

db_books.similarity_search(query, k=50)


This retrieves books with closest meaning, not just keywords.

4️⃣ Emotion Analysis (LLM-based Sentiment Model)

We compute emotional intensity (joy, fear, sadness, anger, surprise, etc.) per book using:

🔹 j-hartmann/emotion-english-distilroberta-base

This allows users to filter books based on emotional tone, e.g.,

Happy

Sad

Suspenseful

Angry

The final dataset becomes books_with_emotions.csv.

5️⃣ Zero-Shot Text Classification for Book Categories

We classify missing categories using:

🔹 facebook/bart-large-mnli (Zero-shot classification)

This predicts whether a description is:

Fiction

Nonfiction

Children’s Fiction

Result stored as simple_categories.

6️⃣ Gradio User Interface

The final recommender is built using Gradio Blocks UI.

Users can input:

✔ Book description
✔ Category filter
✔ Emotional tone

The system returns:

Book cover

Title + author

30-word summary

📦 Installation (Run Locally)
1️⃣ Clone the repository
git clone https://github.com/ZainH25/semantic-book-recommendation.git
cd semantic-book-recommendation

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
python app.py


The app will start at:

http://127.0.0.1:7860

📁 Project Structure
├── app.py                     # Main application (Gradio UI + recommendation logic)
├── books_cleaned.csv
├── books_with_categories.csv
├── books_with_emotions.csv
├── tagged_description.txt
├── requirements.txt
└── Notebooks/
      ├── data-exploration.ipynb
      ├── text-classification.ipynb
      ├── sentimental-analysis.ipynb
      └── vector-search.ipynb

🎯 Features
✔ Semantic search using embeddings
✔ Emotion-based filtering
✔ Category-aware recommendations
✔ Full LLM-powered pipeline
✔ Professional UI
✔ Deployed publicly on HuggingFace
🔗 Live App (HuggingFace Space)

👉 https://huggingface.co/spaces/zainh25/semantic-book-recommender


