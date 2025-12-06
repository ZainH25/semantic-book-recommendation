# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import pandas as pd
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

import gradio as gr


# ---------------------------------------------------------
# LOAD BOOK DATA
# ---------------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

# Create large thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)


# ---------------------------------------------------------
# BUILD DOCUMENT LIST (ONE PER BOOK → VECTOR DB INPUT)
# ---------------------------------------------------------
documents = []

for _, row in books.iterrows():
    documents.append(
        Document(
            page_content=row["description"],       # text we embed
            metadata={"isbn13": row["isbn13"]}     # used for lookup later
        )
    )


# ---------------------------------------------------------
# INITIALIZE EMBEDDINGS + CHROMA DB
# ---------------------------------------------------------
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------
def retrive_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    # 1 — SEARCH TOP MATCHING BOOK DESCRIPTIONS
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract ISBN order
    isbn_list = [rec.metadata["isbn13"] for rec in recs]

    # 2 — FILTER THESE BOOKS IN THE ORIGINAL DATAFRAME
    book_recs = books[books["isbn13"].isin(isbn_list)]

    # Keep same similarity order
    book_recs = (
        book_recs.set_index("isbn13")
        .loc[isbn_list]
        .reset_index()
    )

    # 3 — Filter by Category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # 4 — Sort by Emotion Tone
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)


# ---------------------------------------------------------
# FORMAT OUTPUT FOR GRADIO
# ---------------------------------------------------------
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrive_semantic_recommendations(query, category, tone)
    result = []

    for _, row in recommendations.iterrows():

        # Short description preview
        desc_words = row["description"].split()
        truncated_description = " ".join(desc_words[:30]) + " ..."

        # Fix author formatting
        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            author_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            author_str = f"{', '.join(author_split[:-1])} and {author_split[-1]}"
        else:
            author_str = row["authors"]

        caption = f"{row['title']} by {author_str}: {truncated_description}"

        # Append (IMAGE, TEXT)
        result.append((row["large_thumbnail"], caption))

    return result


# ---------------------------------------------------------
# BUILD GRADIO DASHBOARD
# ---------------------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# **Semantic Book Recommender**")

    with gr.Row():
        user_query = gr.Textbox(
            label="Enter a short description of the book you want:",
            placeholder="Eg: A magical adventure about friendship"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Category",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Tone",
            value="All"
        )

        submit_button = gr.Button("Find Recommended Books")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(columns=4, rows=2, label="Results")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


# ---------------------------------------------------------
# LAUNCH APP
# ---------------------------------------------------------
if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)


