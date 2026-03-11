import psycopg2
from config import POSTGRES_CONFIG, TOP_K


def get_pg_connection():
    return psycopg2.connect(**POSTGRES_CONFIG)


def search(question: str, embed_model, top_k: int = TOP_K) -> tuple[str, list[str]]:
    # Embed the question (e5 models expect "query: " prefix)
    query_vector = embed_model.encode(f"query: {question}", normalize_embeddings=True).tolist()

    conn = get_pg_connection()
    try:
        cursor = conn.cursor()

        # Step 1: find top chunks, deduplicate by article_id
        cursor.execute("""
            SELECT article_id, article_url, score
            FROM (
                SELECT DISTINCT ON (article_id)
                    article_id,
                    article_url,
                    1 - (embedding <=> %s::vector) AS score
                FROM news_chunks
                ORDER BY article_id, embedding <=> %s::vector
            ) ranked
            ORDER BY score DESC
            LIMIT %s
        """, (query_vector, query_vector, top_k))
        top_articles = cursor.fetchall()

        if not top_articles:
            return "", []

        article_ids = [row[0] for row in top_articles]
        urls = {row[0]: row[1] for row in top_articles}

        # Step 2: fetch full article text from news_articles
        cursor.execute("""
            SELECT article_id, title, full_text
            FROM news_articles
            WHERE article_id = ANY(%s)
        """, (article_ids,))
        articles = cursor.fetchall()

    finally:
        conn.close()

    if not articles:
        return "", []

    # Format results for the prompt
    parts = []
    source_urls = []
    for article_id, title, full_text in articles:
        url = urls.get(article_id, "")
        parts.append(f"[Kaynak: {url}]\n{title}\n\n{full_text}")
        if url:
            source_urls.append(url)

    return "\n\n---\n\n".join(parts), source_urls