import logging
import psycopg2
from config import POSTGRES_CONFIG, TOP_K

logger = logging.getLogger(__name__)


def get_pg_connection():
    return psycopg2.connect(**POSTGRES_CONFIG)


def search(question: str, embed_model, top_k: int = TOP_K, intent: str = "vector"):
    logger.info(f"[SEARCH] intent={intent} query={question!r}")
    conn = get_pg_connection()
    try:
        cursor = conn.cursor()

        # ¦¦ gundem: most important + most recent, no query needed ¦¦
        if intent == "gundem":
            cursor.execute("""
                SELECT article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                FROM (
                    SELECT DISTINCT ON (title)
                        article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                    FROM news_articles
                    WHERE ilk_cekilme_tarihi >= NOW() - INTERVAL '24 hours'
                    ORDER BY title, onem_rank DESC NULLS LAST
                ) deduped
                ORDER BY onem_rank DESC NULLS LAST, ilk_cekilme_tarihi DESC
                LIMIT %s
            """, (top_k,))
            rows = cursor.fetchall()

            if not rows:
                cursor.execute("""
                    SELECT article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                    FROM (
                        SELECT DISTINCT ON (title)
                            article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                        FROM news_articles
                        WHERE ilk_cekilme_tarihi >= NOW() - INTERVAL '48 hours'
                        ORDER BY title, onem_rank DESC NULLS LAST
                    ) deduped
                    ORDER BY onem_rank DESC NULLS LAST, ilk_cekilme_tarihi DESC
                    LIMIT %s
                """, (top_k,))
                rows = cursor.fetchall()

            if not rows:
                logger.info("[SEARCH] gundem: no results")
                return "", []

            parts, source_urls = [], []
            for article_id, url, title, full_text, onem_rank, date in rows:
                logger.info(f"[SEARCH] gundem: id={article_id} onem_rank={onem_rank} date={date} title={title!r}")
                parts.append(f"[Kaynak: {url}]\n{title}\n\n{full_text}")
                if url:
                    source_urls.append(url)
            return "\n\n---\n\n".join(parts), source_urls

        # ¦¦ vector / followup: pure semantic search ¦¦
        query_vector = embed_model.encode(f"query: {question}", normalize_embeddings=True).tolist()

        cursor.execute("""
            SELECT article_id, article_url, score
            FROM (
                SELECT DISTINCT ON (article_id)
                    article_id, article_url,
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

    parts, source_urls = [], []
    for article_id, title, full_text in articles:
        url = urls.get(article_id, "")
        logger.info(f"[SEARCH] vector: id={article_id} url={url} title={title!r}")
        parts.append(f"[Kaynak: {url}]\n{title}\n\n{full_text}")
        if url:
            source_urls.append(url)
    return "\n\n---\n\n".join(parts), source_urls