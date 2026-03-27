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

        # ¦¦ gundem: importance + recency, no vector score ¦¦
        if intent == "gundem":
            cursor.execute("""
                SELECT article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                FROM (
                    SELECT DISTINCT ON (title)
                        article_id, article_url, title, full_text, onem_rank, ilk_cekilme_tarihi
                    FROM news_articles
                    WHERE ilk_cekilme_tarihi >= NOW() - INTERVAL '24 hours'
                ) deduped
                ORDER BY (
                    ((onem_rank - 2.0) / 14.0) * 0.6 +
                    (EXTRACT(EPOCH FROM ilk_cekilme_tarihi) / EXTRACT(EPOCH FROM NOW())) * 0.4
                ) DESC NULLS LAST
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
                    ORDER BY (
                        ((onem_rank - 2.0) / 14.0) * 0.6 +
                        (EXTRACT(EPOCH FROM ilk_cekilme_tarihi) / EXTRACT(EPOCH FROM NOW())) * 0.4
                    ) DESC NULLS LAST
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

        # ¦¦ vector / followup: relevance + importance + recency ¦¦
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
        scores = {row[0]: row[2] for row in top_articles}
        urls = {row[0]: row[1] for row in top_articles}

        cursor.execute("""
            SELECT article_id, title, full_text, onem_rank, ilk_cekilme_tarihi
            FROM news_articles
            WHERE article_id = ANY(%s)
        """, (article_ids,))
        articles = cursor.fetchall()

    finally:
        conn.close()

    if not articles:
        return "", []

    # Combined ranking: 60% vector score + 25% importance + 15% recency
    import time
    now_epoch = time.time()

    def combined_score(row):
        article_id, title, full_text, onem_rank, ilk_cekilme_tarihi = row
        vec = scores.get(article_id, 0)
        imp = ((onem_rank or 2) - 2.0) / 14.0
        rec = ilk_cekilme_tarihi.timestamp() / now_epoch if ilk_cekilme_tarihi else 0
        return vec * 0.6 + imp * 0.25 + rec * 0.15

    articles = sorted(articles, key=combined_score, reverse=True)

    parts, source_urls = [], []
    for article_id, title, full_text, onem_rank, ilk_cekilme_tarihi in articles:
        url = urls.get(article_id, "")
        final_score = combined_score((article_id, title, full_text, onem_rank, ilk_cekilme_tarihi))
        logger.info(f"[SEARCH] vector: id={article_id} score={final_score:.3f} onem={onem_rank} url={url} title={title!r}")
        parts.append(f"[Kaynak: {url}]\n{title}\n\n{full_text}")
        if url:
            source_urls.append(url)
    return "\n\n---\n\n".join(parts), source_urls