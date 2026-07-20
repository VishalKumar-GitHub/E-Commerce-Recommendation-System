"""Content-based recommendation engine using TF-IDF + cosine similarity.

Pure scikit-learn, no compiled dependencies, so it deploys cleanly on
Streamlit Cloud. The heavy objects (TF-IDF matrix, similarity lookups) are
built once and cached by the app layer.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self._matrix = self._vectorizer.fit_transform(self.df["text"])
        # map ProductId -> row index for O(1) lookups
        self._id_to_idx = {pid: i for i, pid in enumerate(self.df["ProductId"])}

    @property
    def product_ids(self) -> list[str]:
        return self.df["ProductId"].tolist()

    def get_product(self, product_id: str) -> pd.Series:
        return self.df.iloc[self._id_to_idx[product_id]]

    def recommend_by_product(
        self, product_id: str, top_n: int = 8, same_subcategory: bool = False
    ) -> pd.DataFrame:
        """Return the top_n most similar products to the given product."""
        if product_id not in self._id_to_idx:
            return self.df.iloc[0:0]
        idx = self._id_to_idx[product_id]
        sims = cosine_similarity(self._matrix[idx], self._matrix).ravel()
        order = sims.argsort()[::-1]
        target = self.df.iloc[idx]
        results = []
        for i in order:
            if i == idx:
                continue  # skip the product itself
            if same_subcategory and \
                    self.df.iloc[i]["sub_category"] != target["sub_category"]:
                continue
            row = self.df.iloc[i].copy()
            row["similarity"] = round(float(sims[i]), 3)
            results.append(row)
            if len(results) >= top_n:
                break
        return pd.DataFrame(results)

    def recommend_by_text(self, query: str, top_n: int = 8) -> pd.DataFrame:
        """Free-text search: rank products by similarity to a query string."""
        if not query.strip():
            return self.df.iloc[0:0]
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).ravel()
        order = sims.argsort()[::-1][:top_n]
        results = []
        for i in order:
            if sims[i] <= 0:
                continue
            row = self.df.iloc[i].copy()
            row["similarity"] = round(float(sims[i]), 3)
            results.append(row)
        return pd.DataFrame(results)
