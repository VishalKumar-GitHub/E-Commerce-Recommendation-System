"""Streamlit product recommendation app (content-based, TF-IDF).

Run locally:   streamlit run app.py
Deploy:        push to GitHub, connect the repo on share.streamlit.io
"""
from __future__ import annotations

import os
import urllib.parse

import pandas as pd
import streamlit as st

from recommender import Recommender

DATA_PATH = "data/products.csv"
LEGACY_DATA_PATH = "products.csv"

st.set_page_config(page_title="Product Recommender", page_icon="🛍️", layout="wide")

st.markdown(
    """
    <div style='padding:24px 32px; border-radius:24px; background:linear-gradient(135deg, #4b6cb7 0%, #182848 100%); color:white;'>
        <h1 style='margin:0; font-size:2.6rem;'>Vishal Kumar</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


def build_image_url(sub_cat: str, pid: str) -> str:
    label = urllib.parse.quote_plus(f"{sub_cat.title()} {pid}")
    return (
        f"https://placehold.co/400x300/png?text={label}"
        "&font=montserrat&bg=efefef&txt=333"
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = LEGACY_DATA_PATH if os.path.exists(LEGACY_DATA_PATH) else DATA_PATH
    if not os.path.exists(data_path):
        import generate_data
        os.makedirs("data", exist_ok=True)
        generate_data.build().to_csv(DATA_PATH, index=False)
        data_path = DATA_PATH

    df = pd.read_csv(data_path)
    if "image_url" not in df.columns:
        df["image_url"] = df.apply(
            lambda row: build_image_url(row["sub_category"], row["ProductId"]),
            axis=1,
        )
    return df


@st.cache_resource(show_spinner=False)
def get_recommender(df: pd.DataFrame) -> Recommender:
    return Recommender(df)


def stars(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def product_card(row: pd.Series, wide: bool = False):
    with st.container(border=True):
        if wide:
            if "image_url" in row and pd.notna(row.get("image_url")):
                st.image(row["image_url"], use_column_width=True)
            st.markdown(f"### {row['name']}")
            st.caption(f"{row['main_category']} › {row['sub_category']}")
            st.write(
                f"{stars(row['ratings'])}  {row['ratings']} "
                f"({int(row['no_of_ratings'])} ratings)"
            )
            st.markdown(f"**Price:** {format_price(row['Price'])}")
            if "similarity" in row and pd.notna(row.get("similarity")):
                st.caption(f"Match score: {row['similarity']}")
            st.write(f"_{row['Review']}_")
            st.caption(f"Product ID: {row['ProductId']}")
            return

        left, right = st.columns([1, 2])
        with left:
            if "image_url" in row and pd.notna(row.get("image_url")):
                st.image(row["image_url"], width=220)

        with right:
            st.markdown(f"### {row['name']}")
            st.caption(f"{row['main_category']} › {row['sub_category']}")
            st.write(
                f"{stars(row['ratings'])}  {row['ratings']} "
                f"({int(row['no_of_ratings'])} ratings)"
            )
            st.markdown(f"**{format_price(row['Price'])}**")
            if "similarity" in row and pd.notna(row.get("similarity")):
                st.caption(f"Match score: {row['similarity']}")
            st.write(f"_{row['Review']}_")
            st.caption(f"Product ID: {row['ProductId']}")


def show_grid(df: pd.DataFrame, cols: int = 4):
    if df.empty:
        st.info("No matching products found.")
        return
    records = df.to_dict("records")
    for i in range(0, len(records), cols):
        columns = st.columns(cols)
        for col, rec in zip(columns, records[i:i + cols]):
            with col:
                product_card(pd.Series(rec))


# ---------------------------------------------------------------- load
df = load_data()
rec = get_recommender(df)

st.title("🛍️ Product Recommendation System")
st.caption("Content-based recommendations powered by TF-IDF + cosine similarity.")

tab_browse, tab_similar, tab_search = st.tabs(
    ["Browse", "Find similar", "Search"]
)

# ---------------------------------------------------------------- browse
with tab_browse:
    c1, c2 = st.columns(2)
    with c1:
        main_cat = st.selectbox(
            "Main category", ["All"] + sorted(df["main_category"].unique())
        )
    subset = df if main_cat == "All" else df[df["main_category"] == main_cat]
    with c2:
        sub_cat = st.selectbox(
            "Sub category", ["All"] + sorted(subset["sub_category"].unique())
        )
    if sub_cat != "All":
        subset = subset[subset["sub_category"] == sub_cat]
    subset = subset.sort_values("ratings", ascending=False)
    st.write(f"Showing **{len(subset)}** products")
    show_grid(subset.head(12))

# ---------------------------------------------------------------- similar
with tab_similar:
    st.write("Pick a product to see similar items.")
    # label -> id so the selectbox is readable
    label_to_id = {f"{r['name']}  ·  {r['ProductId']}": r["ProductId"]
                   for _, r in df.iterrows()}
    choice = st.selectbox("Product", list(label_to_id.keys()))
    pid = label_to_id[choice]
    same_sub = st.checkbox("Only recommend within the same sub-category")
    top_n = st.slider("Number of recommendations", 3, 12, 8)

    st.divider()
    st.subheader("Selected product")
    product_card(rec.get_product(pid), wide=True)

    st.subheader("Recommended for you")
    recs = rec.recommend_by_product(pid, top_n=top_n, same_subcategory=same_sub)
    show_grid(recs)

# ---------------------------------------------------------------- search
with tab_search:
    query = st.text_input(
        "Describe what you're looking for",
        placeholder="e.g. lightweight waterproof backpack with padded straps",
    )
    top_n = st.slider("Results", 3, 12, 8, key="search_n")
    if query:
        results = rec.recommend_by_text(query, top_n=top_n)
        st.write(f"Top matches for: _{query}_")
        show_grid(results)
    else:
        st.info("Type a description above to search the catalog.")
