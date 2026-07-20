# Product Recommendation System

A content-based product recommender built with **TF-IDF + cosine similarity**
(scikit-learn) and a **Streamlit** front end. It ships with a self-contained
sample catalog, so it runs the moment you clone it — no dataset setup, no
compiled dependencies.

## Features

- **Browse** products by main / sub category, sorted by rating.
- **Find similar** — pick any product and get the most similar items, with an
  option to stay within the same sub-category.
- **Search** — free-text query ("waterproof hiking backpack with padded
  straps") ranked by TF-IDF similarity.

## Project structure

```
.
├── app.py              # Streamlit UI
├── recommender.py      # TF-IDF content-based engine
├── generate_data.py    # builds the bundled sample dataset
├── data/products.csv   # generated sample catalog (checked in)
├── requirements.txt
└── .streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

To regenerate the sample data:

```bash
python generate_data.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**:
   ```bash
   git init
   git add .
   git commit -m "Content-based product recommender"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**, pick your repo, set the main file to **`app.py`**, and
   deploy. Streamlit installs `requirements.txt` automatically.
4. You'll get a public URL like
   `https://<your-app>.streamlit.app` to share.

## How it works

Each product has a `text` field combining its name, categories, descriptors,
and review. `recommender.py` fits a `TfidfVectorizer` over those blobs, then
ranks items by cosine similarity — either to another product (similar items) or
to a free-text query (search). Everything is cached with Streamlit's
`@st.cache_data` / `@st.cache_resource` so the model is built once per session.

## Notes

- **Why content-based and not the notebook's `surprise` hybrid?** The
  `scikit-surprise` library needs C compilation and frequently fails to build
  on Streamlit Cloud. A pure-sklearn content-based approach gives reliable
  one-click deploys. To extend toward a hybrid later, add a collaborative
  component with `implicit` or `scikit-learn`'s matrix factorization, which
  install as wheels.
- Swap in your own catalog by replacing `data/products.csv` with a CSV that has
  the same columns (`ProductId, name, main_category, sub_category, ratings,
  no_of_ratings, Price, Review, text`).
