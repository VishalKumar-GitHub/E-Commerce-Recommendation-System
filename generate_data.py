"""Generate a self-contained sample product dataset for the recommender.

Run once to (re)create data/products.csv. The Streamlit app reads that file,
so the app has zero external dependencies and deploys with no setup.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# (main_category, sub_category, [product name templates], [descriptor words])
CATALOG = {
    "electronics": {
        "headphones": (
            ["Wireless Over-Ear Headphones", "Noise-Cancelling Earbuds",
             "Studio Monitor Headphones", "Bluetooth Sport Earphones",
             "Gaming Headset with Mic"],
            ["crisp sound", "deep bass", "long battery life", "comfortable fit",
             "noise cancelling", "clear microphone", "foldable design"],
        ),
        "smartphones": (
            ["6.5in Android Smartphone", "Compact 5G Phone",
             "Budget Dual-SIM Phone", "Flagship Camera Phone",
             "Rugged Outdoor Phone"],
            ["fast processor", "bright display", "great camera",
             "all-day battery", "sleek design", "expandable storage"],
        ),
        "laptops": (
            ["14in Ultrabook", "Gaming Laptop 16GB", "2-in-1 Convertible Laptop",
             "Business Notebook", "Student Chromebook"],
            ["lightweight", "fast SSD", "long battery", "backlit keyboard",
             "high-res screen", "powerful GPU"],
        ),
    },
    "home & kitchen": {
        "coffee": (
            ["Programmable Drip Coffee Maker", "Espresso Machine",
             "French Press 1L", "Cold Brew Maker", "Single-Serve Pod Brewer"],
            ["rich flavor", "easy to clean", "fast brewing", "compact size",
             "durable build", "keeps coffee hot"],
        ),
        "cookware": (
            ["Non-Stick Frying Pan", "Stainless Steel Pot Set",
             "Cast Iron Skillet", "Ceramic Baking Dish", "Knife Block Set"],
            ["even heating", "scratch resistant", "dishwasher safe",
             "sturdy handle", "premium finish", "non-stick coating"],
        ),
        "vacuum": (
            ["Cordless Stick Vacuum", "Robot Vacuum Cleaner",
             "Handheld Car Vacuum", "Upright Bagless Vacuum", "Wet-Dry Vacuum"],
            ["powerful suction", "quiet motor", "long runtime",
             "easy to empty", "lightweight", "smart navigation"],
        ),
    },
    "fashion": {
        "shoes": (
            ["Running Sneakers", "Leather Ankle Boots", "Canvas Slip-Ons",
             "Trail Hiking Shoes", "Classic Loafers"],
            ["comfortable", "durable sole", "breathable", "stylish look",
             "good grip", "lightweight"],
        ),
        "bags": (
            ["Laptop Backpack", "Leather Tote Bag", "Travel Duffel",
             "Crossbody Sling", "Waterproof Hiking Pack"],
            ["spacious", "water resistant", "padded straps", "sleek design",
             "many pockets", "durable zippers"],
        ),
        "watches": (
            ["Minimalist Analog Watch", "Digital Sports Watch",
             "Smartwatch Fitness Tracker", "Chronograph Steel Watch",
             "Classic Leather Strap Watch"],
            ["elegant", "water resistant", "long battery", "accurate time",
             "comfortable band", "scratch-proof glass"],
        ),
    },
    "sports & outdoors": {
        "fitness": (
            ["Adjustable Dumbbell Set", "Yoga Mat Non-Slip",
             "Resistance Band Kit", "Foam Roller", "Kettlebell 12kg"],
            ["durable", "good grip", "compact storage", "versatile",
             "comfortable", "sturdy build"],
        ),
        "camping": (
            ["2-Person Tent", "Sleeping Bag -5C", "Portable Camp Stove",
             "LED Headlamp", "Insulated Water Bottle"],
            ["waterproof", "lightweight", "easy setup", "keeps warm",
             "compact", "durable materials"],
        ),
    },
}

POSITIVE = [
    "Absolutely love this, exceeded my expectations and works perfectly.",
    "Great value for the price, highly recommend to anyone.",
    "Excellent quality and does exactly what it promises.",
    "A game changer for my daily routine, so glad I bought it.",
    "Well made and reliable, would buy again without hesitation.",
]
MIXED = [
    "Decent product overall but has a few minor annoyances.",
    "Works fine, though the build quality could be better.",
    "Good enough for the price, not amazing but does the job.",
]
NEGATIVE = [
    "Did not live up to the hype, a bit disappointed.",
    "Stopped working after a short while, not great.",
]


def build(n_users: int = 400) -> pd.DataFrame:
    rows = []
    pid = 0
    for main_cat, subs in CATALOG.items():
        for sub_cat, (names, descriptors) in subs.items():
            # several variants per template so we have a richer catalog
            for base_name in names:
                for _ in range(6):
                    pid += 1
                    descs = RNG.choice(descriptors, size=3, replace=False)
                    name = f"{base_name} - {descs[0].title()}"
                    pools = [POSITIVE, POSITIVE, MIXED, NEGATIVE]
                    review_pool = pools[RNG.integers(0, len(pools))]
                    review = str(RNG.choice(review_pool))
                    # searchable text blob for TF-IDF
                    text = f"{name} {sub_cat} {main_cat} {' '.join(descs)} {review}"
                    rows.append({
                        "ProductId": f"P{pid:06d}",
                        "name": name,
                        "main_category": main_cat,
                        "sub_category": sub_cat,
                        "ratings": round(float(RNG.uniform(3.0, 5.0)), 1),
                        "no_of_ratings": int(RNG.integers(5, 5000)),
                        "Price": round(float(RNG.uniform(10, 300)), 2),
                        "Review": review,
                        "text": text,
                    })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = build()
    df.to_csv("data/products.csv", index=False)
    print(f"Wrote data/products.csv with {len(df)} products "
          f"across {df['main_category'].nunique()} categories.")
