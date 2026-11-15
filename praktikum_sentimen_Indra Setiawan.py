"""
Dashboard Analisis Sentimen Twitter dengan Neural Network (MLP)
Jalankan: streamlit run app_sentimen_nn.py
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

st.set_page_config(page_title="Analisis Sentimen NN", page_icon="📊", layout="wide")

# ---------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------
DEFAULT_FILES = ["Data_Kualitas_Udara.xls", "Data_Kualitas_Udara.xlsx", "tweet_data.csv"]
LABELS = ["positif", "netral", "negatif"]
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}
WC_COLORMAP = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}

# Kamus sentimen untuk topik kualitas udara / kabut asap (bisa diedit di sidebar)
DEFAULT_POS = (
    "bersih, sehat, segar, sejuk, nyaman, jernih, baik, bagus, membaik, lega, aman, alhamdulillah, bersyukur, syukur, semoga, semangat, pulih, terbantu, bantuan, membantu, solusi, berhasil, sukses, apresiasi, peduli, dukung, kondusif, lancar, mantap, hebat, puas, terimakasih, makasih, mereda, reda, turun hujan, hujan lebat, hujan deras, cerah, normal, kembali normal, udara bersih, kualitas baik, mendukung, komitmen, inovasi, efektif, optimis, harapan"
)
DEFAULT_NEG = (
    "buruk, parah, bahaya, berbahaya, beracun, racun, tercemar, pencemaran, kotor, pekat, tebal, sesak, sesek, batuk, ispa, penyakit, kanker, menderita, derita, korban, kecewa, kesal, marah, muak, gagal, lalai, lambat, abai, krisis, darurat, bencana, rusak, merusak, gangguan, mengganggu, terganggu, keluhan, mengeluh, khawatir, cemas, panik, menyesakkan, tidak sehat, tak sehat, tidak baik, tidak aman, tidak layak, kritis, mematikan, kematian, meninggal, sakit, perih, pedih, panas, gerah, kabur, menyiksa, tersiksa, bohong, salah, tinggi pm"
)

STOPWORDS = set("""
yang dan di ke dari ini itu untuk dengan pada adalah atau juga tidak ada akan sudah
saya kami kita kamu dia mereka nya aja ya sih deh dong lah kok yg dgn utk krn karena
jadi bisa lagi udah gak ga nggak tak mau sama oleh dalam sebagai bahwa para lebih
masih sangat banget pun saat ketika apa siapa mana kalau jika agar supaya rt amp
""".split())


# ---------------------------------------------------------------
# Fungsi pemrosesan teks
# ---------------------------------------------------------------
def clean_text(text) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"@\w+", " ", text)                    # mention
    text = re.sub(r"#\w+", " ", text)                    # hashtag
    text = re.sub(r"\brt\b", " ", text, flags=re.I)      # retweet marker
    text = text.lower()                                  # case folding
    text = re.sub(r"[^a-z\s]", " ", text)                # simbol, angka, emoji
    text = re.sub(r"\s+", " ", text).strip()             # spasi berlebih
    return text


def parse_words(raw: str) -> set:
    return {w.strip().lower() for w in raw.replace("\n", ",").split(",") if w.strip()}


def label_sentiment(text: str, pos_words: set, neg_words: set) -> str:
    """Hitung kemunculan kata/frasa positif & negatif. Frasa terpanjang dicocokkan lebih dulu
    (mis. 'tidak sehat' dihitung negatif, bukan 'sehat' positif)."""
    t = " " + text + " "
    terms = sorted([(w, 1) for w in pos_words] + [(w, -1) for w in neg_words],
                   key=lambda x: -len(x[0]))
    pos = neg = 0
    for w, sign in terms:
        k = t.count(f" {w} ")
        if k:
            if sign > 0:
                pos += k
            else:
                neg += k
            t = t.replace(f" {w} ", "  ")
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


# ---------------------------------------------------------------
# Data & model (di-cache supaya tidak training ulang tiap interaksi)
# ---------------------------------------------------------------
def read_any(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xls"):
        return pd.read_excel(file, engine="xlrd")   # butuh: pip install xlrd
    return pd.read_excel(file)                       # .xlsx (openpyxl)


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = read_any(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    needed = {"created_at", "username", "full_text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(sorted(missing))}")
    return df[["created_at", "username", "full_text"]].copy()


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, pos_raw: str, neg_raw: str) -> pd.DataFrame:
    pos_words, neg_words = parse_words(pos_raw), parse_words(neg_raw)
    out = df.dropna(subset=["full_text"]).copy()
    out["clean_text"] = out["full_text"].apply(clean_text)
    out = out[out["clean_text"].str.len() > 0].drop_duplicates(subset="clean_text")
    out["sentimen"] = out["clean_text"].apply(lambda t: label_sentiment(t, pos_words, neg_words))
    return out.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_model(texts: tuple, labels: tuple, hidden: int, max_iter: int, test_size: float, seed: int):
    X_text, y = pd.Series(texts), pd.Series(labels)
    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=test_size, random_state=seed, stratify=stratify)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                          alpha=0.1, max_iter=max_iter, random_state=seed)
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    return vec, model, y_te.to_numpy(), y_pred, len(X_tr), len(X_te)


def make_wordcloud(text: str, colormap: str):
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return None
    return WordCloud(width=800, height=400, background_color="white",
                     colormap=colormap, max_words=100).generate_from_frequencies(Counter(words))


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.title("📊 Analisis Sentimen Twitter dengan Neural Network")
st.caption("Pipeline: Cleansing → Pelabelan Rule-Based → TF-IDF → MLP → Evaluasi → Prediksi")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded = st.file_uploader("Upload data (opsional)", type=["csv", "xls", "xlsx"],
                                help="Jika kosong, aplikasi mencari Data_Kualitas_Udara.xls / tweet_data.csv di folder yang sama.")
    st.subheader("Kamus Sentimen")
    pos_raw = st.text_area("Kata positif (pisahkan koma)", DEFAULT_POS, height=120)
    neg_raw = st.text_area("Kata negatif (pisahkan koma)", DEFAULT_NEG, height=120)
    st.subheader("Parameter Neural Network")
    hidden = st.slider("Neuron hidden layer", 10, 300, 100, step=10)
    max_iter = st.slider("Max iterasi", 100, 1000, 300, step=50)
    test_size = st.slider("Porsi data uji", 0.1, 0.4, 0.2, step=0.05)
    seed = st.number_input("Random state", value=42, step=1)

# ---------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------
import os
source = uploaded if uploaded is not None else next((f for f in DEFAULT_FILES if os.path.exists(f)), None)
if source is None:
    st.error("Data tidak ditemukan. Taruh `Data_Kualitas_Udara.xls` (atau `tweet_data.csv`) di folder yang sama dengan app ini, atau upload lewat sidebar.")
    st.stop()
try:
    raw_df = load_data(source)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

df = preprocess(raw_df, pos_raw, neg_raw)
if df.empty:
    st.error("Data kosong setelah preprocessing.")
    st.stop()
if df["sentimen"].nunique() < 2:
    st.error("Hanya ada 1 kelas sentimen. Tambah/ubah kata di kamus sentimen agar ada minimal 2 kelas.")
    st.stop()

tab_data, tab_dist, tab_wc, tab_eval, tab_pred = st.tabs(
    ["📄 Data & Cleansing", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"])

# ---------------------------------------------------------------
# Tab 1: Data
# ---------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweet mentah", len(raw_df))
    c2.metric("Setelah cleansing", len(df))
    c3.metric("Jumlah kelas", df["sentimen"].nunique())
    st.subheader("Hasil cleansing & pelabelan")
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]],
                 use_container_width=True, height=400)
    st.download_button("⬇️ Unduh data berlabel (CSV)",
                       df.to_csv(index=False).encode("utf-8"), "tweet_berlabel.csv", "text/csv")

# ---------------------------------------------------------------
# Tab 2: Distribusi
# ---------------------------------------------------------------
with tab_dist:
    counts = df["sentimen"].value_counts().reindex(LABELS).fillna(0).astype(int)
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bar chart")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.bar_label(bars)
        ax.set_ylabel("Jumlah tweet")
        st.pyplot(fig)
    with c2:
        st.subheader("Pie chart")
        fig, ax = plt.subplots(figsize=(5, 4))
        nz = counts[counts > 0]
        ax.pie(nz.values, labels=nz.index, autopct="%1.1f%%", startangle=90,
               colors=[colors[LABELS.index(l)] for l in nz.index])
        ax.axis("equal")
        st.pyplot(fig)
    st.dataframe(counts.rename("jumlah").to_frame().T, use_container_width=True)

# ---------------------------------------------------------------
# Tab 3: WordCloud
# ---------------------------------------------------------------
with tab_wc:
    cols = st.columns(3)
    for col, lab in zip(cols, LABELS):
        with col:
            st.subheader(f"{EMOJI[lab]} {lab.capitalize()}")
            text = " ".join(df.loc[df["sentimen"] == lab, "clean_text"])
            wc = make_wordcloud(text, WC_COLORMAP[lab])
            if wc is None:
                st.info("Tidak ada data.")
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ---------------------------------------------------------------
# Tab 4: Model & evaluasi
# ---------------------------------------------------------------
with tab_eval:
    with st.spinner("Melatih model Neural Network..."):
        vec, model, y_true, y_pred, n_train, n_test = train_model(
            tuple(df["clean_text"]), tuple(df["sentimen"]), hidden, max_iter, test_size, int(seed))

    st.write(f"Data latih: **{n_train}** | Data uji: **{n_test}** | "
             f"Arsitektur: TF-IDF (uni+bigram, min_df=2) → MLP({hidden}) ReLU, Adam, alpha=0.1 | Iterasi: {model.n_iter_}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{acc:.2%}")
    m2.metric("Presisi", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1-Score", f"{f1:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Grafik metrik")
        fig, ax = plt.subplots(figsize=(5, 4))
        vals = [acc, prec, rec, f1]
        bars = ax.bar(["Akurasi", "Presisi", "Recall", "F1"], vals,
                      color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals])
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
    with c2:
        st.subheader("Confusion matrix")
        cls = [l for l in LABELS if l in model.classes_]
        cm = confusion_matrix(y_true, y_pred, labels=cls)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=cls).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)

    st.subheader("Classification report")
    rep = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Kurva loss training")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(model.loss_curve_)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

# ---------------------------------------------------------------
# Tab 5: Prediksi
# ---------------------------------------------------------------
with tab_pred:
    st.subheader("Prediksi sentimen teks baru")
    user_text = st.text_area("Masukkan teks / tweet:", height=120,
                             placeholder="Contoh: udara hari ini bersih dan sehat setelah hujan")
    if st.button("Prediksi", type="primary"):
        cleaned = clean_text(user_text)
        if not cleaned:
            st.warning("Teks kosong setelah dibersihkan. Coba masukkan kalimat lain.")
        else:
            proba = model.predict_proba(vec.transform([cleaned]))[0]
            pred = model.classes_[int(np.argmax(proba))]
            st.markdown(f"## {EMOJI[pred]} Sentimen: **{pred.upper()}**")
            st.write(f"Teks setelah cleansing: `{cleaned}`")
            prob_df = pd.DataFrame({"sentimen": model.classes_, "probabilitas": proba})
            prob_df["label"] = prob_df["sentimen"].map(lambda s: f"{EMOJI[s]} {s}")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(prob_df[["label", "probabilitas"]].style.format({"probabilitas": "{:.2%}"}),
                             hide_index=True, use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                ax.barh(prob_df["label"], prob_df["probabilitas"],
                        color=[{"positif": "#2ca02c", "netral": "#1f77b4", "negatif": "#d62728"}[s]
                               for s in prob_df["sentimen"]])
                ax.set_xlim(0, 1)
                st.pyplot(fig)
