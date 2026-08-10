import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from collections import Counter
from datetime import datetime

import spacy
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Text Intelligence | A. Masmi",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD NLP MODELS
# ============================================================

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


@st.cache_resource
def load_transformer_sentiment():
    return pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )


nlp = load_spacy_model()


# ============================================================
# DYNAMIC DATE / TIME
# ============================================================

now = datetime.now()
session_date = now.strftime("%B %d, %Y")
session_time = now.strftime("%I:%M %p")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 14px;
}

.hero-subtitle {
    font-size: 1.15rem;
    opacity: 0.78;
}

.developer-info {
    margin-top: 22px;
    font-size: 0.98rem;
    line-height: 1.7;
    opacity: 0.90;
}

.workflow {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
    margin-bottom: 25px;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 600;
}

.ai-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

header_html = f"""<div class="hero">
<div class="hero-title">🧠 AI Text Intelligence Dashboard</div>
<div class="hero-subtitle">NLP • Transformer AI • Sentiment • Entities • Keywords • TF-IDF • Similarity</div>
<div class="developer-info">
<strong>Developed by A. Masmi</strong><br>
jovina&#64;gmx.us<br>
Session: {session_date} • {session_time}
</div>
</div>"""

st.markdown(
    header_html,
    unsafe_allow_html=True
)


st.write(
    """
    Analyze natural-language text using traditional NLP techniques
    and a pretrained Transformer model. Explore sentiment, entities,
    keywords, TF-IDF features, text similarity and linguistic preprocessing.
    """
)


workflow_html = """<div class="workflow">
Text Input → Preprocess → NLP → Transformer AI → Analyze → Compare → Visualize
</div>"""

st.markdown(
    workflow_html,
    unsafe_allow_html=True
)


# ============================================================
# CHECK SPACY MODEL
# ============================================================

if nlp is None:

    st.error(
        "The spaCy English model en_core_web_sm is not installed."
    )

    st.code(
        "python -m spacy download en_core_web_sm"
    )

    st.stop()


# ============================================================
# TEXT INPUT
# ============================================================

default_text = """
Artificial intelligence is transforming how organizations analyze data,
automate tasks and build intelligent applications. Machine learning,
natural language processing and data science are becoming important
skills across many industries. Companies in Canada and the United States
are increasingly investing in artificial intelligence technologies.
"""


text = st.text_area(
    "Enter or paste text to analyze",
    value=default_text.strip(),
    height=220
)


if not text.strip():

    st.info(
        "Enter some text above to start the NLP analysis."
    )

    st.stop()


# ============================================================
# NLP PROCESSING
# ============================================================

doc = nlp(text)

words = [
    token.text.lower()
    for token in doc
    if token.is_alpha
]

content_words = [
    token.lemma_.lower()
    for token in doc
    if token.is_alpha
    and not token.is_stop
    and not token.is_punct
]

sentences = list(doc.sents)

word_count = len(words)
sentence_count = len(sentences)
character_count = len(text)
unique_words = len(set(words))

avg_sentence_length = (
    word_count / sentence_count
    if sentence_count
    else 0
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Words",
    f"{word_count:,}"
)

m2.metric(
    "Sentences",
    f"{sentence_count:,}"
)

m3.metric(
    "Characters",
    f"{character_count:,}"
)

m4.metric(
    "Unique Words",
    f"{unique_words:,}"
)

m5.metric(
    "Avg Words / Sentence",
    f"{avg_sentence_length:.1f}"
)

st.divider()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📘 Overview",
    "🤖 AI Sentiment",
    "🏷 Entities",
    "🔑 Keywords",
    "📊 Word Frequency",
    "🧮 TF-IDF",
    "🔗 Text Similarity",
    "🧹 NLP Preprocessing"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Text Overview"
    )

    overview = pd.DataFrame({
        "Metric": [
            "Word Count",
            "Sentence Count",
            "Character Count",
            "Unique Words",
            "Average Sentence Length"
        ],

        "Value": [
            word_count,
            sentence_count,
            character_count,
            unique_words,
            round(
                avg_sentence_length,
                2
            )
        ]
    })

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Sentence Explorer"
    )

    sentence_df = pd.DataFrame({
        "Sentence #":
            range(
                1,
                len(sentences) + 1
            ),

        "Text": [
            sentence.text.strip()
            for sentence in sentences
        ]
    })

    st.dataframe(
        sentence_df,
        use_container_width=True,
        hide_index=True
    )

    vocabulary_ratio = (
        unique_words / word_count
        if word_count
        else 0
    )

    v1, v2 = st.columns(2)

    v1.metric(
        "Vocabulary Diversity",
        f"{vocabulary_ratio:.2%}"
    )

    v2.metric(
        "Content Words",
        f"{len(content_words):,}"
    )


# ============================================================
# AI SENTIMENT
# ============================================================

with tabs[1]:

    st.header(
        "AI Sentiment Analysis"
    )

    st.write(
        """
        This section compares a traditional lexical sentiment method
        with a pretrained DistilBERT Transformer model.
        """
    )


    # --------------------------------------------------------
    # TEXTBLOB
    # --------------------------------------------------------

    blob = TextBlob(text)

    polarity = (
        blob.sentiment.polarity
    )

    subjectivity = (
        blob.sentiment.subjectivity
    )

    if polarity > 0.10:

        textblob_label = "POSITIVE"

    elif polarity < -0.10:

        textblob_label = "NEGATIVE"

    else:

        textblob_label = "NEUTRAL"


    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    try:

        transformer_sentiment = (
            load_transformer_sentiment()
        )

        transformer_result = (
            transformer_sentiment(
                text,
                truncation=True,
                max_length=512
            )[0]
        )

        transformer_label = (
            transformer_result[
                "label"
            ]
        )

        transformer_score = float(
            transformer_result[
                "score"
            ]
        )

        transformer_loaded = True

    except Exception as e:

        transformer_loaded = False
        transformer_label = "Unavailable"
        transformer_score = 0.0
        transformer_error = str(e)


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Traditional NLP vs Transformer AI"
    )

    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            "### TextBlob"
        )

        st.metric(
            "Sentiment",
            textblob_label
        )

        st.metric(
            "Polarity",
            f"{polarity:.3f}"
        )

        st.metric(
            "Subjectivity",
            f"{subjectivity:.3f}"
        )


    with c2:

        st.markdown(
            "### DistilBERT Transformer"
        )

        if transformer_loaded:

            st.metric(
                "AI Prediction",
                transformer_label
            )

            st.metric(
                "Confidence",
                f"{transformer_score:.2%}"
            )

            st.progress(
                transformer_score
            )

        else:

            st.error(
                "Transformer model could not be loaded."
            )

            st.caption(
                transformer_error
            )


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    if transformer_loaded:

        comparison = pd.DataFrame({
            "Model": [
                "TextBlob",
                "DistilBERT Transformer"
            ],

            "Prediction": [
                textblob_label,
                transformer_label
            ],

            "Score": [
                abs(polarity),
                transformer_score
            ]
        })

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


        comparison_fig = px.bar(
            comparison,
            x="Model",
            y="Score",
            color="Prediction",
            title="Sentiment Model Comparison"
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SENTENCE LEVEL TRANSFORMER
    # --------------------------------------------------------

    st.subheader(
        "Sentence-Level AI Sentiment"
    )

    sentence_results = []

    if transformer_loaded:

        for number, sentence in enumerate(
            sentences,
            start=1
        ):

            sentence_text = (
                sentence.text.strip()
            )

            if not sentence_text:
                continue

            result = (
                transformer_sentiment(
                    sentence_text,
                    truncation=True,
                    max_length=512
                )[0]
            )

            sentence_results.append({
                "Sentence #":
                    number,

                "Sentence":
                    sentence_text,

                "AI Sentiment":
                    result["label"],

                "Confidence":
                    round(
                        float(
                            result["score"]
                        ),
                        4
                    )
            })

        sentence_ai_df = pd.DataFrame(
            sentence_results
        )

        st.dataframe(
            sentence_ai_df,
            use_container_width=True,
            hide_index=True
        )


    st.info(
        """
        The DistilBERT model predicts POSITIVE or NEGATIVE sentiment.
        A high confidence score means the model is confident in that
        classification; it does not mean the statement is factually true.
        """
    )


# ============================================================
# ENTITIES
# ============================================================

with tabs[2]:

    st.header(
        "Named Entity Recognition"
    )

    entities = []

    for ent in doc.ents:

        entities.append({
            "Entity":
                ent.text,

            "Label":
                ent.label_,

            "Description":
                spacy.explain(
                    ent.label_
                )
        })

    if entities:

        entity_df = pd.DataFrame(
            entities
        )

        st.dataframe(
            entity_df,
            use_container_width=True,
            hide_index=True
        )

        entity_counts = (
            entity_df[
                "Label"
            ]
            .value_counts()
            .reset_index()
        )

        entity_counts.columns = [
            "Entity Type",
            "Count"
        ]

        fig_entities = px.bar(
            entity_counts,
            x="Entity Type",
            y="Count",
            title="Named Entity Types"
        )

        st.plotly_chart(
            fig_entities,
            use_container_width=True
        )

    else:

        st.info(
            "No named entities were detected."
        )


# ============================================================
# KEYWORDS
# ============================================================

with tabs[3]:

    st.header(
        "Keyword Extraction"
    )

    keyword_counts = Counter(
        content_words
    )

    keyword_df = pd.DataFrame(
        keyword_counts.most_common(
            20
        ),
        columns=[
            "Keyword",
            "Frequency"
        ]
    )

    if keyword_df.empty:

        st.info(
            "No meaningful keywords were found."
        )

    else:

        st.dataframe(
            keyword_df,
            use_container_width=True,
            hide_index=True
        )

        fig_keywords = px.bar(
            keyword_df,
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Top Extracted Keywords"
        )

        fig_keywords.update_layout(
            yaxis=dict(
                categoryorder=
                    "total ascending"
            )
        )

        st.plotly_chart(
            fig_keywords,
            use_container_width=True
        )


# ============================================================
# WORD FREQUENCY
# ============================================================

with tabs[4]:

    st.header(
        "Word Frequency Analysis"
    )

    frequency_df = pd.DataFrame(
        Counter(
            content_words
        ).most_common(30),

        columns=[
            "Word",
            "Count"
        ]
    )

    if frequency_df.empty:

        st.info(
            "No content words are available."
        )

    else:

        st.dataframe(
            frequency_df,
            use_container_width=True,
            hide_index=True
        )

        freq_fig = px.bar(
            frequency_df,
            x="Word",
            y="Count",
            title="Most Frequent Content Words"
        )

        st.plotly_chart(
            freq_fig,
            use_container_width=True
        )


# ============================================================
# TF-IDF
# ============================================================

with tabs[5]:

    st.header(
        "TF-IDF Feature Analysis"
    )

    sentence_texts = [
        sentence.text.strip()
        for sentence in sentences
        if sentence.text.strip()
    ]

    if len(sentence_texts) < 2:

        st.info(
            "Enter at least two sentences for TF-IDF analysis."
        )

    else:

        try:

            vectorizer = (
                TfidfVectorizer(
                    stop_words="english"
                )
            )

            tfidf_matrix = (
                vectorizer
                .fit_transform(
                    sentence_texts
                )
            )

            feature_names = np.array(
                vectorizer
                .get_feature_names_out()
            )

            mean_scores = np.asarray(
                tfidf_matrix.mean(
                    axis=0
                )
            ).ravel()

            top_indices = (
                mean_scores
                .argsort()[::-1][:20]
            )

            tfidf_df = pd.DataFrame({
                "Term":
                    feature_names[
                        top_indices
                    ],

                "TF-IDF Score":
                    mean_scores[
                        top_indices
                    ]
            })

            st.dataframe(
                tfidf_df,
                use_container_width=True,
                hide_index=True
            )

            tfidf_fig = px.bar(
                tfidf_df,
                x="TF-IDF Score",
                y="Term",
                orientation="h",
                title="Most Important TF-IDF Terms"
            )

            tfidf_fig.update_layout(
                yaxis=dict(
                    categoryorder=
                        "total ascending"
                )
            )

            st.plotly_chart(
                tfidf_fig,
                use_container_width=True
            )

        except ValueError:

            st.warning(
                "Not enough meaningful terms for TF-IDF."
            )


# ============================================================
# TEXT SIMILARITY
# ============================================================

with tabs[6]:

    st.header(
        "Text Similarity"
    )

    second_text = st.text_area(
        "Enter a second text for comparison",
        value=(
            "Machine learning and artificial intelligence "
            "help organizations analyze information and "
            "automate business processes."
        ),
        height=160
    )

    if second_text.strip():

        try:

            similarity_vectorizer = (
                TfidfVectorizer(
                    stop_words="english"
                )
            )

            vectors = (
                similarity_vectorizer
                .fit_transform([
                    text,
                    second_text
                ])
            )

            similarity = (
                cosine_similarity(
                    vectors[0:1],
                    vectors[1:2]
                )[0][0]
            )

            st.metric(
                "Cosine Similarity",
                f"{similarity:.3f}"
            )

            st.progress(
                min(
                    max(
                        float(
                            similarity
                        ),
                        0.0
                    ),
                    1.0
                )
            )

            if similarity >= 0.70:

                st.success(
                    "Strong lexical similarity."
                )

            elif similarity >= 0.40:

                st.info(
                    "Moderate lexical similarity."
                )

            else:

                st.warning(
                    "Low lexical similarity."
                )

        except ValueError:

            st.warning(
                "Not enough meaningful text for comparison."
            )


# ============================================================
# NLP PREPROCESSING
# ============================================================

with tabs[7]:

    st.header(
        "NLP Preprocessing"
    )

    preprocessing_rows = []

    for token in doc:

        if token.is_alpha:

            preprocessing_rows.append({
                "Original Token":
                    token.text,

                "Lowercase":
                    token.text.lower(),

                "Lemma":
                    token.lemma_,

                "Stop Word":
                    token.is_stop,

                "Part of Speech":
                    token.pos_,

                "POS Description":
                    spacy.explain(
                        token.pos_
                    )
            })

    preprocessing_df = pd.DataFrame(
        preprocessing_rows
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Cleaned NLP Text"
    )

    cleaned_text = " ".join(
        content_words
    )

    st.text_area(
        "Preprocessed output",
        value=cleaned_text,
        height=160
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Original Words",
        word_count
    )

    p2.metric(
        "Content Words",
        len(content_words)
    )

    p3.metric(
        "Removed Stop Words",
        max(
            word_count
            - len(content_words),
            0
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_html = f"""<div style="text-align:center; padding:20px; opacity:0.75; line-height:1.7;">
<strong>AI Text Intelligence Dashboard</strong><br>
Developed by A. Masmi • jovina&#64;gmx.us<br>
Python • spaCy • DistilBERT • Transformers • TF-IDF • scikit-learn • Plotly • Streamlit<br>
Session: {session_date} • {session_time}
</div>"""

st.markdown(
    footer_html,
    unsafe_allow_html=True
)
