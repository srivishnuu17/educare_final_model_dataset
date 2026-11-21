
import os
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="EduCareer AI",
    page_icon="🎓",
    layout="centered"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------
# 2. LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():

    files = [
        "train_data.csv",
        "validation_data.csv",
        "test_data.csv"
    ]

    dataframes = []

    for file in files:
        path = os.path.join(PROJECT_DIR, file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {file}")

        df = pd.read_csv(path)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    data = data.dropna(
        subset=["question", "answer", "program"]
    )

    data["search_text"] = (
        data["program"].astype(str) + " " +
        data["question"].astype(str)
    )

    return data.reset_index(drop=True)


# --------------------------------
# 3. PREPARE SEARCH ENGINE
# --------------------------------

@st.cache_resource
def prepare_search():

    data = load_data()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        data["search_text"]
    )

    return vectorizer, matrix


# --------------------------------
# 4. FIND BEST DATASET ANSWER
# --------------------------------

def find_answer(question, data, vectorizer, matrix):

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.12:
        return None, best_score

    result = data.iloc[best_index]

    return result, best_score
def handle_basic_conversation(question):

    text = question.lower().strip()

    if text in [
        "hi", "hello", "hey", "hii",
        "good morning", "good afternoon", "good evening"
    ]:
        return (
            "Hello! Welcome to EduCareer AI. "
            "I can help you explore educational programs, "
            "graduate earnings and career opportunities. "
            "What would you like to know?"
        )

    if "job recommendation" in text or "career guidance" in text:
        return (
            "I'd be happy to help! Please tell me your "
            "qualification, skills, interests and preferred "
            "job location."
        )

    if "thank" in text:
        return "You're welcome! Happy to help."

    if text in ["bye", "goodbye"]:
        return "Goodbye! Thank you for using EduCareer AI."

    return None

# --------------------------------
# 5. CHATBOT INTERFACE
# --------------------------------

st.title("🎓 EduCareer AI")

st.subheader(
    "Education and Career Intelligence Assistant"
)

st.write(
    "Ask questions about educational programs, "
    "graduate earnings, student debt and career opportunities."
)

try:
    data = load_data()
    vectorizer, matrix = prepare_search()

    st.success(
        f"Dataset successfully loaded: {len(data)} records"
    )

except Exception as error:

    st.error(f"Dataset loading error: {error}")
    st.stop()


# --------------------------------
# 6. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------
# 7. USER QUESTION
# --------------------------------

user_question = st.chat_input(
    "Ask your education or career question..."
)

if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

  
    with st.chat_message("assistant"):

        basic_response = handle_basic_conversation(user_question)

        if basic_response is not None:

            response = basic_response
            st.write(response)

        else:

            result, score = find_answer(
                user_question,
                data,
                vectorizer,
                matrix
            )

            if result is not None:

                response = str(result["answer"])

                st.markdown("### EduCareer AI Answer")
                st.write(response)

                st.caption(
                    "Answer retrieved from the original EduCareer dataset."
                )

                with st.expander("View matching dataset record"):

                    st.write("Program:", result["program"])
                    st.write("Matched question:", result["question"])

                    st.write(
                        "Similarity score:",
                        round(float(score), 3)
                    )

                    if "record_id" in result.index:
                        st.write("Record ID:", result["record_id"])

            else:

                response = (
                    "I could not find a sufficiently relevant "
                    "answer in the EduCareer dataset. "
                    "Please try a more specific question."
                )

                st.warning(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


# --------------------------------
# 8. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("About EduCareer AI")

    st.write(
        "An educational intelligence chatbot "
        "powered by your EduCareer dataset."
    )

    st.write("Language model: FLAN-T5-small (integration planned)")

    st.write(f"Dataset records: {len(data)}")

    if st.button("Clear conversation"):

        st.session_state.messages = []
        st.rerun()
