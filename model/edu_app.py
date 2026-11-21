
import os
import pandas as pd
import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained Random Forest wage prediction model
wage_model = joblib.load("education_wage_model.pkl")

# Features required by the Random Forest model
WAGE_FEATURES = [
    "cip_title",
    "credential_name",
    "cip_family_title",
    "occupation_typical_entry_education",
    "occupation_employment_2024_thousands",
    "occupation_growth_pct_2024_34",
    "occupation_annual_openings_thousands",
    "linked_occupations_count",
    "ai_software_occupation_share",
    "expert_system_occupation_share",
    "hot_technologies_mean_per_occupation"
]


# Predict annual occupation wage
def predict_occupation_wage(record):

    input_data = pd.DataFrame(
        [[record[feature] for feature in WAGE_FEATURES]],
        columns=WAGE_FEATURES
    )

    predicted_wage = wage_model.predict(input_data)[0]

    return round(float(predicted_wage), 2)


def find_wage_prediction(question):

    wage_data = load_wage_data()

    question_lower = question.lower()

    # Find matching education program
    matches = wage_data[
        wage_data["cip_title"].astype(str).str.lower().apply(
            lambda name: name in question_lower
        )
    ]

    if matches.empty:
        return None

    # Select the first matching record with valid model inputs
    matches = matches.dropna(subset=WAGE_FEATURES)

    if matches.empty:
        return None

    record = matches.iloc[0]

    predicted_wage = predict_occupation_wage(record)

    return {
        "program": record["cip_title"],
        "credential": record["credential_name"],
        "predicted_wage": predicted_wage
    }


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="EduCareer AI",
    page_icon="🎓",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Load original dataset for Random Forest wage prediction
@st.cache_data
def load_wage_data():

    file_path = os.path.join(
        PROJECT_DIR,
        "slm_dataset.csv.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Original wage dataset not found: slm_dataset.csv.csv"
        )

    df = pd.read_csv(file_path)

    return df


# --------------------------------
# EDUCAREER AI - MODERN UI THEME
# --------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(
        120deg,
        #80caff 0%,
        #d9d8ff 48%,
        #edc9ff 100%
    );
    color: #25253b;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar glass effect */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.65);
    border-right: 1px solid rgba(255,255,255,0.8);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #30304a;
}

/* Main headings */
h1, h2, h3 {
    color: #292943 !important;
    font-weight: 700;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 5px 20px rgba(80,80,150,0.08);
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(80,80,150,0.10);
}

/* Buttons */
.stButton button {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.8);
    background: rgba(255,255,255,0.75);
    color: #34345a;
    transition: 0.25s ease;
}

.stButton button:hover {
    background: #ffffff;
    border-color: #8b83ef;
    color: #5146b9;
}

/* Dataset status message */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

/* Hide default header decoration */
#MainMenu {
    visibility: hidden;
}

/* --------------------------------
   EDUCAREER AI - READABILITY FIX
-------------------------------- */

/* Make sidebar more solid and readable */
[data-testid="stSidebar"] {
    background: rgba(245, 247, 255, 0.97) !important;
    border-right: 1px solid rgba(100, 110, 170, 0.15);
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span {
    color: #30304f !important;
}

/* Main content text */
[data-testid="stAppViewContainer"] .main {
    color: #292943 !important;
}

/* Chat message text */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88) !important;
    color: #25253b !important;
    border: 1px solid rgba(255,255,255,0.95);
}

/* All text inside chat messages */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li {
    color: #25253b !important;
}

/* Chat input text */
[data-testid="stChatInput"] textarea {
    color: #25253b !important;
    -webkit-text-fill-color: #25253b !important;
    caret-color: #5146b9 !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #77778f !important;
    -webkit-text-fill-color: #77778f !important;
}

/* Bottom chat area */
[data-testid="stBottom"] {
    background: transparent !important;
}

/* Top header */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* FINAL CHAT INPUT READABILITY FIX */

[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 2px solid #d4d2f5 !important;
    border-radius: 20px !important;
    box-shadow: 0 6px 22px rgba(70,70,130,0.12) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #25253b !important;
    -webkit-text-fill-color: #25253b !important;
    caret-color: #5146b9 !important;
    font-size: 16px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #77778f !important;
    -webkit-text-fill-color: #77778f !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #8178e8 !important;
}

[data-testid="stChatInput"] button {
    background: #6559e8 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
}

</style>

""", unsafe_allow_html=True)

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



def detect_question_intent(question):

    text = question.lower()

    # University-related questions
    if any(word in text for word in [
        "university", "universities", "college",
        "colleges", "admission", "offering courses"
    ]):
        return "university"

    # Graduate earnings
    if any(word in text for word in [
        "earnings", "salary", "salaries",
        "income", "median graduate"
    ]):
        return "earnings"

    # Occupation linked to a specific program
    if any(phrase in text for phrase in [
        "which occupation is most strongly linked",
        "largest linked occupation",
        "occupation linked to",
        "occupations linked to"
    ]):
        return "occupation_link"

    # General career guidance
    if any(phrase in text for phrase in [
        "career options", "career paths",
        "what career", "career opportunities",
        "jobs can i explore"
    ]):
        return "career_guidance"

    # Student debt
    if any(word in text for word in [
        "student debt", "student loan", "debt"
    ]):
        return "debt"

    # Employment statistics
    if any(phrase in text for phrase in [
        "employment rate", "employment trends",
        "unemployment rate"
    ]):
        return "employment"

    # Other general knowledge questions
    return "general"




def find_answer(question, data, vectorizer, matrix):

    # Identify the user's question category
    user_intent = detect_question_intent(question)

    # General questions should go to Ollama
    if user_intent == "general":
        return None, 0.0

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

    # Identify the category of the matched question
    matched_intent = detect_question_intent(
        str(result["question"])
    )

    # Reject unrelated dataset matches
    if user_intent != matched_intent:
        return None, best_score

    return result, best_score





def get_ollama_response(question):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "llama3.2:1b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EduCareer AI, an educational "
                    "and career guidance assistant. "
                    "Give clear, helpful and responsible answers. "
                    "Do not invent statistics or claim that "
                    "general answers came from the EduCareer dataset."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.RequestException as error:
        return (
            "I could not connect to the general AI service. "
            f"Please try again. Details: {error}"
        )

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

# --------------------------------
# EDUCAREER AI - WELCOME SCREEN
# --------------------------------

if not st.session_state.messages:

    
    st.markdown("""
<div style="text-align:center; padding:35px 15px 25px 15px;">
    <div style="font-size:75px; margin-bottom:15px;">🎓</div>
    <h1 style="color:#30305c; font-size:38px; margin-bottom:12px;">
        How can I help you today?
    </h1>
    <p style="color:#505071; font-size:17px;">
        Explore education, discover career paths
        and make informed decisions.
    </p>
</div>
""", unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎓 Explore educational programs",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Tell me about the educational programs available."
            )
            st.rerun()

        if st.button(
            "📊 Compare graduate salaries",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Tell me about graduate earnings and salaries."
            )
            st.rerun()

    with col2:
        if st.button(
            "💼 Explore career opportunities",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What career opportunities are available?"
            )
            st.rerun()

        if st.button(
            "📈 Explore employment trends",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What are the projected employment growth trends?"
            )
            st.rerun()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------
# 7. USER QUESTION
# --------------------------------

user_question = st.chat_input(
    "Ask your education or career question..."
)

if not user_question and "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")

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

                # Generate a general answer using Ollama
                response = get_ollama_response(user_question)

                st.markdown("### EduCareer AI - General AI Response")
                st.markdown(response)

                st.caption(
                    "This response was generated by Ollama "
                    "and is not verified against the EduCareer dataset."
                )

                # Suggest useful external platforms
                platform_info = """

                **You can also explore these platforms:**

                - [College Scorecard](https://collegescorecard.ed.gov/)
                  - College information, costs and graduate earnings.

                - [CareerOneStop](https://www.careeronestop.org/)
                  - Career exploration, occupations and salary information.

                - [LinkedIn](https://www.linkedin.com/)
                  - Professional networking, skills and job opportunities.

                - [Glassdoor](https://www.glassdoor.com/)
                  - Company reviews and job information.

                """

                response += "\n\n" + platform_info

                st.markdown(platform_info)


    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


# --------------------------------
# 8. SIDEBAR
# --------------------------------


# --------------------------------
# EDUCAREER AI - CUSTOM SIDEBAR
# --------------------------------

with st.sidebar:

    st.markdown("""
        <div style="padding: 12px 4px 24px 4px;">
            <h2 style="color:#303052; margin:0;">
                🎓 EduCareer AI
            </h2>
            <p style="color:#666680; font-size:13px;">
                Education & Career Assistant
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.text_input(
        "🔍 Search",
        placeholder="Search conversations..."
    )

    if st.button("✏️  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### Explore")

    if st.button("📚  Career Library", use_container_width=True):
        st.session_state.sidebar_page = "Career Library"

    if st.button("💼  Career Recommendations", use_container_width=True):
        st.session_state.sidebar_page = "Career Recommendations"

    if st.button("📊  Career Comparison", use_container_width=True):
        st.session_state.sidebar_page = "Career Comparison"

    st.markdown("---")

    st.markdown("### Chat History")

    if not st.session_state.messages:
        st.caption("Your conversations will appear here.")
    else:
        st.caption("Current conversation is active.")

    st.markdown("---")

    st.caption("Powered by EduCareer AI")
    st.caption("Fine-tuned FLAN-T5 | EduCareer Dataset")

