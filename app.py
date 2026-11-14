import streamlit as st
import requests
import uuid
import json

# FastAPI backend URLs
API_URL = "http://localhost:8000/chat"
STREAM_URL = "http://localhost:8000/chat/stream"
FEEDBACK_URL = "http://localhost:8000/feedback"
RELOAD_URL = "http://localhost:8000/reload"
LOGIN_URL = "http://localhost:8000/login"
REGISTER_URL = "http://localhost:8000/register"
UPLOAD_URL = "http://localhost:8000/upload-runbook"
RUNBOOKS_URL = "http://localhost:8000/runbooks"
SESSIONS_URL = "http://localhost:8000/sessions"
HISTORY_URL = "http://localhost:8000/history"
FEEDBACK_EXPORT_URL = "http://localhost:8000/feedback/export"

WELCOME_MESSAGE = "Welcome! How can I help you?"

st.set_page_config(page_title="DevOps Support RAG Assistant", page_icon="🤖")

# 1. Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = "user"

# 2. Login Gate (shown until user logs in)
if not st.session_state.authenticated:
    st.title("🔐 DevOps ChatBot Login")
    st.markdown("Sign in to access your enterprise infrastructure runbooks.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            try:
                res = requests.post(
                    LOGIN_URL,
                    json={"username": username, "password": password},
                    timeout=15
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.authenticated = True
                    st.session_state.username = data["username"]
                    st.session_state.role = data["role"]
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend API service. Ensure FastAPI is running.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
    st.stop()

# 3. Main Chat Interface (visible only after login)
st.title("🤖 DevOps ChatBot")
st.markdown("Ask questions about your enterprise infrastructure/APM runbooks.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = f"{st.session_state.username}_{uuid.uuid4()}"

# Display chat history with Thumbs Up / Down feedback buttons
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and idx > 0 and message["content"] != WELCOME_MESSAGE:
            prev_query = st.session_state.messages[idx - 1]["content"]
            f_col1, f_col2, _ = st.columns([0.08, 0.08, 0.84])
            with f_col1:
                if st.button("👍", key=f"up_{st.session_state.session_id}_{idx}", type="tertiary"):
                    try:
                        requests.post(FEEDBACK_URL, json={
                            "username": st.session_state.username,
                            "thread_id": st.session_state.session_id,
                            "query": prev_query,
                            "response": message["content"],
                            "rating": "up"
                        }, timeout=10)
                        st.toast("Thanks for the positive feedback!")
                    except Exception:
                        st.toast("Could not save feedback.")
            with f_col2:
                if st.button("👎", key=f"down_{st.session_state.session_id}_{idx}", type="tertiary"):
                    try:
                        requests.post(FEEDBACK_URL, json={
                            "username": st.session_state.username,
                            "thread_id": st.session_state.session_id,
                            "query": prev_query,
                            "response": message["content"],
                            "rating": "down"
                        }, timeout=10)
                        st.toast("Feedback logged for runbook review.")
                    except Exception:
                        st.toast("Could not save feedback.")

# User input field with real-time token streaming
if prompt := st.chat_input("Ask a troubleshooting question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def token_generator():
            try:
                with requests.post(
                    STREAM_URL,
                    json={"thread_id": st.session_state.session_id, "query": prompt},
                    stream=True,
                    timeout=90
                ) as res:
                    if res.status_code != 200:
                        yield f"Error from server: {res.status_code} - {res.text}"
                        return
                    for line in res.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            payload = line[len("data: "):]
                            if payload == "[DONE]":
                                break
                            chunk = json.loads(payload)
                            yield chunk.get("token", "")
            except requests.exceptions.ConnectionError:
                yield "Error: Could not connect to the backend API service. Ensure FastAPI is running."
            except Exception as e:
                yield f"An unexpected error occurred: {str(e)}"

        answer = st.write_stream(token_generator())

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# 4. Sidebar Controls (ChatGPT Dark Sidebar)
with st.sidebar:
    st.markdown("""
        <style>
        /* Dark ChatGPT sidebar theme */
        section[data-testid="stSidebar"] {
            background-color: #171717 !important;
            color: #ECECEC !important;
        }

        /* Comfortable vertical spacing between sidebar elements to prevent overlap */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }

        /* Light text on sidebar headers, labels, captions, and recent chat items */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] p,
        section[data-testid="stSidebar"] button[kind="tertiary"] p {
            color: #ECECEC !important;
        }

        /* High-contrast dark text inside white Admin Upload & Secondary buttons */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] * {
            color: #1F2937 !important;
        }

        /* Dark-theme styling for Active Runbooks Expander so it is always visible */
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            background-color: #212121 !important;
            border: 1px solid #374151 !important;
            border-radius: 0.5rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            background-color: #212121 !important;
            color: #ECECEC !important;
        }

        /* Dark-theme inline code badges inside Active Runbooks list */
        section[data-testid="stSidebar"] code {
            background-color: #262730 !important;
            color: #60A5FA !important;
            border: 1px solid #374151 !important;
            font-size: 0.78rem !important;
        }

        /* Left-align tertiary buttons (Recent Chat titles) */
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"],
        section[data-testid="stSidebar"] button[kind="tertiary"] {
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            padding: 0.42rem 0.65rem !important;
            border-radius: 0.5rem !important;
            min-height: 2.1rem !important;
            height: 2.1rem !important;
            width: 100% !important;
            border: none !important;
            background: transparent !important;
        }

        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] div,
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] p,
        section[data-testid="stSidebar"] button[kind="tertiary"] div,
        section[data-testid="stSidebar"] button[kind="tertiary"] p {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            margin: 0 !important;
        }

        /* Remove red Streamlit focus ring & style hover/active states */
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:focus,
        section[data-testid="stSidebar"] button[kind="tertiary"]:focus {
            box-shadow: none !important;
            outline: none !important;
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover,
        section[data-testid="stSidebar"] button[kind="tertiary"]:hover {
            background-color: #212121 !important;
        }
        section[data-testid="stSidebar"] .active-chat button {
            background-color: #2F2F2F !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Top bar: App title on left + clean Material compose icon on right
    top_col1, top_col2 = st.columns([0.75, 0.25])
    with top_col1:
        st.markdown("### 🤖 DevOps Bot")
    with top_col2:
        if st.button("", icon=":material/edit_square:", help="New chat", type="tertiary"):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]
            st.session_state.session_id = f"{st.session_state.username}_{uuid.uuid4()}"
            st.rerun()

# Fetch and render Recent Chats" block right here (4 spaces indentation)
    try:
        sess_res = requests.get(f"{SESSIONS_URL}/{st.session_state.username}", timeout=10)
        if sess_res.status_code == 200:
            past_sessions = sess_res.json().get("sessions", [])
            if past_sessions:
                st.caption("Recent Chats")
                for item in past_sessions:
                    tid = item["thread_id"]
                    title = item["title"]
                    is_active = (tid == st.session_state.session_id)

                    chat_col, del_col = st.columns([0.82, 0.18])

                    with chat_col:
                        if is_active:
                            st.markdown('<div class="active-chat">', unsafe_allow_html=True)

                        if st.button(
                            title,
                            key=f"chat_{tid}",
                            type="tertiary",
                            use_container_width=True
                        ):
                            hist_res = requests.get(f"{HISTORY_URL}/{tid}", timeout=10)
                            if hist_res.status_code == 200:
                                st.session_state.session_id = tid
                                st.session_state.messages = hist_res.json().get("messages", [])
                                st.rerun()

                        if is_active:
                            st.markdown('</div>', unsafe_allow_html=True)

                    with del_col:
                        if st.button(
                            "🗑️",
                            key=f"del_{tid}",
                            help="Delete chat",
                            type="tertiary"
                        ):
                            del_res = requests.delete(f"{SESSIONS_URL}/{tid}", timeout=10)
                            if del_res.status_code == 200:
                                if is_active:
                                    st.session_state.messages = [
                                        {"role": "assistant", "content": WELCOME_MESSAGE}
                                    ]
                                    st.session_state.session_id = f"{st.session_state.username}_{uuid.uuid4()}"
                                st.rerun()
            else:
                st.caption("No past chats yet.")
    except Exception as e:
        st.caption(f"Unable to load chats: {e}")


    # Admin-only Knowledge Base & User Management controls
    if st.session_state.role == "admin":
        st.divider()
        st.caption("ADMIN TOOLS")
        uploaded_files = st.file_uploader(
            "Upload Runbook PDFs",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("⬆️ Upload & Index", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                for up_file in uploaded_files:
                    files = {"file": (up_file.name, up_file.getvalue(), up_file.type)}
                    res = requests.post(UPLOAD_URL, files=files, timeout=300)
                    if res.status_code == 200:
                        st.success(res.json().get("message"))
                    else:
                        st.error(f"Failed ({up_file.name}): {res.text}")

        if st.button("🔄 Reload Runbooks", use_container_width=True):
            with st.spinner("Re-indexing PDFs..."):
                res = requests.post(RELOAD_URL, timeout=300)
                if res.status_code == 200:
                    st.success("Runbooks reloaded!")

        with st.expander("📂 Active Runbooks"):
            try:
                rb_res = requests.get(RUNBOOKS_URL, timeout=10)
                if rb_res.status_code == 200:
                    rb_data = rb_res.json()
                    st.caption(f"Total chunks: **{rb_data.get('total_chunks', 0)}**")
                    for fname in rb_data.get("files", []):
                        st.markdown(f"- `{fname}`")
            except Exception:
                pass

        with st.expander("👥 Create New User"):
            with st.form("register_user_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                create_btn = st.form_submit_button("Create User")

                if create_btn:
                    try:
                        reg_res = requests.post(
                            REGISTER_URL,
                            json={"username": new_user, "password": new_pass, "role": new_role},
                            timeout=15
                        )
                        if reg_res.status_code == 200:
                            st.success(f"User '{new_user}' created!")
                        else:
                            st.error(reg_res.json().get("detail", "Registration failed."))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # 👇 Replace your existing Feedback Summary block with this updated version
        with st.expander("📊 Feedback Summary"):
            try:
                fb_res = requests.get(FEEDBACK_URL, timeout=10)
                if fb_res.status_code == 200:
                    fb_data = fb_res.json()
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("👍 Positive", fb_data.get("positive", 0))
                    with m_col2:
                        st.metric("👎 Negative", fb_data.get("negative", 0))

                    # CSV download button appears inside Feedback Summary when negative ratings exist
                    if fb_data.get("negative", 0) > 0:
                        csv_res = requests.get(FEEDBACK_EXPORT_URL, timeout=10)
                        if csv_res.status_code == 200:
                            st.download_button(
                                label="📥 Download 👎 Queries (CSV)",
                                data=csv_res.text,
                                file_name="negative_feedback_report.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                    recent = fb_data.get("recent", [])
                    if recent:
                        st.caption("Recent Ratings")
                        for item in recent:
                            icon = "👍" if item["rating"] == "up" else "👎"
                            short_q = item["query"][:38] + "..." if len(item["query"]) > 38 else item["query"]
                            st.markdown(f"{icon} **{item['username']}**: {short_q}")
                    else:
                        st.caption("No feedback submitted yet.")
                else:
                    st.caption("Failed to load feedback.")
            except Exception:
                st.caption("Unable to connect to feedback API.")
                

    # Bottom User Profile & Logout footer
    st.divider()

    # Bottom User Profile & Logout footer
    st.divider()
    user_col1, user_col2 = st.columns([0.65, 0.35])
    with user_col1:
        st.markdown(f"👤 **{st.session_state.username}**")
    with user_col2:
        if st.button("Logout", type="tertiary"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = "user"
            st.session_state.messages = []
            if "session_id" in st.session_state:
                del st.session_state["session_id"]
            st.rerun()
