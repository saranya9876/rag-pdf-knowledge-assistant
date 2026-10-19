def init_session(session_state):
    if "messages" not in session_state:
        session_state.messages = []
    return session_state.messages