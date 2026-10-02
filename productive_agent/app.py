import streamlit as st
from database import init_db
from agent import handle_message
from tools import (
    get_tasks,
    get_habit_summary,
    get_goals
)

def main():
    init_db()

    st.set_page_config(
        page_title="Productivity OS",
        layout="centered"
    )

    st.title("Productivity OS")

    page = st.sidebar.selectbox(
        "Page",
        ["Chat", "Tasks", "Habits", "Goals"]
    )

    # ---------------- CHAT ----------------

    if page == "Chat":

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "How can I help?"
            }]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Type here..."):

            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.write(prompt)

            reply = handle_message(prompt)

            with st.chat_message("assistant"):
                st.write(reply)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

    # ---------------- TASKS ----------------

    elif page == "Tasks":

        st.subheader("Tasks")

        st.markdown(get_tasks())

        st.divider()

        task = st.text_input("New task")

        if st.button("Add Task"):
            if task:
                result = handle_message(f"add task {task}")
                st.success(result)
                st.rerun()

    # ---------------- HABITS ----------------

    elif page == "Habits":

        st.subheader("Habits")

        st.markdown(get_habit_summary())

        st.divider()

        habit = st.text_input("New habit")

        if st.button("Add Habit"):
            if habit:
                result = handle_message(f"add habit {habit}")
                st.success(result)
                st.rerun()

    # ---------------- GOALS ----------------

    elif page == "Goals":

        st.subheader("Goals")

        st.markdown(get_goals())

        st.divider()

        goal = st.text_input("New goal")

        if st.button("Add Goal"):
            if goal:
                result = handle_message(f"add goal {goal}")
                st.success(result)
                st.rerun()

if __name__ == "__main__":
    main()