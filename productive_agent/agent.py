from http.client import responses

import ollama
import json
import re
from tools import (
add_task, get_tasks, complete_task,
    add_habit, log_habit, get_habit_summary,
    add_goal, update_goal_progress, get_goals,
    get_briefing_data
)

MODEL = "qwen3:8b"

SYSTEM_PROMPT = """/no_think
You are a personal productivity assistant. Parse the user's message and respond ONLY with a JSON tool call.

Available tools:
- add_task(title, priority, deadline, est_hours)     priority: low/medium/high, deadline: YYYY-MM-DD
- get_tasks()
- complete_task(task_id)
- add_habit(name, frequency)                          frequency: daily/weekly
- log_habit(habit_name, completed)                    completed: true/false
- get_habit_summary()
- add_goal(title, deadline)
- update_goal_progress(goal_id, progress)             progress: 0-100
- get_goals()
- morning_briefing()

Always respond ONLY with JSON:
{"tool": "add_task", "args": {"title": "...", "priority": "high", "deadline": "2025-06-01"}}

If nothing fits, use:
{"tool": "chat", "args": {"message": "your reply"}}
"""

TOOLS = {
    "add_task": add_task,
    "get_tasks": get_tasks,
    "complete_task": complete_task,
    "add_habit": add_habit,
    "log_habit": log_habit,
    "get_habit_summary": get_habit_summary,
    "add_goal": add_goal,
    "update_goal_progress": update_goal_progress,
    "get_goals": get_goals,
}

def ask_llm(user_message):
    response = ollama.chat(
        model = MODEL,
        messages=[
            {"role":"system" , "content":SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    raw = response["message"]["content"].strip()
    try:
        return json.loads(raw) # convert json string to python dictionary
    # if parsing fails (happen when model does not return pure json)
    except json.JSONDecodeError:
        # regex(\{.*\}) extraction try to find json inside larger text
        json_match = re.search(r'\{.*\}',raw,re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # if everything fail then return the response of llm directly
        return {
            "tool": "chat",
            "args": {"message": raw}
        }

def handle_message(user_massage):
    if "morning_briefing" in user_massage.lower() or "briefing" in user_massage.lower():
        return generate_morning_briefing()

    decision = ask_llm(user_massage)
    tool_name = decision.get("tool","chat") # if tool exist return tool else return default value(chat)
    args = decision.get("args",{})

    if tool_name == "chat":# key message ,default message
        return args.get("message","I'm not sure how to help with that.")

    if tool_name == "morning_briefing":
        return generate_morning_briefing()

    if tool_name in TOOLS:
        try:
            result = TOOLS[tool_name](**args)
            return result
        except TypeError as e :
            return f"⚠️ Invalid arguments for '{tool_name}': {e}"
        except Exception as e:
            return f"⚠️ Error in '{tool_name}': {str(e)}"

    return f"Unknown tool: {tool_name}"


def generate_morning_briefing():
    data = get_briefing_data()
    prompt = f"""
    /no_think
    Generate a friendly, motivating morning briefing based on this data.
    Keep it concise — max 10 lines. Be direct.

    Date: {data['date']}
    Overdue tasks: {data['overdue_tasks']}
    Top pending tasks: {data['top_tasks']}
    Habits not yet logged today: {data['pending_habits']}
    Active goals: {data['goals']}

    Write it as a morning message, not as a list of facts.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()