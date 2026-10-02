import sqlite3
from datetime import date , datetime
from database import get_conn
from datetime import timedelta

def add_task(title , priority="medium", deadline = "",est_hours=None):
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO tasks(title , priority , deadline , est_hours , created_at)
            VALUES (?,?,?,?,?)
        """, (title,priority,deadline,est_hours,datetime.now().isoformat()))
        conn.commit()

    finally:
        conn.close()

    result = f"✅ Task Added: '{title}' [{priority} priority ]"
    if deadline:
        result += f"- due {deadline}"
    return result

def get_tasks():
    conn = get_conn()
    # order by done if same then priority if same then deadline
    try:
        rows = conn.execute("""
             SELECT id, title, priority, deadline, done FROM tasks
             ORDER BY done, 
             CASE priority
                 WHEN 'high' THEN 1 
                 WHEN 'medium' THEN 2
                 ELSE 3
             END,
             deadline
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No tasks yet."

    pending = [r for r in rows if not r[4]]
    done = [r for r in rows if r[4]]
    lines = []

    if pending:
        lines.append("*- Pending task -*")
        for tid , title , priority , deadline , _ in pending:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            due = f"-due {deadline}" if deadline else ""
            lines.append(f"{emoji} [{tid} {title}{due}]")

    if done:
        lines.append("*- Completed Tasks -*")
        for _,title,_,_,_ in done:
            lines.append(f"✓ ~~{title} ~~")

    return "\n".join(lines)

def complete_task(task_id):
    conn = get_conn()
    try:
        conn.execute("""
             UPDATE tasks SET done = 1 , done_at = ?  WHERE id= ?
        """, (datetime.now().isoformat(),task_id))
        conn.commit()
    finally:
        conn.close()

    return f"Task id: {task_id} marked as completed "

def get_overdue_task():
    today = date.today().isoformat()
    conn = get_conn()
    try:
        # deadline != "" as some task might not have deadline
        rows = conn.execute("""
            SELECT title , deadline FROM tasks 
            WHERE done = 0 and deadline != '' and deadline < ?
        """, (today,)).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    return [f"'{title}'(was overdue from {deadline})" for title , deadline in rows]

def add_habit(name , frequency= "daily"):
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO habits (name , frequency) VALUES (?,?)
        """, ( name , frequency))
        conn.commit()
    finally:
        conn.close()

    return f"Habit added:'{name}'({frequency})"

def log_habit(habit_name , completed = True):
    conn  = get_conn()
    try:
        rows = conn.execute("""
             SELECT id FROM habits WHERE name LIKE ?
        """, (f"%{habit_name}%",)).fetchone()
        if not rows:
            conn.close()
            return (f" habit: {habit_name} does not exist , ADD Habit")
        today = date.today().isoformat()
        existing = conn.execute("""
            SELECT id FROM habits_log WHERE habit_id = ? and date = ?
        """, (rows[0],today)).fetchone()
        if existing :
            conn.execute("""
                    UPDATE habits_log SET completed = ? WHERE id = ?
                """, (1 if completed else 0,existing[0]))
        else:
            conn.execute("""
                    INSERT INTO habits_log (habit_id,date,completed) VALUES (?,?,?)
            """, (rows[0],today,1 if completed else 0))
        conn.commit()
    finally:
        conn.close()

    status = "done" if completed else "skipped"
    return f"✅ '{habit_name}' logged as {status} for today."

def get_habit_summary():
    conn = get_conn()
    try:
        habit = conn.execute(" SELECT id , name , frequency from habits").fetchall()
        if not habit:
            return (f"habits does not exist yet.")
        today = date.today().isoformat()
        lines = ["**Habit check-in (last 7 days):**"]
        for hid , name , freq in habit:
            logs = conn.execute("""
                    SELECT date , completed FROM habits_log WHERE habit_id = ?
                    ORDER BY date DESC LIMIT 7
            """, (hid,)).fetchall()
            done_count = sum(1 for _,c in logs if c)
            log_dict = {d : c for d , c in logs }

            strip = ""
            for i in range(6, -1, -1):
                day = (date.today() - timedelta(days=i)).isoformat()
                if day in log_dict:
                    strip += "✅" if log_dict[day] else "❌"
                else:
                    strip += "⬜"
            lines.append(f"**{name}** ({freq}) — {done_count}/7 this week")
            lines.append(f"`{strip}`\n")
    finally:
        conn.close()

    return "\n\n".join(lines)


def add_goal(title , deadline=""):
    conn = get_conn()
    try:
        conn.execute("""
             INSERT INTO goals (title , deadline) VALUES (?,?)
        """, (title , deadline))
        conn.commit()
    finally:
        conn.close()
    result = f"Goal added :'{title}'"
    if deadline:
        result += f"- do before:{deadline}"
    return result

def update_goal_progress (goal_id , progress):
    progress = max(0 , min(100, int(progress)))
    conn = get_conn()
    try:
        done = (1 if progress == 100 else 0)
        conn.execute("""
            UPDATE goals SET progress = ? , done = ? WHERE id = ?   
        """,(progress , done , goal_id))
        conn.commit()
    finally:
        conn.close()

    return f"Goal id #{goal_id} progress update to {progress}%"

def get_goals():
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT id , title , deadline , progress , done  FROM goals ORDER BY done , deadline  
        """,).fetchall()
    finally:
        conn.close()

    if not rows :
        return "No goals set yet."

    lines = ["**Your goals:**"]
    for gid, title , deadline , progress , done in rows:
        bar = "█" * (progress // 10) + "░"* (10 - progress // 10)
        status = "✅" if done else "🎯"
        due = f"- target {deadline}" if deadline else ""
        lines.append(f"{status} [{gid}] **{title}**{due}\n`{bar}` {progress}%")
    return "\n".join(lines)

def get_briefing_data():
    overdue = get_overdue_task()
    conn = get_conn()
    try:
        today_tasks = conn.execute("""
              SElECT title , priority , deadline from tasks
              WHERE done = 0 ORDER BY 
              CASE priority 
              WHEN 'high' THEN 1  
              WHEN 'medium' THEN 2
              ELSE 3
              END LIMIT 5       
        """).fetchall()

        today = date.today().isoformat()
        all_habits = conn.execute("SELECT id , name FROM habits").fetchall()
        logged_today = {r[0] for r in conn.execute("""
              SELECT habit_id FROM habits_log WHERE date = ?      
        """,(today,)).fetchall()}
        pending_habits = [name for hid , name in all_habits if hid not in logged_today]

        goals = conn.execute("SELECT title, progress , deadline FROM goals WHERE done = 0").fetchall()
    finally:
        conn.close()

    return {
        "overdue_tasks": overdue or [],
        "top_tasks": today_tasks,
        "pending_habits": pending_habits,
        "goals": goals,
        "date": date.today().strftime("%A, %B %d, %Y")
    }

# def get_briefing_data():
#     """Collects all current state for the morning briefing."""
#     overdue = get_overdue_task()
#     conn = get_conn()
#
#     # Today's pending tasks
#     today_tasks = conn.execute("""
#         SELECT title, priority, deadline FROM tasks
#         WHERE done = 0 ORDER BY
#         CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
#         LIMIT 5
#     """).fetchall()
#
#     # Habits not yet logged today
#     today = date.today().isoformat()
#     all_habits = conn.execute("SELECT id, name FROM habits").fetchall()
#     logged_today = {r[0] for r in conn.execute("""
#         SELECT habit_id FROM habits_log WHERE date = ?
#     """, (today,)).fetchall()}
#     pending_habits = [name for hid, name in all_habits if hid not in logged_today]
#
#     # Active goals
#     goals = conn.execute("""
#         SELECT title, progress, deadline FROM goals WHERE done = 0
#     """).fetchall()
#
#     conn.close()
#
#     return {
#         "overdue_tasks": overdue or [],
#         "top_tasks": today_tasks,
#         "pending_habits": pending_habits,
#         "goals": goals,
#         "date": date.today().strftime("%A, %B %d %Y")
#     }