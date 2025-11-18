import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

# ----------------------------
# Load data (same as original)
# ----------------------------
attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

# Try to parse Date column if present
if 'Date' in attendance_df.columns:
    try:
        attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])
    except Exception:
        pass

# ----------------------------
# Helper functions for sidebar
# ----------------------------
def compute_absence_rate(att_df, start_date=None, end_date=None):
    df = att_df.copy()
    if 'Date' in df.columns and start_date and end_date:
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        df = df.loc[mask]
    if 'Status' in df.columns:
        rate = df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate')
    else:
        unique = df['StudentID'].unique() if 'StudentID' in df.columns else []
        rate = pd.DataFrame({'StudentID': unique, 'AbsenceRate': 0.0})
    return rate

def build_students_df(att_df, events_df, lms_df):
    # combine unique IDs from available sources
    parts = []
    if 'StudentID' in att_df.columns:
        parts.append(att_df['StudentID'])
    if 'StudentID' in events_df.columns:
        parts.append(events_df['StudentID'])
    if 'StudentID' in lms_df.columns:
        parts.append(lms_df['StudentID'])
    if parts:
        all_ids = pd.Series(pd.concat(parts).unique())
    else:
        all_ids = pd.Series(dtype=str)
    students = pd.DataFrame({'StudentID': all_ids.astype(str)})
    # attach optional metadata if present in attendance or events
    for df in (att_df, events_df, lms_df):
        if 'StudentName' in df.columns:
            meta = df[['StudentID','StudentName','Class','Department']].drop_duplicates('StudentID').set_index('StudentID')
            students = students.join(meta, on='StudentID')
            break
    return students

students_df = build_students_df(attendance_df, events_df, lms_df)
students_df['StudentID'] = students_df['StudentID'].astype(str)
students_df = students_df.set_index('StudentID', drop=False)

# ----------------------------
# Improved Sidebar (replaces simple multiselect)
# ----------------------------
st.sidebar.header("🔍 Student filters (improved)")

filter_mode = st.sidebar.radio("Filter mode", (
    "By list / search",
    "By class / department",
    "By date range & absence rate",
    "Presets (quick)"
))

search_text = st.sidebar.text_input("Search (ID or name)", value="")

# Quick selection controls
select_all = st.sidebar.checkbox("Select all candidates", value=False)
clear_selection = st.sidebar.button("Clear selection")

# Date range (if available)
min_date = attendance_df['Date'].min() if 'Date' in attendance_df.columns else None
max_date = attendance_df['Date'].max() if 'Date' in attendance_df.columns else None

date_range = None
if filter_mode == "By date range & absence rate" and min_date is not None:
    date_range = st.sidebar.date_input("Attendance Date range", value=(min_date.date(), max_date.date()))
    absence_df = compute_absence_rate(attendance_df, start_date=date_range[0], end_date=date_range[1])
else:
    absence_df = compute_absence_rate(attendance_df)

# Absence slider (used in date-range and presets)
absence_filter = None
if filter_mode in ("By date range & absence rate", "Presets (quick)"):
    absence_filter = st.sidebar.slider("Absence rate range", 0.0, 1.0, (0.0, 0.5), step=0.01)

# Class / Department options (if present)
class_options = []
dept_options = []
if 'Class' in students_df.columns and students_df['Class'].notna().any():
    class_options = sorted(students_df['Class'].dropna().unique().tolist())
if 'Department' in students_df.columns and students_df['Department'].notna().any():
    dept_options = sorted(students_df['Department'].dropna().unique().tolist())

selected_classes = []
selected_depts = []
if filter_mode == "By class / department":
    if class_options:
        selected_classes = st.sidebar.multiselect("Select Class(es)", class_options, default=class_options)
    if dept_options:
        selected_depts = st.sidebar.multiselect("Select Department(s)", dept_options, default=dept_options)

# Presets quick selection
preset = None
if filter_mode == "Presets (quick)":
    preset = st.sidebar.selectbox("Choose preset", ["Top active (lowest absence)", "At-risk (high absence)", "Most LMS-active"])

# Build candidate list
candidates = students_df.copy()

# Apply search
if search_text:
    q = search_text.strip().lower()
    mask_id = candidates['StudentID'].str.lower().str.contains(q)
    name_mask = False
    if 'StudentName' in candidates.columns:
        name_mask = candidates['StudentName'].fillna('').str.lower().str.contains(q)
    candidates = candidates[mask_id | name_mask]

# Apply class/dept filters
if filter_mode == "By class / department":
    if selected_classes:
        candidates = candidates[candidates['Class'].isin(selected_classes)]
    if selected_depts:
        candidates = candidates[candidates['Department'].isin(selected_depts)]

# Join absence rates
candidates = candidates.reset_index(drop=True).set_index('StudentID', drop=False)
candidates = candidates.join(absence_df.set_index('StudentID'), how='left')
candidates['AbsenceRate'] = candidates['AbsenceRate'].fillna(0.0)

# Apply absence slider filter if present
if absence_filter:
    lo, hi = absence_filter
    candidates = candidates[(candidates['AbsenceRate'] >= lo) & (candidates['AbsenceRate'] <= hi)]

# Apply presets logic
if preset == "Top active (lowest absence)":
    candidates = candidates.sort_values('AbsenceRate', ascending=True).head(50)
elif preset == "At-risk (high absence)":
    candidates = candidates.sort_values('AbsenceRate', ascending=False).head(50)
elif preset == "Most LMS-active":
    if 'StudentID' in lms_df.columns and 'PagesViewed' in lms_df.columns:
        lms_avg = lms_df.groupby('StudentID')[['SessionDuration','PagesViewed']].mean().reset_index().set_index('StudentID')
        candidates = candidates.join(lms_avg, how='left').fillna(0)
        candidates = candidates.sort_values('PagesViewed', ascending=False).head(50)

student_options = candidates.index.astype(str).tolist()

# Default selection logic
default_selection = student_options if select_all else []
if clear_selection:
    default_selection = []

selected_students = st.sidebar.multiselect("Select Students (final)", student_options, default=default_selection)

# Preview expandable
with st.sidebar.expander("Preview candidates & stats", expanded=False):
    st.write(f"Candidates: {len(student_options)}")
    st.write(f"Selected: {len(selected_students)}")

    preview_cols = ['StudentID']
    if 'StudentName' in candidates.columns:
        preview_cols.append('StudentName')
    if 'Class' in candidates.columns:
        preview_cols.append('Class')
    if 'Department' in candidates.columns:
        preview_cols.append('Department')
    preview_cols.append('AbsenceRate')

    preview_df = candidates.copy()
    preview_df.index.name = None
    preview_df = preview_df[preview_cols].head(10)

    st.dataframe(preview_df)


if not selected_students:
    # fallback: if nothing selected, use all candidates
    selected_students = student_options

# ----------------------------
# Use selected_students for filtering (identical to your original usage)
# ----------------------------
filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# ----------------------------
# Rest of your original dashboard (kept same)
# ----------------------------
st.title("📊 Smart Campus Insights")
st.sidebar.header("🔍 Filters")  # kept for visual continuity (main sidebar already used above)

st.subheader("📋 Attendance Trends")
# compute summary similar to your original code
if 'Date' in filtered_attendance.columns and 'Status' in filtered_attendance.columns:
    attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
    st.line_chart(attendance_summary)
else:
    st.write("Attendance data missing 'Date' or 'Status' column for trend chart.")

st.subheader("🎓 Event Participation")
if 'EventName' in filtered_events.columns:
    event_counts = filtered_events['EventName'].value_counts()
    st.bar_chart(event_counts)
else:
    st.write("No 'EventName' column in events data.")

st.subheader("💻 LMS Usage Patterns")
if 'StudentID' in filtered_lms.columns and 'SessionDuration' in filtered_lms.columns and 'PagesViewed' in filtered_lms.columns:
    lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
    st.dataframe(lms_summary)
else:
    st.write("LMS data missing expected columns.")

st.subheader("🤖 Predict Student Engagement Risk")

# Original ML pipeline (kept as-is, using full attendance & lms like original)
ml_data = pd.merge(
    attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
    lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
    on='StudentID'
)

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y = ml_data['Engagement']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Performance:")
st.text(classification_report(y_test, y_pred))

st.subheader("📈 Predict Engagement for New Student")
absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

if st.button("Predict Engagement"):
    try:
        prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
        result = "Engaged" if prediction[0] == 1 else "At Risk"
        st.success(f"Predicted Engagement Status: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# End of app
