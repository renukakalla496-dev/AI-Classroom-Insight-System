
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ollama import chat


# =========================================================
# STEP 6 - GEMMA AI FUNCTION
# =========================================================

def generate_ai_insight(student):

    prompt = f"""
You are an AI classroom assistant.

Analyze the following student's performance:

Student Name: {student["Name"]}
Attendance: {student["Attendance"]}%
DBMS: {student["DBMS"]}
Python: {student["Python"]}
DSA: {student["DSA"]}
Average Score: {student["Average"]:.1f}

Give a short and simple analysis.

Include:
1. Student's performance
2. Main problem
3. Weakest subject
4. One recommendation

Use simple English.
Keep the answer within 80 words.
"""

    response = chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Classroom Insight",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Upload Data")

    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"]
    )

    st.divider()

    st.header("🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "🤖 AI Assistant",
            "👤 Student Profile",
            "📈 Predictions",
            "📥 Reports"
        ]
    )

    st.divider()

    st.info(
        "CSV columns:\n\n"
        "Name, Attendance, DBMS, Python, DSA"
    )


# =========================================================
# CHECK FILE
# =========================================================

if uploaded_file is None:

    st.title("🎓 AI Classroom Insight System")

    st.info(
        "👈 Upload a student CSV file from the sidebar to begin."
    )

    st.stop()


# =========================================================
# READ DATA
# =========================================================

df = pd.read_csv(uploaded_file)


# =========================================================
# VALIDATE COLUMNS
# =========================================================

required_columns = [
    "Name",
    "Attendance",
    "DBMS",
    "Python",
    "DSA"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {', '.join(missing_columns)}"
    )

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

numeric_columns = [
    "Attendance",
    "DBMS",
    "Python",
    "DSA"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=numeric_columns
)


# =========================================================
# CALCULATE PERFORMANCE
# =========================================================

df["Total"] = (
    df["DBMS"]
    + df["Python"]
    + df["DSA"]
)

df["Average"] = df["Total"] / 3


# =========================================================
# AT-RISK DETECTION
# =========================================================

df["Status"] = "Good Standing"

for index, student in df.iterrows():

    if (
        student["Attendance"] < 75
        or student["Average"] < 60
    ):

        df.loc[index, "Status"] = "At Risk"


# =========================================================
# CLASSROOM STATISTICS
# =========================================================

total_students = len(df)

average_attendance = df["Attendance"].mean()

average_dbms = df["DBMS"].mean()

average_python = df["Python"].mean()

average_dsa = df["DSA"].mean()

class_average = df["Average"].mean()

at_risk_count = len(
    df[df["Status"] == "At Risk"]
)



# =========================================================
# PREDICTION DATA
# =========================================================

prediction_df = df.copy()


# ---------------------------------------------------------
# ATTENDANCE RISK
# ---------------------------------------------------------

def attendance_prediction(attendance):

    if attendance < 60:

        return "🔴 High Risk"

    elif attendance < 75:

        return "🟠 Medium Risk"

    else:

        return "🟢 Low Risk"


prediction_df["Attendance Risk"] = (
    prediction_df["Attendance"]
    .apply(attendance_prediction)
)


# ---------------------------------------------------------
# PERFORMANCE PREDICTION
# ---------------------------------------------------------

def performance_prediction(
    average,
    attendance
):

    predicted_score = average

    if attendance >= 85:

        predicted_score += 5

    elif attendance >= 75:

        predicted_score += 2

    elif attendance < 60:

        predicted_score -= 5

    elif attendance < 75:

        predicted_score -= 2

    return max(
        0,
        min(
            100,
            predicted_score
        )
    )


prediction_df["Predicted Score"] = prediction_df.apply(
    lambda row:
    performance_prediction(
        row["Average"],
        row["Attendance"]
    ),
    axis=1
)


# ---------------------------------------------------------
# PREDICTION LEVEL
# ---------------------------------------------------------

prediction_df["Prediction Level"] = (
    prediction_df["Predicted Score"].apply(
        lambda score:
        "🟢 Good"
        if score >= 75
        else
        "🟠 Average"
        if score >= 60
        else
        "🔴 Needs Improvement"
    )
)


# =========================================================
# TOP STUDENT
# =========================================================

highest_student = df.loc[
    df["Average"].idxmax()
]


# =========================================================
# HEADER
# =========================================================

st.title("🎓 AI Classroom Insight System")

st.caption(
    "Smart classroom analytics for attendance, academic "
    "performance, prediction, and AI-based support."
)

st.divider()


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header("📊 Classroom Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👨‍🎓 Students",
            total_students
        )

    with col2:

        st.metric(
            "📅 Attendance",
            f"{average_attendance:.1f}%"
        )

    with col3:

        st.metric(
            "📚 Class Average",
            f"{class_average:.1f}"
        )

    with col4:

        st.metric(
            "⚠️ At Risk",
            at_risk_count
        )


    # -----------------------------------------------------
    # CLASSROOM INSIGHT
    # -----------------------------------------------------

    st.divider()

    st.header("🧠 Classroom Insight")

    if at_risk_count == 0:

        st.success(
            "🎉 All students are currently performing "
            "within the defined safe range."
        )

    else:

        st.warning(
            f"⚠️ {at_risk_count} student(s) may need "
            "additional academic or attendance support."
        )


    # -----------------------------------------------------
    # SUBJECT PERFORMANCE
    # -----------------------------------------------------

    st.subheader("📚 Subject Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "DBMS",
            f"{average_dbms:.1f}"
        )

    with col2:

        st.metric(
            "Python",
            f"{average_python:.1f}"
        )

    with col3:

        st.metric(
            "DSA",
            f"{average_dsa:.1f}"
        )


    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    st.divider()

    st.header("📈 Performance Analytics")

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Average Marks")

        subjects = [
            "DBMS",
            "Python",
            "DSA"
        ]

        averages = [
            average_dbms,
            average_python,
            average_dsa
        ]

        fig1, ax1 = plt.subplots(
            figsize=(5, 3)
        )

        ax1.bar(
            subjects,
            averages
        )

        ax1.set_ylabel("Marks")

        ax1.set_ylim(
            0,
            100
        )

        ax1.set_title(
            "Subject Performance"
        )

        plt.tight_layout()

        st.pyplot(
            fig1,
            use_container_width=True
        )


    with col2:

        st.subheader("Student Average")

        fig2, ax2 = plt.subplots(
            figsize=(5, 3)
        )

        ax2.bar(
            df["Name"],
            df["Average"]
        )

        ax2.set_ylabel(
            "Average Marks"
        )

        ax2.set_ylim(
            0,
            100
        )

        ax2.set_title(
            "Student Performance"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )


    # -----------------------------------------------------
    # ATTENDANCE CHART
    # -----------------------------------------------------

    st.subheader("📅 Attendance Analysis")

    fig3, ax3 = plt.subplots(
        figsize=(10, 3)
    )

    ax3.bar(
        df["Name"],
        df["Attendance"]
    )

    ax3.axhline(
        y=75,
        linestyle="--",
        label="75% Required"
    )

    ax3.set_ylabel(
        "Attendance (%)"
    )

    ax3.set_ylim(
        0,
        100
    )

    ax3.set_title(
        "Student Attendance"
    )

    ax3.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig3,
        use_container_width=True
    )


    # -----------------------------------------------------
    # AT-RISK STUDENTS
    # -----------------------------------------------------

    st.divider()

    st.header(
        "⚠️ Students Requiring Attention"
    )

    at_risk_students = df[
        df["Status"] == "At Risk"
    ]

    if len(at_risk_students) > 0:

        st.warning(
            f"{len(at_risk_students)} student(s) "
            "have been identified."
        )

        display_columns = [
            "Name",
            "Attendance",
            "Average",
            "Status"
        ]

        st.dataframe(
            at_risk_students[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "🎉 No students currently require attention."
        )


    # -----------------------------------------------------
    # TOP PERFORMER
    # -----------------------------------------------------

    st.divider()

    st.header("🏆 Top Performer")

    st.success(
        f"{highest_student['Name']} has the highest "
        f"average score of "
        f"{highest_student['Average']:.1f}."
    )


# =========================================================
# AI ASSISTANT
# =========================================================

elif page == "🤖 AI Assistant":

    st.header("🤖 Teacher AI Assistant")

    st.write(
        "Ask questions about your classroom data."
    )


    classroom_data = f"""
Classroom Summary:

Total Students: {total_students}

Average Attendance: {average_attendance:.1f}%

Class Average: {class_average:.1f}

Average DBMS Marks: {average_dbms:.1f}

Average Python Marks: {average_python:.1f}

Average DSA Marks: {average_dsa:.1f}

Students At Risk: {at_risk_count}

Student Details:

{df[
    [
        "Name",
        "Attendance",
        "DBMS",
        "Python",
        "DSA",
        "Average",
        "Status"
    ]
].to_string(index=False)}
"""


    teacher_question = st.text_input(
        "💬 Ask the AI",
        placeholder="Example: Who are the students at risk?"
    )


    if st.button(
        "🤖 Ask AI"
    ):

        if teacher_question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            prompt = f"""
You are an AI assistant for a college teacher.

Use ONLY the classroom data below.

{classroom_data}

Teacher's Question:
{teacher_question}

Instructions:

- Give a clear answer.
- Use simple English.
- Mention student names when relevant.
- Use actual numbers.
- Do not invent information.
- Keep the answer below 150 words.
"""


            try:

                with st.spinner(
                    "🤖 AI is analyzing..."
                ):

                    response = chat(
                        model="gemma3:1b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )


                st.subheader(
                    "💡 AI Response"
                )

                st.write(
                    response[
                        "message"
                    ]["content"]
                )


            except Exception as e:

                st.error(
                    f"AI Assistant Error: {e}"
                )


    # -----------------------------------------------------
    # AI QUICK QUESTIONS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "💡 Example Questions"
    )

    st.write(
        "• Who are the students at risk?"
    )

    st.write(
        "• Which subject has the lowest average?"
    )

    st.write(
        "• Who is the top performer?"
    )

    st.write(
        "• Give me a summary of the class."
    )

    st.write(
        "• How can I improve student performance?"
    )


# =========================================================
# STUDENT PROFILE
# =========================================================

elif page == "👤 Student Profile":

    st.header("👤 Student Profile")

    student_names = df[
        "Name"
    ].tolist()

    selected_student = st.selectbox(
        "🔍 Select a student",
        student_names
    )


    student = df[
        df["Name"] == selected_student
    ].iloc[0]


    st.subheader(
        f"📋 {student['Name']}"
    )


    # -----------------------------------------------------
    # STUDENT METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📅 Attendance",
            f"{student['Attendance']:.1f}%"
        )


    with col2:

        st.metric(
            "📊 Average",
            f"{student['Average']:.1f}"
        )


    with col3:

        st.metric(
            "📚 Total Marks",
            f"{student['Total']:.1f}"
        )


    with col4:

        if student["Status"] == "At Risk":

            st.metric(
                "⚠️ Status",
                "At Risk"
            )

        else:

            st.metric(
                "✅ Status",
                "Good"
            )


    # -----------------------------------------------------
    # SUBJECT SCORES
    # -----------------------------------------------------

    st.subheader(
        "📚 Subject Scores"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "DBMS",
            f"{student['DBMS']:.1f}"
        )


    with col2:

        st.metric(
            "Python",
            f"{student['Python']:.1f}"
        )


    with col3:

        st.metric(
            "DSA",
            f"{student['DSA']:.1f}"
        )


    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "🤖 AI Analysis"
    )


    if st.button(
        "Generate AI Analysis",
        key="student_analysis"
    ):

        try:

            with st.spinner(
                "🤖 Gemma is analyzing..."
            ):

                student_ai_result = (
                    generate_ai_insight(
                        student
                    )
                )


            st.success(
                "AI analysis generated!"
            )

            st.write(
                student_ai_result
            )


        except Exception as e:

            st.error(
                f"AI analysis failed: {e}"
            )


# =========================================================
# STEP 11 + 12 - PREDICTIONS
# =========================================================

elif page == "📈 Predictions":

    st.header("📈 Student Predictions")

    st.info(
        "These are rule-based estimates using the current "
        "student data. Historical data is required for "
        "true machine-learning predictions."
    )


    # =====================================================
    # STEP 11 - ATTENDANCE RISK PREDICTION
    # =====================================================

    st.subheader(
        "📅 Attendance Risk Prediction"
    )

    prediction_df = df.copy()


    def attendance_prediction(attendance):

        if attendance < 60:

            return "🔴 High Risk"

        elif attendance < 75:

            return "🟠 Medium Risk"

        else:

            return "🟢 Low Risk"


    prediction_df[
        "Attendance Risk"
    ] = prediction_df[
        "Attendance"
    ].apply(
        attendance_prediction
    )


    st.dataframe(
        prediction_df[
            [
                "Name",
                "Attendance",
                "Attendance Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # STEP 12 - PERFORMANCE PREDICTION
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Performance Prediction"
    )


    def performance_prediction(
        average,
        attendance
    ):

        # Estimate based on current performance
        # and attendance.

        predicted_score = average

        if attendance >= 85:

            predicted_score += 5

        elif attendance >= 75:

            predicted_score += 2

        elif attendance < 60:

            predicted_score -= 5

        elif attendance < 75:

            predicted_score -= 2


        predicted_score = max(
            0,
            min(
                100,
                predicted_score
            )
        )

        return predicted_score


    prediction_df[
        "Predicted Score"
    ] = prediction_df.apply(
        lambda row:
        performance_prediction(
            row["Average"],
            row["Attendance"]
        ),
        axis=1
    )


    prediction_df[
        "Prediction Level"
    ] = prediction_df[
        "Predicted Score"
    ].apply(
        lambda score:
        "🟢 Good"
        if score >= 75
        else
        "🟠 Average"
        if score >= 60
        else
        "🔴 Needs Improvement"
    )


    st.dataframe(
        prediction_df[
            [
                "Name",
                "Average",
                "Attendance",
                "Predicted Score",
                "Prediction Level"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # PREDICTION CHART
    # -----------------------------------------------------

    st.subheader(
        "📈 Current vs Predicted Performance"
    )


    fig4, ax4 = plt.subplots(
        figsize=(10, 4)
    )


    ax4.plot(
        prediction_df["Name"],
        prediction_df["Average"],
        marker="o",
        label="Current Average"
    )


    ax4.plot(
        prediction_df["Name"],
        prediction_df["Predicted Score"],
        marker="o",
        linestyle="--",
        label="Predicted Score"
    )


    ax4.set_ylabel(
        "Score"
    )

    ax4.set_ylim(
        0,
        100
    )

    ax4.set_title(
        "Current vs Predicted Performance"
    )

    ax4.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig4,
        use_container_width=True
    )


# =========================================================
# REPORTS
# =========================================================

elif page == "📥 Reports":

    st.header("📥 Download Reports")

    st.write(
        "Download classroom analysis for future reference."
    )


    # -----------------------------------------------------
    # COMPLETE REPORT
    # -----------------------------------------------------

    report_columns = [
        "Name",
        "Attendance",
        "DBMS",
        "Python",
        "DSA",
        "Total",
        "Average",
        "Status"
    ]


    report_df = df[
        report_columns
    ]


    csv_report = report_df.to_csv(
        index=False
    )


    st.download_button(
        label="📊 Download Complete Student Report",
        data=csv_report,
        file_name="classroom_analysis.csv",
        mime="text/csv"
    )


    # -----------------------------------------------------
    # AT-RISK REPORT
    # -----------------------------------------------------

    at_risk_report = df[
        df["Status"] == "At Risk"
    ][
        report_columns
    ]


    at_risk_csv = at_risk_report.to_csv(
        index=False
    )


    st.download_button(
        label="⚠️ Download At-Risk Student Report",
        data=at_risk_csv,
        file_name="at_risk_students.csv",
        mime="text/csv"
    )


    # -----------------------------------------------------
    # PREDICTION REPORT
    # -----------------------------------------------------

    prediction_report = prediction_df[
        [
            "Name",
            "Attendance",
            "Average",
            "Attendance Risk",
            "Predicted Score",
            "Prediction Level"
        ]
    ]


    prediction_csv = prediction_report.to_csv(
        index=False
    )


    st.download_button(
        label="📈 Download Prediction Report",
        data=prediction_csv,
        file_name="student_predictions.csv",
        mime="text/csv"
    )


# =========================================================
# COMPLETE DATA
# =========================================================

st.divider()

with st.expander(
    "📋 View Complete Student Data"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Classroom Insight System • "
    "Student Performance Analytics • "
    "Ollama + Gemma"
)
