
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Employee Engagement & Retention System",
    page_icon="👥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = (
    BASE_DIR / "Employee Engagement System" /
    "Dataset" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Load trained model and threshold
@st.cache_resource
def load_model():
    with open(MODEL_DIR / "best_attrition_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "best_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    return model, float(threshold)

model, threshold = load_model()
df = pd.read_csv(DATA_PATH)

# Dashboard heading
st.title("👥 Employee Engagement & Retention System")
st.caption("AI-powered employee attrition analysis and engagement indicators")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Dashboard", "Employee Attrition Prediction", "Engagement Analysis"]
)

if page == "Dashboard":
    st.header("Organization Overview")

    total = len(df)
    left = (df["Attrition"] == "Yes").sum()
    stayed = total - left

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", total)
    c2.metric("Employees Left", int(left))
    c3.metric("Historical Attrition Rate", f"{left / total * 100:.1f}%")

    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        [stayed, left],
        labels=["Stayed", "Left"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.info(
        "This dashboard uses the IBM HR Analytics practice dataset. "
        "It is a demonstration dataset, not live company employee data."
    )

elif page == "Employee Attrition Prediction":
    st.header("Employee Attrition Risk Prediction")
    st.write("Enter employee details to estimate attrition risk.")

    with st.form("employee_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 65, 30)
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            environment_satisfaction = st.selectbox(
                "Environment Satisfaction", [1, 2, 3, 4]
            )
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            overtime = st.selectbox("OverTime", ["Yes", "No"])

        with col2:
            years_at_company = st.number_input("Years at Company", 0, 40, 3)
            years_with_manager = st.number_input(
                "Years With Current Manager", 0, 20, 2
            )
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced"]
            )
            department = st.selectbox(
                "Department",
                ["Sales", "Research & Development", "Human Resources"]
            )
            business_travel = st.selectbox(
                "Business Travel",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
            )

        submitted = st.form_submit_button("Predict Attrition Risk")

    if submitted:
        # Start with a valid dataset row so all model features are present
        employee = df.drop(columns=["Attrition"]).iloc[[0]].copy()

        # Replace entered values
        values = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "OverTime": overtime,
            "YearsAtCompany": years_at_company,
            "YearsWithCurrManager": years_with_manager,
            "WorkLifeBalance": work_life_balance,
            "StockOptionLevel": stock_option,
            "MaritalStatus": marital_status,
            "Department": department,
            "BusinessTravel": business_travel
        }

        for column, value in values.items():
            employee.loc[:, column] = value

        try:
            probability = model.predict_proba(employee)[0][1]

            st.subheader("Prediction Result")
            st.metric("Estimated Attrition Probability", f"{probability * 100:.1f}%")

            if probability >= threshold:
                st.warning("Higher predicted attrition risk")
                st.write(
                    "Consider a supportive conversation to understand "
                    "the employee's workplace concerns."
                )
            else:
                st.success("Lower predicted attrition risk")

            st.caption(
                f"Classification threshold: {threshold:.2f}. "
                "This is a model estimate, not a certainty."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")


elif page == "Engagement Analysis":
    st.header("Employee Engagement Analysis")

    st.write(
        "Analysis of employee satisfaction and workplace indicators "
        "to understand employee engagement patterns."
    )

    col1, col2 = st.columns(2)

    # Job Satisfaction
    with col1:
        st.subheader("Job Satisfaction")

        satisfaction = df["JobSatisfaction"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(satisfaction.index, satisfaction.values)
        ax.set_xlabel("Job Satisfaction Rating (1–4)")
        ax.set_ylabel("Number of Employees")
        ax.set_title("Job Satisfaction Distribution")
        ax.set_xticks([1, 2, 3, 4])

        st.pyplot(fig)
        plt.close(fig)

    # Environment Satisfaction
    with col2:
        st.subheader("Environment Satisfaction")

        environment = (
            df["EnvironmentSatisfaction"].value_counts().sort_index()
        )

        fig, ax = plt.subplots()
        ax.bar(environment.index, environment.values)
        ax.set_xlabel("Environment Satisfaction Rating (1–4)")
        ax.set_ylabel("Number of Employees")
        ax.set_title("Workplace Environment Satisfaction")
        ax.set_xticks([1, 2, 3, 4])

        st.pyplot(fig)
        plt.close(fig)

    # Overtime Distribution
    st.subheader("Overtime Distribution")

    overtime = df["OverTime"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(overtime.index, overtime.values)
    ax.set_xlabel("Overtime Status")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Employee Overtime Distribution")

    st.pyplot(fig)
    plt.close(fig)

    # Department Analysis
    st.subheader("Department-wise Satisfaction Analysis")

    department_data = df.groupby("Department")[
        ["JobSatisfaction", "EnvironmentSatisfaction", "JobInvolvement"]
    ].mean()

    st.dataframe(department_data.round(2), use_container_width=True)
