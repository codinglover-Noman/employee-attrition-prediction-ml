import gradio as gr
import pandas as pd
import joblib


# Load trained pipeline
model = joblib.load("employee_attrition_pipeline.pkl")


def predict_attrition(
    Age,
    BusinessTravel,
    DailyRate,
    Department,
    DistanceFromHome,
    Education,
    EducationField,
    EnvironmentSatisfaction,
    Gender,
    HourlyRate,
    JobInvolvement,
    JobLevel,
    JobRole,
    JobSatisfaction,
    MaritalStatus,
    MonthlyIncome,
    MonthlyRate,
    NumCompaniesWorked,
    OverTime,
    PercentSalaryHike,
    PerformanceRating,
    RelationshipSatisfaction,
    StockOptionLevel,
    TotalWorkingYears,
    TrainingTimesLastYear,
    WorkLifeBalance,
    YearsAtCompany,
    YearsInCurrentRole,
    YearsSinceLastPromotion,
    YearsWithCurrManager
):

    # Create dataframe from user input
    input_data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }])


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    if prediction == 1:
        result = "Employee is likely to leave the company"
    else:
        result = "Employee is likely to stay"


    return f"{result}\n\nAttrition probability: {probability:.2%}"


# Gradio interface

demo = gr.Interface(
    fn=predict_attrition,

    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
            label="Business Travel"
        ),
        gr.Number(label="Daily Rate"),
        gr.Dropdown(
            ["Sales", "Research & Development", "Human Resources"],
            label="Department"
        ),
        gr.Number(label="Distance From Home"),
        gr.Number(label="Education"),
        gr.Dropdown(
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ],
            label="Education Field"
        ),
        gr.Number(label="Environment Satisfaction"),
        gr.Dropdown(
            ["Male", "Female"],
            label="Gender"
        ),
        gr.Number(label="Hourly Rate"),
        gr.Number(label="Job Involvement"),
        gr.Number(label="Job Level"),
        gr.Dropdown(
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ],
            label="Job Role"
        ),
        gr.Number(label="Job Satisfaction"),
        gr.Dropdown(
            ["Single", "Married", "Divorced"],
            label="Marital Status"
        ),
        gr.Number(label="Monthly Income"),
        gr.Number(label="Monthly Rate"),
        gr.Number(label="Number of Companies Worked"),
        gr.Dropdown(
            ["Yes", "No"],
            label="Over Time"
        ),
        gr.Number(label="Percent Salary Hike"),
        gr.Number(label="Performance Rating"),
        gr.Number(label="Relationship Satisfaction"),
        gr.Number(label="Stock Option Level"),
        gr.Number(label="Total Working Years"),
        gr.Number(label="Training Times Last Year"),
        gr.Number(label="Work Life Balance"),
        gr.Number(label="Years At Company"),
        gr.Number(label="Years In Current Role"),
        gr.Number(label="Years Since Last Promotion"),
        gr.Number(label="Years With Current Manager")
    ],

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="Employee Attrition Prediction System",

    description="Machine learning model predicting employee attrition probability."
)


# Launch app

if __name__ == "__main__":
    demo.launch()