import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #343a40 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar select boxes */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Cards and metrics */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4e73df;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4e73df;
        color: white !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4e73df;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    
    .stButton button:hover {
        background-color: #2e59d9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Check for xlsxwriter
try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False

# Main app
st.title("🚀 Employee Attrition Analysis Dashboard")
st.markdown("Analyze attrition risks and develop retention strategies")

# File upload in sidebar
# Styled file upload section in sidebar
with st.sidebar:
    st.markdown("""
        <style>
            .custom-upload-box {
                background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 40%, #fad0c4 60%, #fbc2eb 100%);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 20px;
                margin-top: 10px;
                margin-bottom: 20px;
                color: #ffffffcc;
                text-align: center;
                font-size: 14px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: 0.3s ease;
            }
            .custom-upload-box:hover {
                background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
                border-color: #fff;
            }
        </style>

        <div class="custom-upload-box">
            <h4 style="margin-bottom: 10px;">📂 Upload CSV File</h4>
            <p style="margin: 0;">Dataset must include:</p>
            <p><code>EmployeeNumber</code>, <code>Attrition</code>, <code>JobRole</code>, <code>Department</code></p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",  # Hide default label
        type=["csv"]
    )

    if uploaded_file:
        st.success("✅ File uploaded successfully!", icon="📁")
    else:
        st.warning("⚠️ Please upload a CSV file to continue.")
        st.stop()




# Data processing function
@st.cache_data
def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    required_cols = ['EmployeeNumber', 'Attrition', 'JobRole', 'Department']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {', '.join(required_cols)}")
        st.stop()
    
    df_reason = df.copy()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    employee_info = df[['EmployeeNumber', 'JobRole', 'Department']].copy()

    for col in ['EmployeeCount', 'StandardHours', 'Over18']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(['Attrition'], axis=1)
    y = df['Attrition']

    model = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        min_samples_split=5, random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)

    employee_info['Attrition_Probability (%)'] = model.predict_proba(X)[:, 1] * 100
    employee_info['RiskLevel'] = pd.cut(employee_info['Attrition_Probability (%)'], 
                                      bins=[0, 30, 70, 100], 
                                      labels=['Low', 'Medium', 'High'])
    employee_info = employee_info.sort_values(by='Attrition_Probability (%)', ascending=False).reset_index(drop=True)
    
    return df_reason, employee_info, model, X

# Process the data
df_reason, employee_info, model, X = process_data(uploaded_file)

# Filters in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    selected_depts = st.multiselect(
        "Select Departments", 
        options=employee_info['Department'].unique(), 
        default=employee_info['Department'].unique()
    )
    selected_roles = st.multiselect(
        "Select Job Roles", 
        options=employee_info['JobRole'].unique(), 
        default=employee_info['JobRole'].unique()
    )
    risk_levels = st.multiselect(
        "Select Risk Levels", 
        options=['Low', 'Medium', 'High'], 
        default=['High', 'Medium']
    )
    prob_range = st.slider(
        "Attrition Probability Range (%)", 
        0, 100, (50, 100)
    )

# Filter data based on selections
filtered_data = employee_info[
    (employee_info['Department'].isin(selected_depts)) &
    (employee_info['JobRole'].isin(selected_roles)) &
    (employee_info['RiskLevel'].isin(risk_levels)) &
    (employee_info['Attrition_Probability (%)'].between(prob_range[0], prob_range[1]))
]

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Employee List", "📊 Analytics", "🧪 What-If", "👤 Employee Profile"])

with tab1:
    st.subheader("Employee Attrition Risk Assessment")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(employee_info))
    col2.metric("High Risk Employees", len(employee_info[employee_info['RiskLevel'] == 'High']), 
                f"{len(employee_info[employee_info['RiskLevel'] == 'High'])/len(employee_info):.1%} of total")
    col3.metric("Average Risk", f"{employee_info['Attrition_Probability (%)'].mean():.1f}%")

    # Employee data table
    st.dataframe(
        filtered_data,
        column_config={
            "EmployeeNumber": st.column_config.NumberColumn("ID", format="%d"),
            "JobRole": "Job Role",
            "Department": "Department",
            "Attrition_Probability (%)": st.column_config.ProgressColumn(
                "Risk %", 
                format="%.1f%%", 
                min_value=0, 
                max_value=100
            ),
            "RiskLevel": st.column_config.TextColumn(
                "Risk Level", 
                help="High (≥70%), Medium (30-69%), Low (<30%)"
            )
        },
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data", 
        data=csv, 
        file_name="attrition_risk.csv", 
        mime="text/csv"
    )

with tab2:
    st.subheader("Attrition Risk Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        # Box plot by department
        fig1 = px.box(
            filtered_data, 
            x='Department', 
            y='Attrition_Probability (%)', 
            color='Department',
            title='Risk Distribution by Department',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Feature importance
        importance = pd.DataFrame({
            'Feature': X.columns, 
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig3 = px.bar(
            importance, 
            x='Importance', 
            y='Feature', 
            title='Top 10 Influencing Factors', 
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Pie chart of risk levels
        fig2 = px.pie(
            filtered_data, 
            names='RiskLevel', 
            title='Risk Level Distribution', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Treemap
        fig4 = px.treemap(
            filtered_data, 
            path=['Department', 'JobRole'], 
            values='EmployeeNumber', 
            color='Attrition_Probability (%)', 
            color_continuous_scale='RdYlGn_r', 
            title='Risk by Department/Job Role'
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Scenario Analysis")
    st.markdown("Simulate how changes would affect an employee's attrition risk")

    selected_emp = st.selectbox(
        "Select Employee", 
        options=filtered_data['EmployeeNumber'].tolist()
    )
    emp_data = df_reason[df_reason['EmployeeNumber'] == selected_emp].iloc[0]
    current_prob = filtered_data.loc[filtered_data['EmployeeNumber'] == selected_emp, 'Attrition_Probability (%)'].values[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Risk", f"{current_prob:.1f}%")
        new_job_sat = st.slider(
            "Job Satisfaction (1-4)", 
            1, 4, 
            value=int(emp_data.get('JobSatisfaction', 3))
        )
    with col2:
        new_income = st.number_input(
            "Monthly Income ($)", 
            value=int(emp_data.get('MonthlyIncome', 5000))
        )
        new_training = st.slider(
            "Training Sessions Last Year", 
            0, 6, 
            value=int(emp_data.get('TrainingTimesLastYear', 2))
        )

    if st.button("Calculate New Risk"):
        modified_data = X[df_reason['EmployeeNumber'] == selected_emp].copy()
        if 'JobSatisfaction' in modified_data.columns:
            modified_data['JobSatisfaction'] = new_job_sat
        if 'MonthlyIncome' in modified_data.columns:
            modified_data['MonthlyIncome'] = new_income
        if 'TrainingTimesLastYear' in modified_data.columns:
            modified_data['TrainingTimesLastYear'] = new_training

        new_prob = model.predict_proba(modified_data)[0, 1] * 100
        st.success(f"""
        **Risk Change Prediction**  
        - Original Risk: {current_prob:.1f}%  
        - New Estimated Risk: {new_prob:.1f}%  
        - Change: {new_prob - current_prob:+.1f}%
        """)

    with st.expander("💰 Cost Analysis", expanded=False):
        avg_salary = st.number_input(
            "Average Annual Salary ($)", 
            value=65000, 
            min_value=30000, 
            max_value=200000
        )
        replacement_cost_pct = st.slider(
            "Replacement Cost (% of salary)", 
            50, 200, 100
        )
        high_risk_count = len(filtered_data[filtered_data['RiskLevel'] == 'High'])
        potential_cost = high_risk_count * (avg_salary * (replacement_cost_pct / 100))
        st.metric(
            "Potential Attrition Cost", 
            f"${potential_cost:,.0f}", 
            delta=f"{high_risk_count} high-risk employees"
        )

    with st.expander("📄 Generate Report", expanded=False):
        if XLSXWRITER_AVAILABLE:
            if st.button("Generate Excel Report"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    filtered_data.to_excel(writer, sheet_name='Risk Assessment', index=False)
                    analysis = filtered_data.groupby(['Department', 'JobRole']).agg({
                        'Attrition_Probability (%)': ['mean', 'count']
                    })
                    analysis.to_excel(writer, sheet_name='Analysis')
                    importance.to_excel(writer, sheet_name='Key Factors')
                
                st.download_button(
                    "⬇️ Download Report", 
                    data=output.getvalue(), 
                    file_name="attrition_report.xlsx", 
                    mime="application/vnd.ms-excel"
                )
        else:
            st.warning("Excel report generation requires xlsxwriter package. Install with: `pip install xlsxwriter`")

with tab4:
    st.subheader("Employee Profile Analysis")
    selected_emp_profile = st.selectbox(
        "Select Employee to View Profile", 
        options=filtered_data['EmployeeNumber'].tolist(),
        key='profile_select'
    )
    emp_profile_data = df_reason[df_reason['EmployeeNumber'] == selected_emp_profile].iloc[0]
    
    st.markdown("### 📊 Employee Profile Breakdown")

    # Features to show in chart
    chart_features = {
        'JobSatisfaction': 'Job Satisfaction',
        'EnvironmentSatisfaction': 'Environment Satisfaction',
        'RelationshipSatisfaction': 'Relationship Satisfaction',
        'WorkLifeBalance': 'Work-Life Balance',
        'PerformanceRating': 'Performance Rating'
    }

    # Extract values (with fallback to 0 if feature doesn't exist)
    values = [emp_profile_data.get(k, 0) for k in chart_features.keys()]
    labels = list(chart_features.values())

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=values,
        theta=labels,
        marker_color='#4e73df',
        marker_line_color='#2e59d9',
        marker_line_width=1.5,
        opacity=0.8
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 5],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title="Employee Ratings Overview",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional employee details
    st.markdown("### Employee Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Job Role", emp_profile_data.get('JobRole', 'N/A'))
        st.metric("Department", emp_profile_data.get('Department', 'N/A'))
        st.metric("Years at Company", emp_profile_data.get('YearsAtCompany', 'N/A'))
    with col2:
        st.metric(
            "Monthly Income", 
            f"${emp_profile_data.get('MonthlyIncome', 0):,.0f}" if emp_profile_data.get('MonthlyIncome') else 'N/A'
        )
        st.metric("Age", emp_profile_data.get('Age', 'N/A'))
        st.metric("Total Working Years", emp_profile_data.get('TotalWorkingYears', 'N/A'))