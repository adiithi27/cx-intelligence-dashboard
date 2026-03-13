import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="CX Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------

st.markdown("""
<style>

.insight-box{
background-color:#0b1a33;
padding:18px;
border-radius:10px;
border:1px solid #2563eb;
margin-top:10px;
margin-bottom:25px;
}

.insight-title{
font-size:12px;
font-weight:700;
color:#60a5fa;
margin-bottom:6px;
}

.insight-text{
font-size:14px;
color:#e5e7eb;
line-height:1.6;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR COLLAPSIBLE NAVIGATION
# ---------------------------------------------------
# ---------------------------------------------------
# SIDEBAR COLLAPSIBLE NAVIGATION
# ---------------------------------------------------

st.sidebar.title("CX Intelligence Platform")

page = None

with st.sidebar.expander("Dashboard Overview", expanded=True):

    if st.button("Overview"):
        page = "overview"

with st.sidebar.expander("EDA"):

    if st.button("EDA Overview"):
        page = "eda"

    if st.button("Correlation Analysis"):
        page = "correlation"

with st.sidebar.expander("Customer Insights"):

    if st.button("User Segments"):
        page = "segments"

    if st.button("Behaviour Patterns"):
        page = "behaviour"

with st.sidebar.expander("Model Insights"):

    if st.button("Performance Drivers"):
        page = "drivers"

# default page
if page is None:
    page = "overview"

# ---------------------------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------------------------

if page == "overview":

    st.title("Customer Experience Intelligence Dashboard")

    st.subheader("Dashboard Overview")

    st.write("""
This dashboard presents **descriptive analytics and exploratory data analysis (EDA)** 
for understanding customer experience performance.

The goal of this dashboard is to analyze customer behaviour and identify 
patterns that influence customer satisfaction and retention.

Key objectives include:

• Evaluating customer engagement patterns  
• Identifying relationships between CX variables  
• Understanding customer behavioural segments  
• Identifying drivers of CX performance using machine learning
""")

    st.subheader("Dataset Description")

    st.write("""
The dataset used in this analysis contains **400 simulated customer observations**.
Each observation represents a customer and includes several variables that capture 
engagement behaviour, support interactions, retention probability, and overall 
customer experience performance.
""")

    st.subheader("Key Variables")

    st.markdown("""
- **CXI Score** – Overall customer experience performance indicator  
- **User Engagement Score** – Level of interaction customers have with the platform  
- **Customer Retention** – Probability of customers continuing to use the service  
- **Support Tickets per Month** – Frequency of support requests made by customers  
- **CX Adoption Success** – Degree to which customers adopt CX initiatives  
- **Training Completion Rate** – Percentage of customers completing training modules
""")

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Customers", df.shape[0])

    with col2:
        st.metric("Number of Variables", df.shape[1])

    st.dataframe(df.head())
# ---------------------------------------------------
# EDA OVERVIEW
# ---------------------------------------------------

if page == "eda":

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="cxi_score",
            title="Distribution of CXI Score"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">EDA INSIGHT</div>
        <div class="insight-text">
        The CXI score distribution shows how customer experience varies 
        across the dataset. A wide spread suggests that some users 
        experience significantly better service than others.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        fig = px.histogram(
            df,
            x="customer_retention",
            title="Distribution of Customer Retention"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">EDA INSIGHT</div>
        <div class="insight-text">
        Customer retention values indicate the probability that customers 
        continue using the platform. Higher retention suggests stronger 
        customer satisfaction and loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------

elif page == "correlation":

    st.title("Correlation Analysis")

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">ANALYTICAL INSIGHT</div>
    <div class="insight-text">
    The correlation matrix helps identify relationships between CX variables. 
    Positive correlations indicate variables that increase together, while 
    negative correlations indicate inverse relationships.
    </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# CUSTOMER SEGMENTS
# ---------------------------------------------------

elif page == "segments":

    st.title("Customer Segmentation")

    features = df[
        ["user_engagement_score",
         "customer_retention",
         "support_tickets_per_month"]
    ]

    kmeans = KMeans(n_clusters=3)
    df["segment"] = kmeans.fit_predict(features)

    fig = px.scatter(
        df,
        x="user_engagement_score",
        y="customer_retention",
        color="segment",
        size="support_tickets_per_month",
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">SEGMENTATION INSIGHT</div>
    <div class="insight-text">
    K-Means clustering identifies groups of customers with similar behaviour. 
    Segments may represent loyal customers, struggling users, or at-risk users.
    </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# BEHAVIOUR PATTERNS
# ---------------------------------------------------

elif page == "behaviour":

    st.title("Customer Behaviour Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="user_engagement_score",
            y="customer_retention",
            title="Engagement vs Retention"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">BEHAVIOUR INSIGHT</div>
        <div class="insight-text">
        Higher engagement levels tend to correspond with improved 
        customer retention. Increasing engagement features may 
        therefore strengthen customer loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        fig = px.scatter(
            df,
            x="support_tickets_per_month",
            y="cxi_score",
            title="Support Tickets vs CXI Score"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">BEHAVIOUR INSIGHT</div>
        <div class="insight-text">
        Customers submitting more support tickets often report 
        lower CXI scores, suggesting product usability issues 
        or service friction.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PERFORMANCE DRIVERS
# ---------------------------------------------------

elif page == "drivers":

    st.title("Drivers of CX Performance")

    X = df.drop(columns=["cxi_score"])
    y = df["cxi_score"]

    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">MODEL INSIGHT</div>
    <div class="insight-text">
    The Random Forest model identifies variables that most strongly 
    influence CXI score. Higher importance indicates a stronger 
    impact on overall customer experience.
    </div>
    </div>
    """, unsafe_allow_html=True)
