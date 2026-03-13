import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="CX Intelligence Dashboard", layout="wide")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------

st.markdown("""
<style>

.sidebar-title{
font-size:22px;
font-weight:700;
}

.variable-card{
background-color:#0f172a;
padding:20px;
border-radius:10px;
border:1px solid #1e293b;
margin-bottom:15px;
}

.variable-title{
font-weight:600;
font-size:15px;
color:#60a5fa;
}

.variable-text{
font-size:14px;
color:#e5e7eb;
line-height:1.6;
}

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
# SIDEBAR NAVIGATION
# ---------------------------------------------------

st.sidebar.markdown("## CX Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "App Overview",
        "EDA Summary",
        "EDA Overview",
        "Correlation Analysis",
        "User Segments",
        "Behaviour Patterns",
        "Performance Drivers"
    ]
)

# ---------------------------------------------------
# APP OVERVIEW
# ---------------------------------------------------

if page == "App Overview":

    st.title("Customer Experience Intelligence Dashboard")

    st.write("""
This dashboard provides **descriptive analytics and exploratory data analysis (EDA)** 
to evaluate customer engagement, support behaviour, retention patterns, and overall 
customer experience performance.
""")

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Customers", df.shape[0])

    with col2:
        st.metric("Number of Variables", df.shape[1])

    st.divider()

    st.subheader("Variable Definitions")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">CXI Score</div>
        <div class="variable-text">
        Customer Experience Index representing overall customer satisfaction.
        Higher values indicate better overall experience.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">User Engagement Score</div>
        <div class="variable-text">
        Measures the level of interaction customers have with the platform.
        Higher engagement generally indicates stronger product usage.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">Customer Retention</div>
        <div class="variable-text">
        Indicates the likelihood of customers continuing to use the platform.
        Higher retention values suggest stronger customer loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">Support Tickets per Month</div>
        <div class="variable-text">
        Represents how frequently customers contact support.
        Higher ticket counts may indicate usability issues.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">CX Adoption Success</div>
        <div class="variable-text">
        Measures how successfully customers adopt CX initiatives
        such as new features or service improvements.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="variable-card">
        <div class="variable-title">Training Completion Rate</div>
        <div class="variable-text">
        Indicates how many users completed onboarding or training modules,
        which can influence product understanding and adoption success.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# EDA SUMMARY (NEW)
# ---------------------------------------------------

elif page == "EDA Summary":

    st.title("Automated EDA Insights")

    corr1 = df["user_engagement_score"].corr(df["customer_retention"])
    corr2 = df["support_tickets_per_month"].corr(df["cxi_score"])

    st.markdown("""
    ### Key Analytical Insights
    """)

    if corr1 > 0.5:
        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">Engagement Impact</div>
        <div class="insight-text">
        Customer engagement has a strong positive relationship with retention,
        indicating that improving engagement can significantly increase loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)

    if corr2 < 0:
        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">Support Impact</div>
        <div class="insight-text">
        Higher support ticket frequency correlates with lower CXI scores,
        suggesting usability challenges or product friction.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">General Observation</div>
    <div class="insight-text">
    Customer experience is influenced by multiple behavioural factors.
    Engagement and support interactions are two key drivers of CX performance.
    </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# EDA OVERVIEW
# ---------------------------------------------------

elif page == "EDA Overview":

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(df, x="cxi_score", title="Distribution of CXI Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(df, x="customer_retention", title="Retention Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------

elif page == "Correlation Analysis":

    st.title("Correlation Matrix")

    corr = df.corr()

    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# USER SEGMENTS
# ---------------------------------------------------

elif page == "User Segments":

    st.title("Customer Segmentation")

    features = df[[
        "user_engagement_score",
        "customer_retention",
        "support_tickets_per_month"
    ]]

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

# ---------------------------------------------------
# BEHAVIOUR PATTERNS
# ---------------------------------------------------

elif page == "Behaviour Patterns":

    st.title("Behaviour Patterns")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="user_engagement_score",
            y="customer_retention",
            title="Engagement vs Retention"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            df,
            x="support_tickets_per_month",
            y="cxi_score",
            title="Support Tickets vs CXI Score"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PERFORMANCE DRIVERS
# ---------------------------------------------------

elif page == "Performance Drivers":

    st.title("CX Performance Drivers")

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
