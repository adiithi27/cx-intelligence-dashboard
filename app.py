import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="CX Analytics Dashboard", layout="wide")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# ----------------------------------------------------
# CSS STYLE
# ----------------------------------------------------

st.markdown("""
<style>

.insight-box{
background-color:#0b1a33;
padding:20px;
border-radius:10px;
border:1px solid #2563eb;
margin-top:10px;
margin-bottom:25px;
}

.insight-title{
color:#60a5fa;
font-size:12px;
font-weight:600;
letter-spacing:1px;
margin-bottom:8px;
}

.insight-text{
color:#e5e7eb;
font-size:14px;
line-height:1.6;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------

st.sidebar.title("CX Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "App Overview",
        "EDA Overview",
        "Correlation Analysis",
        "User Segments",
        "Behaviour Patterns",
        "Performance Drivers"
    ]
)

# ----------------------------------------------------
# APP OVERVIEW
# ----------------------------------------------------

if page == "App Overview":

    st.title("Customer Experience Intelligence Dashboard")

    st.write("""
This dashboard provides **descriptive analytics and exploratory data analysis (EDA)** 
to understand customer behaviour, engagement, retention, and CX performance.

The objective of this analysis is to:
- Evaluate customer engagement patterns
- Identify correlations between CX variables
- Understand behavioural customer segments
- Identify drivers of CX performance
""")

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Customers", df.shape[0])

    with col2:
        st.metric("Number of Variables", df.shape[1])

    st.dataframe(df.head())


# ----------------------------------------------------
# EDA OVERVIEW
# ----------------------------------------------------

elif page == "EDA Overview":

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(df,
                           x="cxi_score",
                           title="Distribution of CXI Score")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">EDA INSIGHT</div>
        <div class="insight-text">
        The CXI Score distribution shows how overall customer experience 
        varies across users. A wider spread suggests variability in service 
        quality experienced by customers.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        fig = px.histogram(df,
                           x="customer_retention",
                           title="Distribution of Customer Retention")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">EDA INSIGHT</div>
        <div class="insight-text">
        Retention distribution helps identify how many customers are likely 
        to stay with the platform. Higher retention values indicate stronger 
        customer loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------
# CORRELATION ANALYSIS
# ----------------------------------------------------

elif page == "Correlation Analysis":

    st.title("Correlation Analysis")

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix of CX Variables"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">ANALYTICAL INSIGHT</div>
    <div class="insight-text">
    The correlation matrix shows relationships between CX variables. 
    Positive values indicate that two variables increase together, 
    while negative correlations suggest inverse relationships. 
    Strong correlations highlight variables that may influence 
    customer experience performance.
    </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------
# USER SEGMENTS
# ----------------------------------------------------

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

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">SEGMENTATION INSIGHT</div>
    <div class="insight-text">
    K-Means clustering groups customers based on engagement, retention, 
    and support usage patterns. This helps identify high-value customers, 
    struggling users, and at-risk segments.
    </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------
# BEHAVIOUR PATTERNS
# ----------------------------------------------------

elif page == "Behaviour Patterns":

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
        Higher engagement levels are generally associated with improved 
        retention rates. This suggests that increasing product interaction 
        may help strengthen customer loyalty.
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
        Customers submitting more support tickets tend to report lower CXI 
        scores, suggesting usability issues or service friction.
        </div>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------
# PERFORMANCE DRIVERS
# ----------------------------------------------------

elif page == "Performance Drivers":

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
        title="Key Drivers of CXI Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <div class="insight-title">MODEL INSIGHT</div>
    <div class="insight-text">
    The Random Forest model identifies the most influential variables 
    affecting CXI Score. Features with higher importance values have 
    a stronger impact on customer experience performance.
    </div>
    </div>
    """, unsafe_allow_html=True)
