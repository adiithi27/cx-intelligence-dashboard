import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

st.set_page_config(page_title="CX Intelligence Platform", layout="wide")

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.metric-card {
background-color:#0f172a;
padding:20px;
border-radius:12px;
border:1px solid #1e293b;
text-align:center;
box-shadow:0px 4px 15px rgba(0,0,0,0.4);
}

.metric-title {
font-size:14px;
color:#94a3b8;
}

.metric-value {
font-size:32px;
font-weight:700;
color:#22c55e;
}

.metric-sub {
font-size:12px;
color:#64748b;
}

.insight-box {
background-color:#0b1a33;
padding:20px;
border-radius:12px;
border:1px solid #2563eb;
margin-top:10px;
margin-bottom:30px;
}

.insight-title {
color:#60a5fa;
font-size:12px;
letter-spacing:1px;
font-weight:600;
margin-bottom:10px;
}

.insight-text {
color:#e5e7eb;
font-size:14px;
line-height:1.6;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------

st.sidebar.title("CX Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Descriptive Analytics",
        "Exploratory Data Analysis",
        "Customer Segmentation",
        "CX Driver Analysis"
    ]
)

# ------------------------------------------------
# EXECUTIVE OVERVIEW
# ------------------------------------------------

if page == "Executive Overview":

    st.title("CX Intelligence Platform")
    st.caption("Customer Experience Analytics • Behaviour Intelligence • CX Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">CXI Score</div>
        <div class="metric-value">{df['cxi_score'].mean():.2f}</div>
        <div class="metric-sub">Overall CX performance</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Customer Retention</div>
        <div class="metric-value">{df['customer_retention'].mean():.2f}</div>
        <div class="metric-sub">Retention rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">User Engagement</div>
        <div class="metric-value">{df['user_engagement_score'].mean():.2f}</div>
        <div class="metric-sub">Average engagement</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">CX Adoption</div>
        <div class="metric-value">{df['cx_adoption_success'].mean():.2f}</div>
        <div class="metric-sub">Adoption success</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Gauge

    health_score = df["cxi_score"].mean() * 10

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "CX Health Score"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range':[0,40],'color':"red"},
                {'range':[40,70],'color':"orange"},
                {'range':[70,100],'color':"lightgreen"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# DESCRIPTIVE ANALYTICS
# ------------------------------------------------

elif page == "Descriptive Analytics":

    st.title("Descriptive Analytics")

    st.subheader("Dataset Overview")

    st.write("Number of Observations:", df.shape[0])
    st.write("Number of Variables:", df.shape[1])

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

# ------------------------------------------------
# EDA
# ------------------------------------------------

elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="user_engagement_score",
            y="customer_retention",
            color="customer_retention",
            title="Engagement vs Retention"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">WHAT THIS MEANS FOR THE BUSINESS</div>
        <div class="insight-text">
        Higher engagement users demonstrate significantly stronger retention.
        Investing in engagement features will improve customer loyalty.
        </div>
        </div>
        """, unsafe_allow_html=True)


    with col2:

        fig = px.scatter(
            df,
            x="support_tickets_per_month",
            y="cxi_score",
            color="cxi_score",
            title="Support Tickets vs CXI"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <div class="insight-title">WHAT THIS MEANS FOR THE BUSINESS</div>
        <div class="insight-text">
        Customers generating more support tickets tend to show lower CX scores.
        Improving onboarding and help resources may reduce ticket volume.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------------------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    features = df[
        ["user_engagement_score","customer_retention","support_tickets_per_month"]
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

# ------------------------------------------------
# DRIVER ANALYSIS
# ------------------------------------------------

elif page == "CX Driver Analysis":

    st.title("CX Driver Analysis")

    X = df.drop(columns=["cxi_score"])
    y = df["cxi_score"]

    model = RandomForestRegressor()

    model.fit(X,y)

    importance = pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Drivers of CXI Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Automated CX Insights")

    top_driver = importance.iloc[0]["Feature"]

    st.success(f"Top CX Driver: **{top_driver}**")
