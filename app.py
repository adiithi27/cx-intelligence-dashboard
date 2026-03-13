import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="CX Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("cx_simulated_dataset_400.csv")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("CX Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "User Segments",
        "Revenue Intelligence",
        "Behaviour Patterns",
        "Performance Drivers"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Dataset")
st.sidebar.write(f"{len(df)} customers")

# -----------------------------
# EXECUTIVE OVERVIEW
# -----------------------------
if page == "Executive Overview":

    st.title("CX Intelligence Platform")
    st.caption("Customer Experience Performance & Monetisation Analytics")

    # ---------------- KPI CARDS ----------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "CXI Score",
        f"{df['cxi_score'].mean():.2f}"
    )

    col2.metric(
        "Customer Retention",
        f"{df['customer_retention'].mean():.2f}"
    )

    col3.metric(
        "User Engagement",
        f"{df['user_engagement_score'].mean():.2f}"
    )

    col4.metric(
        "CX Adoption",
        f"{df['cx_adoption_success'].mean():.2f}"
    )

    st.divider()

    # ---------------- HEALTH GAUGE ----------------

    health_score = df["cxi_score"].mean() * 10

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Customer Experience Health"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [0, 40], 'color': "#7f1d1d"},
                {'range': [40, 70], 'color': "#b45309"},
                {'range': [70, 100], 'color': "#14532d"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CHART GRID ----------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="user_engagement_score",
            y="customer_retention",
            color="customer_retention",
            title="Engagement vs Retention",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            df,
            x="support_tickets_per_month",
            y="cxi_score",
            color="cxi_score",
            title="Support Load vs CXI Score",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            df,
            x="cxi_score",
            nbins=30,
            title="CXI Score Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        fig = px.histogram(
            df,
            x="customer_retention",
            nbins=30,
            title="Retention Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# USER SEGMENTS
# -----------------------------
elif page == "User Segments":

    st.title("Customer Segmentation")

    fig = px.scatter(
        df,
        x="user_engagement_score",
        y="cxi_score",
        color="customer_retention",
        size="support_tickets_per_month",
        title="Customer Segmentation Map",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# REVENUE INTELLIGENCE
# -----------------------------
elif page == "Revenue Intelligence":

    st.title("Revenue Intelligence")

    fig = px.box(
        df,
        y="cxi_score",
        x="cx_adoption_success",
        title="CX Adoption vs CXI",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# BEHAVIOUR PATTERNS
# -----------------------------
elif page == "Behaviour Patterns":

    st.title("Customer Behaviour Insights")

    fig = px.scatter_matrix(
        df,
        dimensions=[
            "user_engagement_score",
            "customer_retention",
            "support_tickets_per_month",
            "cxi_score"
        ],
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PERFORMANCE DRIVERS
# -----------------------------
elif page == "Performance Drivers":

    st.title("Key CX Drivers")

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
        title="Drivers of CXI Score",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Automated Insights")

    top_driver = importance.iloc[-1]["Feature"]

    st.success(f"Top CX driver is **{top_driver}**")
