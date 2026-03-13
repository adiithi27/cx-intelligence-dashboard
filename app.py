import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="CX Intelligence Platform",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (KPI CARDS)
# -------------------------------

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

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# -------------------------------
# TITLE
# -------------------------------

st.title("CX Intelligence Platform")
st.caption("Customer Experience Analytics • Behaviour Intelligence • CX Performance")

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------

st.sidebar.header("Filters")

engagement_filter = st.sidebar.slider(
    "Minimum Engagement Score",
    float(df["user_engagement_score"].min()),
    float(df["user_engagement_score"].max()),
    float(df["user_engagement_score"].min())
)

ticket_filter = st.sidebar.slider(
    "Maximum Support Tickets",
    int(df["support_tickets_per_month"].min()),
    int(df["support_tickets_per_month"].max()),
    int(df["support_tickets_per_month"].max())
)

filtered_df = df[
    (df["user_engagement_score"] >= engagement_filter) &
    (df["support_tickets_per_month"] <= ticket_filter)
]

# -------------------------------
# KPI CARDS
# -------------------------------

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CXI Score</div>
        <div class="metric-value">{filtered_df['cxi_score'].mean():.2f}</div>
        <div class="metric-sub">Overall CX performance</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customer Retention</div>
        <div class="metric-value">{filtered_df['customer_retention'].mean():.2f}</div>
        <div class="metric-sub">Retention rate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">User Engagement</div>
        <div class="metric-value">{filtered_df['user_engagement_score'].mean():.2f}</div>
        <div class="metric-sub">Average engagement</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CX Adoption</div>
        <div class="metric-value">{filtered_df['cx_adoption_success'].mean():.2f}</div>
        <div class="metric-sub">Adoption success</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -------------------------------
# CX HEALTH GAUGE
# -------------------------------

health_score = filtered_df["cxi_score"].mean() * 10

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "CX Health Score"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "#22c55e"},
        'steps': [
            {'range':[0,40], 'color':"red"},
            {'range':[40,70], 'color':"orange"},
            {'range':[70,100], 'color':"lightgreen"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------
# MULTI GRAPH ANALYTICS
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        filtered_df,
        x="user_engagement_score",
        y="customer_retention",
        color="customer_retention",
        title="Engagement vs Retention"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        filtered_df,
        x="support_tickets_per_month",
        y="cxi_score",
        color="cxi_score",
        title="Support Tickets vs CXI Score"
    )
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(
        filtered_df,
        x="cxi_score",
        title="CXI Score Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.histogram(
        filtered_df,
        x="customer_retention",
        title="Retention Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------

st.subheader("Customer Segmentation")

features = filtered_df[
    ["user_engagement_score","customer_retention","support_tickets_per_month"]
]

kmeans = KMeans(n_clusters=3)
filtered_df["segment"] = kmeans.fit_predict(features)

fig = px.scatter(
    filtered_df,
    x="user_engagement_score",
    y="customer_retention",
    color="segment",
    size="support_tickets_per_month",
    title="Customer Segments"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------
# DRIVER ANALYSIS
# -------------------------------

st.subheader("Key Drivers of CXI Score")

X = filtered_df.drop(columns=["cxi_score","segment"])
y = filtered_df["cxi_score"]

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

st.divider()

# -------------------------------
# AUTOMATED CX INSIGHTS
# -------------------------------

st.subheader("Automated CX Insights")

top_driver = importance.iloc[0]["Feature"]

st.success(f"Top CX Driver: **{top_driver}**")

if filtered_df["user_engagement_score"].corr(filtered_df["customer_retention"]) > 0.5:
    st.write("Higher engagement strongly correlates with customer retention.")

if filtered_df["support_tickets_per_month"].corr(filtered_df["cxi_score"]) < 0:
    st.write("More support tickets are associated with lower CX scores.")

if filtered_df["training_completion_rate"].corr(filtered_df["cx_adoption_success"]) > 0.4:
    st.write("Training completion improves CX adoption success.")

st.divider()

# -------------------------------
# DATASET VIEW
# -------------------------------

with st.expander("View Dataset"):
    st.dataframe(filtered_df)
