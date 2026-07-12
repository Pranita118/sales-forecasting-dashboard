import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
.stApp { background-color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #EAF1F4; }
div[data-testid="stMetric"] {
    background-color: #F7FAFB;
    border: 1px solid #D9E4E8;
    border-radius: 10px;
    padding: 12px 16px;
}
h1, h2, h3 { color: #1F4E5F; }
.insight-box {
    background-color: #F7FAFB;
    border-left: 4px solid #1F4E5F;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df

df = load_data()

st.sidebar.title("Sales Intelligence")
st.sidebar.caption("Superstore Forecasting & Demand System")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data range: {df['Order Date'].min().date()} -> {df['Order Date'].max().date()}")
st.sidebar.caption(f"Total records: {len(df):,}")

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        region_filter = st.multiselect("Region", df["Region"].unique(), default=list(df["Region"].unique()))
    with col2:
        category_filter = st.multiselect("Category", df["Category"].unique(), default=list(df["Category"].unique()))
    with col3:
        date_range = st.date_input("Date range", [df["Order Date"].min(), df["Order Date"].max()])

    filtered = df[
        df["Region"].isin(region_filter) &
        df["Category"].isin(category_filter) &
        (df["Order Date"] >= pd.to_datetime(date_range[0])) &
        (df["Order Date"] <= pd.to_datetime(date_range[1]))
    ]

    total_sales = filtered["Sales"].sum()
    total_orders = filtered["Order ID"].nunique()
    avg_order_value = filtered.groupby("Order ID")["Sales"].sum().mean()
    years = sorted(filtered["Year"].unique())
    yoy = None
    if len(years) >= 2:
        last, prev = filtered[filtered["Year"]==years[-1]]["Sales"].sum(), filtered[filtered["Year"]==years[-2]]["Sales"].sum()
        yoy = ((last - prev) / prev) * 100 if prev else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${total_sales:,.0f}")
    k2.metric("Total Orders", f"{total_orders:,}")
    k3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    k4.metric("YoY Growth", f"{yoy:.1f}%" if yoy is not None else "N/A")

    st.markdown("---")

    c1, c2 = st.columns([1,1])
    with c1:
        yearly_sales = filtered.groupby("Year")["Sales"].sum().reset_index()
        fig1 = px.bar(yearly_sales, x="Year", y="Sales", title="Total Sales by Year",
                      color_discrete_sequence=["#1F4E5F"])
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        monthly = filtered.set_index("Order Date")["Sales"].resample("M").sum().reset_index()
        fig2 = px.line(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend",
                       color_discrete_sequence=["#1F4E5F"])
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        reg_sales = filtered.groupby("Region")["Sales"].sum().reset_index()
        fig3 = px.pie(reg_sales, names="Region", values="Sales", title="Sales by Region",
                      color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig4 = px.pie(cat_sales, names="Category", values="Sales", title="Sales by Category",
                      color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Top 10 Sub-Categories by Revenue")
    top10 = filtered.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(top10, use_container_width=True)

    st.download_button("Download filtered data as CSV",
                        filtered.to_csv(index=False).encode("utf-8"),
                        "filtered_sales_data.csv", "text/csv")

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    options = df[dim].unique()
    choice = st.selectbox(f"Select {dim}", options)
    horizon = st.slider("Forecast horizon (months)", 1, 3, 3)

    sub = df[df[dim] == choice]
    ts = sub.set_index("Order Date")["Sales"].resample("M").sum().reset_index()
    ts["Lag1"] = ts["Sales"].shift(1)
    ts["Lag2"] = ts["Sales"].shift(2)
    ts["Lag3"] = ts["Sales"].shift(3)
    ts["RollingMean3"] = ts["Sales"].shift(1).rolling(3).mean()
    ts["Month"] = ts["Order Date"].dt.month
    ts["Quarter"] = ts["Order Date"].dt.quarter
    ts.dropna(inplace=True)

    feats = ["Lag1","Lag2","Lag3","RollingMean3","Month","Quarter"]
    X, y = ts[feats], ts["Sales"]
    Xtr, Xte, ytr, yte = X.iloc[:-3], X.iloc[-3:], y.iloc[:-3], y.iloc[-3:]

    with st.spinner("Training forecast model..."):
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)[:horizon]

    mae = mean_absolute_error(yte[:horizon], pred)
    rmse = np.sqrt(mean_squared_error(yte[:horizon], pred))

    fig = px.line(ts.tail(15), x="Order Date", y="Sales", title=f"Forecast for {choice}",
                  color_discrete_sequence=["#1F4E5F"])
    future_dates = ts["Order Date"].iloc[-3:-3+horizon] if horizon < 3 else ts["Order Date"].iloc[-3:]
    fig.add_scatter(x=future_dates, y=pred, mode="lines+markers", name="Forecast",
                     line=dict(dash="dash", color="#D97706"))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")

    forecast_str = ", ".join(f"${v:,.0f}" for v in pred)
    st.markdown(f"""
    <div class="insight-box">
    Forecast for <b>{choice}</b>: over the next {horizon} month(s), predicted sales are {forecast_str}.
    </div>
    """, unsafe_allow_html=True)

    forecast_table = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(len(pred))],
        "Forecasted Sales": pred
    })
    st.download_button("Download forecast as CSV",
                        forecast_table.to_csv(index=False).encode("utf-8"),
                        f"forecast_{choice}.csv", "text/csv")

elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly = df.set_index("Order Date")["Sales"].resample("W").sum().reset_index()
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["anomaly"] = iso.fit_predict(weekly[["Sales"]])
    weekly["anomaly"] = weekly["anomaly"].map({1: 0, -1: 1})
    anom = weekly[weekly["anomaly"]==1]

    k1, k2 = st.columns(2)
    k1.metric("Anomalous Weeks Detected", len(anom))
    k2.metric("% of Total Weeks", f"{len(anom)/len(weekly)*100:.1f}%")

    fig = px.line(weekly, x="Order Date", y="Sales", title="Weekly Sales with Anomalies Highlighted",
                  color_discrete_sequence=["#1F4E5F"])
    fig.add_scatter(x=anom["Order Date"], y=anom["Sales"], mode="markers",
                     marker=dict(color="#D97706", size=10), name="Anomaly")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")
    st.dataframe(anom[["Order Date","Sales"]].reset_index(drop=True), use_container_width=True)

    st.download_button("Download anomaly list as CSV",
                        anom.to_csv(index=False).encode("utf-8"),
                        "anomalies.csv", "text/csv")

elif page == "Demand Segments":
    st.title("Product Demand Segments")

    subcat = df.groupby("Sub-Category").agg(Total_Sales=("Sales","sum"), Avg_Order_Value=("Sales","mean")).reset_index()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subcat[["Total_Sales","Avg_Order_Value"]])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    subcat["Cluster"] = km.fit_predict(X_scaled)

    label_map = {0: "High Value, Volatile Demand", 1: "Low Volume, Stable Demand", 2: "High Volume, Stable Demand", 3: "Niche, Rapidly Growing"}
    subcat["Cluster_Label"] = subcat["Cluster"].map(label_map)

    fig = px.scatter(subcat, x="Total_Sales", y="Avg_Order_Value", color="Cluster_Label",
                      hover_data=["Sub-Category"], title="Product Clusters",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category to Cluster Mapping")
    selected_cluster = st.selectbox("Filter by cluster", ["All"] + list(subcat["Cluster_Label"].unique()))
    display_df = subcat if selected_cluster == "All" else subcat[subcat["Cluster_Label"]==selected_cluster]
    st.dataframe(display_df[["Sub-Category","Cluster_Label","Total_Sales","Avg_Order_Value"]], use_container_width=True)

    st.download_button("Download cluster mapping as CSV",
                        subcat.to_csv(index=False).encode("utf-8"),
                        "product_clusters.csv", "text/csv")
