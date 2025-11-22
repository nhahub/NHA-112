import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Load data
data_path = os.path.join(
    os.path.dirname(__file__),
    "../../data/interim/reddit_complaints_dataset.csv"
)
df = pd.read_csv(data_path)

st.title("People Complaints Dashboard")

# ---------------- Summary ----------------
st.header("Overview")
st.metric("Total Complaints", df.shape[0])
st.metric("Unique Categories", df["category"].nunique())
st.metric("Unique Subcategories", df["subreddit"].nunique())

# ---------------- Charts ----------------

# by Category
st.subheader("Distribution by Category")
fig_pie = px.pie(df, names="category", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_pie)

# by Subcategory

st.subheader("Number of Complaints per Subcategory")
sub_counts = df["subreddit"].value_counts().reset_index()
sub_counts.columns = ["subreddit", "count"]

fig_sub = px.bar(
    sub_counts,
    x="subreddit",
    y="count",
)

fig_sub.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig_sub)

# by Problem Type
st.subheader("Number of Complaints per Problem Type (Horizontal)")

problem_counts = df["problem_type"].value_counts().reset_index()
problem_counts.columns = ["problem_type", "count"]

fig_bar = px.bar(
    problem_counts,
    y="problem_type",    # categories on y-axis
    x="count",           # values on x-axis
    orientation='h',     # horizontal bars
    text='count'         # show counts on bars
)

# Sort bars descending
fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})  
# Note: 'total ascending' = largest on top

st.plotly_chart(fig_bar)





# ---------------- Filterable Data ----------------
st.header("View and Filter Complaints")
category_filter = st.multiselect("Filter by Category:", options=df["category"].unique())
filtered_df = df[df["category"].isin(category_filter)] if category_filter else df.copy()

subcategory_filter = st.multiselect("Filter by Subcategory:", options=filtered_df["subreddit"].unique())
filtered_df = filtered_df[filtered_df["subreddit"].isin(subcategory_filter)] if subcategory_filter else filtered_df

st.subheader("Filtered Data Summary")
st.write(f"Total complaints in filtered data: {filtered_df.shape[0]}")

st.dataframe(filtered_df)
