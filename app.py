import os
import openai
import pandas as pd
import streamlit as st
from pandasai import SmartDatalake
from pandasai.llm.azure_openai import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser
import matplotlib.pyplot as plt
import seaborn as sns

# 🔐 Hardcoded Azure OpenAI Settings
AZURE_OPENAI_API_ENDPOINT = "Your AZURE_OPENAI_API_ENDPOINT"
AZURE_OPENAI_API_KEY = "Your AZURE_OPENAI_API_KEY"
AZURE_OPENAI_API_DEPLOYMENT_NAME = "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2023-07-01-preview"

# Configure OpenAI for beautification
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_API_ENDPOINT
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_version = AZURE_OPENAI_API_VERSION

# Streamlit config
st.set_page_config(layout="wide", page_title="Chat with Excel / CSV Data")
st.title("📊 Chat with Excel / CSV Data")

# File Upload Section
with st.container():
    input_files = st.file_uploader("Upload Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)
    data_list = []

    if input_files:
        for input_file in input_files:
            if input_file.name.lower().endswith(".csv"):
                data = pd.read_csv(input_file)
            else:
                data = pd.read_excel(input_file)

            # Clean: remove 'Total' etc. in 'Sr.No.'
            if "Sr.No." in data.columns:
                data = data[pd.to_numeric(data["Sr.No."], errors="coerce").notnull()]

            st.dataframe(data, use_container_width=True)
            data_list.append(data)
    else:
        st.header("Example Data (Default)")
        try:
            data = pd.read_excel(r"C:\Users\Acer\Desktop\combined_data.xlsx")
            if "Sr.No." in data.columns:
                data = data[pd.to_numeric(data["Sr.No."], errors="coerce").notnull()]
            st.dataframe(data, use_container_width=True)
            data_list = [data]
        except FileNotFoundError:
            st.error("Default file not found. Please upload a file to continue.")
            st.stop()

# Initialize SmartDatalake
df = SmartDatalake(
    dfs=data_list,
    config={
        "llm": AzureOpenAI(
            api_base=AZURE_OPENAI_API_ENDPOINT,
            api_token=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_API_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        ),
        "response_parser": ResponseParser,
        "use_cache": False,
    },
)

# User Query
st.header("🧠 Ask anything from your data!")
input_text = st.text_area(
    "Enter your question",
    value="What is the total Applications Received in April for all districts?"
)

if input_text:
    if st.button("Start Execution"):
        first_result = df.chat(input_text)
        st.subheader("🧾 Raw Table / Data Output:")
        st.write(first_result)

        # 🔍 Visualization
        st.subheader("📊 Visualizations")

        # Try to convert dict/list to DataFrame if needed
        if isinstance(first_result, dict):
            first_result = pd.DataFrame([first_result])
        elif isinstance(first_result, list):
            try:
                first_result = pd.DataFrame(first_result)
            except:
                pass

        if isinstance(first_result, pd.DataFrame):
            generated_any_graph = False

            # Auto-detect label and value columns
            label_col = None
            value_col = None
            for col in first_result.columns:
                if 'district' in col.lower() or 'name' in col.lower():
                    label_col = col
                if 'application' in col.lower() and 'received' in col.lower():
                    value_col = col

            if label_col and value_col:
                # Bar Chart
                st.markdown("#### 📊 Bar Chart")
                fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                sns.barplot(x=label_col, y=value_col, data=first_result, ax=ax_bar)
                plt.xticks(rotation=65, ha='right')
                plt.title("Applications Received in April for All Districts")
                plt.ylabel("Applications Received")
                plt.xlabel("District")
                st.pyplot(fig_bar)
                generated_any_graph = True

                # Pie Chart
                st.markdown("#### 🥧 Pie Chart")
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                ax_pie.pie(first_result[value_col], labels=first_result[label_col], autopct='%1.1f%%', startangle=140)
                ax_pie.axis('equal')
                plt.title("Applications Share by District")
                st.pyplot(fig_pie)

            if not generated_any_graph:
                st.info("📢 Not enough structure to render graphs. Ensure your result includes districts and values.")
        else:
            st.info("📢 Not applicable for this question: Response is not a table.")

        # ✨ Beautification
        user_question = input_text
        first_result_str = str(first_result)

        with st.spinner('Beautifying the answer...'):
            response = openai.ChatCompletion.create(
                deployment_id=AZURE_OPENAI_API_DEPLOYMENT_NAME,
                model="gpt-35-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a highly professional data analyst and insights writer. "
                            "Given a user's question and a raw answer generated from a dataset, "
                            "your task is to craft a clean, structured, and insightful professional report. "
                            "Ensure the response is easy to understand, highlights key observations, "
                            "provides any important trends or patterns, and sounds polished and formal."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Here is the user's question:\n\n{user_question}\n\n"
                            f"Here is the raw answer based on the data:\n\n{first_result_str}\n\n"
                            "Please craft a professional, insightful answer according to the question."
                        )
                    }
                ]
            )
            beautified_answer = response['choices'][0]['message']['content']
            st.subheader("📝 Constructive Analytical Report:")
            st.write(beautified_answer)
