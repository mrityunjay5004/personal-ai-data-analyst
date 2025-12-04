# 🧠 Personal AI Data Analyst — The Vishleshak

A fully interactive **AI-powered data analysis web application** that allows users to upload datasets and perform intelligent analysis using **Python, Streamlit, and Groq LLM**.

This project supports:
- Automated exploratory data analysis
- Smart visualizations
- LLM-powered natural language queries
- CSV, Excel, and JSON file formats

🔗 **Live App:** [](https://vishleshak.streamlit.app/)  
📁 **GitHub Repo:** https://github.com/mrityunjay5004/personal-ai-data-analyst

---

## 🖼️ Application Preview

![App Preview](app_preview.png)

---

## 🚀 Features

✅ Upload CSV, XLSX, XLS, and JSON files  
✅ Automatic dataset preview (first 100 rows)  
✅ Smart prompt suggestions based on column types  
✅ Built-in analysis without AI:
- Summary statistics  
- Histograms  
- Scatter plots  
- Correlation heatmaps  
- Time-series aggregation  
- Anomaly detection using Z-Score  

✅ AI-powered custom analysis using **Groq LLM**  
✅ Download analysis results as CSV  
✅ Clean, dark-mode professional UI  
✅ Fully deployed on **Streamlit Cloud**

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit  
- **Backend:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **AI Engine:** Groq LLM (LLaMA 3.3 – 70B)  
- **Deployment:** Streamlit Community Cloud  

---

## 📁 Project Structure

```text
personal-ai-data-analyst/
│
├── app.py                 # Streamlit main application
├── data_loader.py         # File upload and parsing
├── prompt_engine.py       # Prompt suggestions & rule-based logic
├── code_runner.py         # Secure code execution engine
├── llm_client.py          # Groq API connector
├── requirements.txt       # Dependencies
├── .gitignore
└── app_preview.png        # UI screenshot
