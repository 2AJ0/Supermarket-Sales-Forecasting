import streamlit as st #type:ignore
import pandas as pd#type:ignore
import matplotlib.pyplot as plt#type:ignore
import seaborn as sns#type:ignore
from sklearn.ensemble import RandomForestRegressor#type:ignore
from sklearn.model_selection import train_test_split#type:ignore
from sklearn.metrics import mean_absolute_error, r2_score#type:ignore
import calendar
import warnings

# Initial Configuration
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Supermarket Analytics Pro", layout="wide")

# --- 1. OPTIMIZED DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Loading only essential columns to save memory/time
        items = pd.read_csv('annex1.csv')
        sales = pd.read_csv('annex2.csv')
        
        # Clean column names (removes hidden spaces)
        items.columns = items.columns.str.strip()
        sales.columns = sales.columns.str.strip()

        # Filter for actual sales only
        sales = sales[sales['Sale or Return'].str.lower() == 'sale'].copy()
        
        # Fast Date Processing
        sales['Date'] = pd.to_datetime(sales['Date'], format='mixed')
        sales['Month'] = sales['Date'].dt.month
        sales['Year'] = sales['Date'].dt.year
        return items, sales
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None

items_df, sales_df = load_data()

if items_df is not None:
    # --- 2. SIDEBAR NAVIGATION ---
    st.sidebar.title("📊 Control Panel")
    choice = st.sidebar.radio("Navigation", ["📅 Monthly Sales", "📈 Sales Graph"])

    # --- 3. MODE: MONTHLY SALES (Reporting) ---
    if choice == "📅 Monthly Sales":
        st.header("Monthly Sales Analysis")
        st.info("Filter historical data to generate performance reports.")

        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Select Year", sorted(sales_df['Year'].unique(), reverse=True))
        with col2:
            selected_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: calendar.month_name[x])

        if st.button("Generate Report"):
            month_data = sales_df[(sales_df['Year'] == selected_year) & (sales_df['Month'] == selected_month)].copy()
            
            if month_data.empty:
                st.warning(f"No data available for {calendar.month_name[selected_month]} {selected_year}")
            else:
                # Calculations
                month_data['Revenue'] = month_data['Quantity Sold (kilo)'] * month_data['Unit Selling Price (RMB/kg)']
                summary = month_data.groupby('Item Code').agg({'Quantity Sold (kilo)':'sum', 'Revenue':'sum'}).reset_index()
                final = pd.merge(summary, items_df, on='Item Code')
                
                # Column Formatting
                final['item name&code'] = final['Item Name'] + " (" + final['Item Code'].astype(str) + ")"
                final = final.rename(columns={'Quantity Sold (kilo)': 'Total Sold (kg)', 'Revenue': 'Total Revenue'})
                
                # Display Results
                display_cols = ['item name&code', 'Category Name', 'Total Sold (kg)', 'Total Revenue']
                final_display = final[display_cols].sort_values(by='Total Sold (kg)', ascending=False)
                
                st.success(f"Displaying Top Items for {calendar.month_name[selected_month]}")
                st.dataframe(final_display.style.format({'Total Sold (kg)': '{:.2f}', 'Total Revenue': '{:.2f}'}), use_container_width=True)
                
                # CSV Export
                csv = final_display.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report as CSV", data=csv, file_name=f"Sales_Report_{selected_month}.csv", mime='text/csv')

    # --- 4. MODE: SALES GRAPH (AI Prediction) ---
    elif choice == "📈 Sales Graph":
        st.header("Sales Prediction Engine")
        st.write("Predict future sales volume using Random Forest Machine Learning.")

        target_month = st.select_slider("Select Month to Forecast", options=range(1, 13), format_func=lambda x: calendar.month_name[x])

        if st.button("Run Prediction Model"):
            with st.spinner('Training AI Model...'):
                # Data Preparation for ML
                monthly = sales_df.groupby(['Month', 'Item Code']).agg({'Quantity Sold (kilo)':'sum', 'Unit Selling Price (RMB/kg)':'mean'}).reset_index()
                df_model = pd.merge(monthly, items_df, on='Item Code')
                
                # Encoding text data to numbers
                df_model['Item_Enc'] = pd.factorize(df_model['Item Code'])[0]
                df_model['Cat_Enc'] = pd.factorize(df_model['Category Code'])[0]

                # Features and Target
                X = df_model[['Month', 'Item_Enc', 'Cat_Enc', 'Unit Selling Price (RMB/kg)']]
                y = df_model['Quantity Sold (kilo)']

                # Accuracy Check (Train/Test Split)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Model Execution
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Metrics
                mae = mean_absolute_error(y_test, model.predict(X_test))
                r2 = r2_score(y_test, model.predict(X_test))

                # Prediction for selected month
                predict_df = df_model.drop_duplicates('Item Code').copy()
                predict_df['Month'] = target_month
                predict_df['Predicted_Quantity'] = model.predict(predict_df[['Month', 'Item_Enc', 'Cat_Enc', 'Unit Selling Price (RMB/kg)']])
                
                top_10 = predict_df.nlargest(10, 'Predicted_Quantity')

                # --- UI OUTPUT ---
                st.divider()
                st.subheader(f"Prediction Graph for {calendar.month_name[target_month]}")
                
                # The Bar Chart
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=top_10, x='Item Name', y='Predicted_Quantity', palette='magma', ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Expected Sales (kg)")
                st.pyplot(fig)
                
                # Accuracy Metrics Display
                st.subheader("Model Performance")
                m1, m2 = st.columns(2)
                m1.metric("Average Error (MAE)", f"{mae:.2f} kg")
                m2.metric("Prediction Confidence (R²)", f"{r2*100:.1f}%")
                
                st.write("### Predicted Sales Values")
                st.table(top_10[['Item Name', 'Predicted_Quantity']].rename(columns={'Predicted_Quantity': 'Expected kg'}).round(2))

else:
    st.warning("Data files not found. Ensure 'annex1.csv' and 'annex2.csv' are in the project folder.")
