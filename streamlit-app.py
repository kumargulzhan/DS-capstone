import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.title("Прогнозирование с Prophet")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Убедимся, что колонка с датами имеет правильный тип
    data['createdon_date'] = pd.to_datetime(data['createdon_date'])
    data = data.set_index('createdon_date')

    if 'seats' not in data.columns:
        st.error("Файл должен содержать колонку 'seats' (значение).")
    else:
        data = data.sort_values('createdon_date')

        # Выбор горизонта прогнозирования
        option = st.selectbox("Выберите горизонт прогноза", ["30 дней", "90 дней", "365 дней"])

        # Прогноз на 365 дней
        if option == "365 дней":
            monthly_data = data.resample('MS').sum()
            df = monthly_data.reset_index()
            df.columns = ['ds', 'y']

            test_months = 12
            train_df = df[:-test_months]
            test_df = df[-test_months:]

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(train_df)

            future = model.make_future_dataframe(periods=test_months + 12, freq='MS')
            forecast = model.predict(future)
            forecast_result = forecast[['ds', 'yhat']]
            test_df = test_df.set_index('ds')
            forecast_result = forecast_result.set_index('ds')

            y_true = test_df['y']
            y_pred = forecast_result.loc[test_df.index]['yhat']

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            st.subheader("График прогноза на 365 дней")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_df['ds'], train_df['y'], label='Monthly Train', color='blue')

            x_link = [train_df['ds'].iloc[-1], test_df.index[0]]
            y_link = [train_df['y'].iloc[-1], test_df['y'].iloc[0]]
            ax.plot(x_link, y_link, color='blue')

            ax.plot(test_df.index, test_df['y'], label='Monthly Actual (Test)', color='green')
            ax.plot(forecast_result.index, forecast_result["yhat"], label='12 Months Forecast', color='orange')

            ax.legend()
            ax.grid(True)
            ax.set_title('12 Months Forecast')
            st.pyplot(fig)

            st.subheader("Метрики качества для 365 дней")
            st.markdown(f"""
            - **MAE**: {mae:.2f}  
            - **RMSE**: {rmse:.2f}  
            - **R²**: {r2:.4f}
            """)

        # Прогноз на 30 дней
        elif option == "30 дней":
            daily_data = data.resample('D').sum()
            daily_df = daily_data.reset_index()
            daily_df.columns = ['ds', 'y']

            daily_test_days = 365
            daily_train_df = daily_df[:-daily_test_days]
            daily_test_df = daily_df[-daily_test_days:]

            daily_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            daily_model.fit(daily_train_df)

            daily_future = daily_model.make_future_dataframe(periods=daily_test_days+30, freq='D')
            daily_forecast = daily_model.predict(daily_future)

            daily_forecast_result = daily_forecast[['ds', 'yhat']]
            daily_test_df = daily_test_df.set_index('ds')
            daily_forecast_result = daily_forecast_result.set_index('ds')

            daily_y_true = daily_test_df['y']
            daily_y_pred = daily_forecast_result.loc[daily_test_df.index]['yhat']

            daily_mae = mean_absolute_error(daily_y_true, daily_y_pred)
            daily_rmse = np.sqrt(mean_squared_error(daily_y_true, daily_y_pred))
            daily_r2 = r2_score(daily_y_true, daily_y_pred)

            st.subheader("График прогноза на 30 дней")
            forecast_end_date = pd.to_datetime("2025-05-15")
            display_start_date = forecast_end_date - pd.DateOffset(months=6)

            fig, ax = plt.subplots(figsize=(12, 6))
            train_plot = daily_train_df[daily_train_df['ds'] >= display_start_date]
            ax.plot(train_plot['ds'], train_plot['y'], label='Daily Train', color='blue')

            if daily_train_df['ds'].iloc[-1] >= display_start_date:
                x_link = [daily_train_df['ds'].iloc[-1], daily_test_df.index[0]]
                y_link = [daily_train_df['y'].iloc[-1], daily_test_df['y'].iloc[0]]
                ax.plot(x_link, y_link, color='blue')

            test_plot = daily_test_df[daily_test_df.index >= display_start_date]
            ax.plot(test_plot.index, test_plot['y'], label='Daily Actual (Test)', color='green')

            forecast_plot = daily_forecast_result[
                (daily_forecast_result.index >= display_start_date) &
                (daily_forecast_result.index <= forecast_end_date)
            ]
            ax.plot(forecast_plot.index, forecast_plot['yhat'], label='30 days forecast', color='orange')

            ax.axvline(x=daily_test_df.index.max(), color='black', linestyle='--', label='Test End')
            ax.set_xlim(display_start_date, forecast_end_date)

            ax.legend()
            ax.grid(True)
            ax.set_title('30 Days Forecast — Last 6 Months')
            st.pyplot(fig)

            st.subheader("Метрики качества для 30 дней")
            st.markdown(f"""
            - **MAE**: {daily_mae:.2f}  
            - **RMSE**: {daily_rmse:.2f}  
            - **R²**: {daily_r2:.4f}
            """)

        # Прогноз на 90 дней
        else:
            daily_data = data.resample('D').sum()
            daily_df = daily_data.reset_index()
            daily_df.columns = ['ds', 'y']

            daily_test_days = 365
            daily_train_df = daily_df[:-daily_test_days]
            daily_test_df = daily_df[-daily_test_days:]

            daily_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            daily_model.fit(daily_train_df)

            daily_future = daily_model.make_future_dataframe(periods=daily_test_days+90, freq='D')
            daily_forecast = daily_model.predict(daily_future)

            daily_forecast_result = daily_forecast[['ds', 'yhat']]
            daily_test_df = daily_test_df.set_index('ds')
            daily_forecast_result = daily_forecast_result.set_index('ds')

            daily_y_true = daily_test_df['y']
            daily_y_pred = daily_forecast_result.loc[daily_test_df.index]['yhat']

            daily_mae = mean_absolute_error(daily_y_true, daily_y_pred)
            daily_rmse = np.sqrt(mean_squared_error(daily_y_true, daily_y_pred))
            daily_r2 = r2_score(daily_y_true, daily_y_pred)

            st.subheader("График прогноза на 90 дней")
            forecast_end_date = pd.to_datetime("2025-07-15")
            display_start_date = forecast_end_date - pd.DateOffset(months=12)

            fig, ax = plt.subplots(figsize=(12, 6))
            train_plot = daily_train_df[daily_train_df['ds'] >= display_start_date]
            ax.plot(train_plot['ds'], train_plot['y'], label='Daily Train', color='blue')

            if daily_train_df['ds'].iloc[-1] >= display_start_date:
                x_link = [daily_train_df['ds'].iloc[-1], daily_test_df.index[0]]
                y_link = [daily_train_df['y'].iloc[-1], daily_test_df['y'].iloc[0]]
                ax.plot(x_link, y_link, color='blue')

            test_plot = daily_test_df[daily_test_df.index >= display_start_date]
            ax.plot(test_plot.index, test_plot['y'], label='Daily Actual (Test)', color='green')

            forecast_plot = daily_forecast_result[
                (daily_forecast_result.index >= display_start_date) &
                (daily_forecast_result.index <= forecast_end_date)
            ]
            ax.plot(forecast_plot.index, forecast_plot['yhat'], label='90 days forecast', color='orange')

            ax.axvline(x=daily_test_df.index.max(), color='black', linestyle='--', label='Test End')
            ax.set_xlim(display_start_date, forecast_end_date)

            ax.legend()
            ax.grid(True)
            ax.set_title('90 Days Forecast — Last 12 Months')
            st.pyplot(fig)

            st.subheader("Метрики качества для 90 дней")
            st.markdown(f"""
            - **MAE**: {daily_mae:.2f}  
            - **RMSE**: {daily_rmse:.2f}  
            - **R²**: {daily_r2:.4f}
            """)

else:
    st.info("Пожалуйста, загрузите CSV-файл с колонками `createdon_date` (дата) и `seats` (значение).")
