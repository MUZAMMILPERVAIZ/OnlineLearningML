import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib




class OnlineLearningModel:
    def __init__(self):
        self.sgd = SGDRegressor(loss='squared_loss', penalty='l2', random_state=0)
        self.scaler = StandardScaler()
        self.batch_size = 100
        self.n_iterations = 10
        self.scaler_fit = False

    def train(self, X, y):
        self.scaler.fit(X)
        self.scaler_fit = True

        total_samples = len(X)
        for _ in range(self.n_iterations):
            indices = np.random.permutation(total_samples)
            batches = [indices[i:i + self.batch_size] for i in range(0, total_samples, self.batch_size)]

            for batch in batches:
                X_batch = X[batch]
                y_batch = y[batch]
                X_batch_scaled = self.scaler.transform(X_batch)
                self.sgd.partial_fit(X_batch_scaled, y_batch)

    def process_new_data(self, X_new, y_new):
        if self.scaler_fit:
            self.scaler.partial_fit(X_new)
        else:
            self.scaler.fit(X_new)
            self.scaler_fit = True

        total_samples = len(X_new)
        for _ in range(self.n_iterations):
            indices = np.random.permutation(total_samples)
            batches = [indices[i:i + self.batch_size] for i in range(0, total_samples, self.batch_size)]

            for batch in batches:
                X_batch = np.array(X_new)[batch]
                y_batch = np.array(y_new)[batch]
                X_batch_scaled = self.scaler.transform(X_batch)
                self.sgd.partial_fit(X_batch_scaled, y_batch)

    def predict(self, year):
        X_new_scaled = self.scaler.transform([[year]])
        y_pred = self.sgd.predict(X_new_scaled)
        y_pred_rounded = np.round((y_pred), 4)  
        return y_pred_rounded

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        return joblib.load(filepath)



def calculate_value(naira_amount, years, future=True):
    if not future and years > 23:
        print("Please enter a number of years less than or equal to 23 for the past value calculation.")
        return None
    
   
    projected_year = 2023 + years if future else 2023 - years
    projected_inflation_rate = max(0, loaded_model.predict(projected_year))
    projected_exchange_rate = max(0, Exchange_loaded_model.predict(projected_year))
    
    current_exchange_rate = 460.95
    
    if future:
        # Calculate future value
        future_value = naira_amount * (1 + projected_inflation_rate)**years * (projected_exchange_rate / current_exchange_rate)
        future_value = np.round((future_value), 2)
        return future_value
    else:
        # Calculate past value
        past_value = naira_amount / ((1 + projected_inflation_rate)**years * (projected_exchange_rate / current_exchange_rate))
        past_value = np.round((past_value), 2)
        return past_value


import streamlit as st

def load_models():
    return OnlineLearningModel.load_model("model.pkl"), OnlineLearningModel.load_model("exchange_model.pkl")

loaded_model, Exchange_loaded_model = load_models()

def main():
    st.markdown("<h1 style='text-align: center; color: darkblue;'>💡 Welcome to the Dashboard 💡</h1>", unsafe_allow_html=True)

    st.sidebar.title('💼 Menu')
    session_choice = st.sidebar.radio("", ['Inflation Learning', 'Exchange Learning', 'Calculation'])

    if session_choice == 'Inflation Learning':
        st.header('📈 Inflation Learning')
        with st.form(session_choice):
            inflation_years = st.text_input("Enter the years (separated by commas)")
            inflation_rates = st.text_input("Enter the inflation rates (separated by commas)")
            submit = st.form_submit_button('Submit')

        if submit:
            years = [int(year.strip()) for year in inflation_years.split(',')]
            rates = [float(rate.strip()) for rate in inflation_rates.split(',')]

            for year, rate in zip(years, rates):
                if not 1950 <= year <= 2023:
                    st.error("🚫 Please enter a year between 1950 and 2023.")
                    return
                loaded_model.process_new_data([[year]], [[rate]])
            loaded_model.save_model('model.pickle')
            st.success("✅ Model trained and saved successfully.")

    elif session_choice == 'Exchange Learning':
        st.header('💱 Exchange Learning')
        with st.form(session_choice):
            exchange_years = st.text_input("Enter the years (separated by commas)")
            currency_rates = st.text_input("Enter the currency rates (separated by commas)")
            submit = st.form_submit_button('Submit')

        if submit:
            years = [int(year.strip()) for year in exchange_years.split(',')]
            rates = [float(rate.strip()) for rate in currency_rates.split(',')]

            for year, rate in zip(years, rates):
                if not 1950 <= year <= 2023:
                    st.error("🚫 Please enter a year between 1950 and 2023.")
                    return
                Exchange_loaded_model.process_new_data([[year]], [[rate]])
            Exchange_loaded_model.save_model('exchange_model.pkl')
            st.success("✅ Model trained and saved successfully.")

    elif session_choice == 'Calculation':
        st.header('🧮 Calculation')
        with st.form('Calculation'):
            st.subheader('Input Parameters')
            naira_amount = st.number_input('Enter the naira amount', value=1000, step=1000, min_value=0)
            years = st.selectbox('Select the number of years', list(range(1, 24)))  
            action = st.selectbox('Do you want to calculate the future value or past value?', ('Future', 'Past'))
            submit_button = st.form_submit_button(label='Calculate')

        if submit_button:
            if action == 'Future':
                result = calculate_value(naira_amount, years, future=True)
                if result is not None:
                    st.write(f"The future value of {naira_amount} naira in {years} years is: {result}")
            else:
                result = calculate_value(naira_amount, years, future=False)
                if result is not None:
                    st.write(f"The value of {naira_amount} naira {years} years ago was: {result}")
                else:
                    st.error("🚫 Please enter a number of years less than or equal to 23 for the past value calculation.")

if __name__ == "__main__":
    main()


