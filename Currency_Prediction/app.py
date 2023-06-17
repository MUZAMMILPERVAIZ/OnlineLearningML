from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib


app = Flask(__name__)

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


def load_models():
    return OnlineLearningModel.load_model("model.pkl"), OnlineLearningModel.load_model("exchange_model.pkl")

loaded_model, Exchange_loaded_model = load_models()


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


@app.route('/', methods=['GET', 'POST'])
def home():
    result_message = ""

    if request.method == 'POST':

        if 'inflation_years' in request.form and 'inflation_rates' in request.form:
            inflation_years = request.form.get('inflation_years')
            inflation_rates = request.form.get('inflation_rates')
            years = [int(year.strip()) for year in inflation_years.split(',')]
            rates = [float(rate.strip()) for rate in inflation_rates.split(',')]

            for year, rate in zip(years, rates):
                if not 1950 <= year <= 2023:
                    return "🚫 Please enter a year between 1950 and 2023."
                loaded_model.process_new_data([[year]], [[rate]])
            loaded_model.save_model('model.pickle')

            result_message =  "✅ Model trained and saved successfully."

        elif 'exchange_years' in request.form and 'currency_rates' in request.form:
            exchange_years = request.form.get('exchange_years')
            currency_rates = request.form.get('currency_rates')
            years = [int(year.strip()) for year in exchange_years.split(',')]
            rates = [float(rate.strip()) for rate in currency_rates.split(',')]

            for year, rate in zip(years, rates):
                if not 1950 <= year <= 2023:
                    return "🚫 Please enter a year between 1950 and 2023."
                Exchange_loaded_model.process_new_data([[year]], [[rate]])
            Exchange_loaded_model.save_model('exchange_model.pkl')

            result_message =  "✅ Model trained and saved successfully."

        elif 'naira_amount' in request.form and 'years' in request.form and 'action' in request.form:
            naira_amount = float(request.form.get('naira_amount'))
            years = int(request.form.get('years'))
            action = request.form.get('action')

            if action == 'Future':
                result = calculate_value(naira_amount, years, future=True)
                if result is not None:
                    result_message =  f"The future value of {naira_amount} naira in {years} years is: {result}"
            else:
                result = calculate_value(naira_amount, years, future=False)
                if result is not None:
                    result_message =  f"The value of {naira_amount} naira {years} years ago was: {result}"
                else:
                    result_message =  "🚫 Please enter a number of years less than or equal to 23 for the past value calculation."

    return render_template('index.html', result=result_message)

if __name__ == "__main__":
    app.run(debug=True)
