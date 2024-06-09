from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
dataset = pd.read_csv("[Dataset]_(Asuransi).csv")

# Function needed for data processing
dataset = dataset.drop(columns=['Unnamed: 0'])

# Data cleaning and transformation
dataset['insured sex'] = dataset['insured_sex_MALE'].apply(lambda x: 'Male' if x else 'Female')
dataset = dataset.drop(columns=['insured_sex_MALE', 'insured_sex_FEMALE'])
dataset['fraud reported'] = dataset['fraud_reported'].apply(lambda x: 'Yes' if x else 'No')
dataset = dataset.drop(columns=['fraud_reported'])

occupation_columns = [
    'insured_occupation_adm-clerical', 'insured_occupation_armed-forces',
    'insured_occupation_craft-repair', 'insured_occupation_exec-managerial',
    'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners',
    'insured_occupation_machine-op-inspct', 'insured_occupation_other-service',
    'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty',
    'insured_occupation_protective-serv', 'insured_occupation_sales',
    'insured_occupation_tech-support', 'insured_occupation_transport-moving'
]
def get_occupation(row):
    for col in occupation_columns:
        if row[col] == 1:
            return col.replace('insured_occupation_', '')
    return 'Unknown'
dataset['occupation'] = dataset.apply(get_occupation, axis=1)
dataset = dataset.drop(columns=occupation_columns)

hobbies_columns = [
    'insured_hobbies_chess', 'insured_hobbies_cross-fit',
    'insured_hobbies_other'
]
def get_hobbies(row):
    for col in hobbies_columns:
        if row[col] == 1:
            return col.replace('insured_hobbies_', '')
    return 'Unknown'
dataset['hobbies'] = dataset.apply(get_hobbies, axis=1)
dataset = dataset.drop(columns=hobbies_columns)

incident_columns = [
    'incident_type_Multi-vehicle Collision', 'incident_type_Parked Car',
    'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft'
]
def get_incident(row):
    for col in incident_columns:
        if row[col] == 1:
            return col.replace('incident_type_', '')
    return 'Unknown'
dataset['incident'] = dataset.apply(get_incident, axis=1)
dataset = dataset.drop(columns=incident_columns)

collision_columns = [
    'collision_type_?', 'collision_type_Front Collision',
    'collision_type_Rear Collision', 'collision_type_Side Collision'
]
def get_collision(row):
    for col in collision_columns:
        if row[col] == 1:
            if col.split('_')[2] == '?':
                return col.replace('collision_type_?', 'other')
            return col.replace('collision_type_', '')
    return 'Unknown'
dataset['collision'] = dataset.apply(get_collision, axis=1)
dataset = dataset.drop(columns=collision_columns)

severity_columns = [
    'incident_severity_Major Damage', 'incident_severity_Minor Damage',
    'incident_severity_Total Loss', 'incident_severity_Trivial Damage'
]
def get_severity(row):
    for col in severity_columns:
        if row[col] == 1:
            return col.replace('incident_severity_', '')
    return 'Unknown'
dataset['severity'] = dataset.apply(get_severity, axis=1)
dataset = dataset.drop(columns=severity_columns)

contacted_columns = [
    'authorities_contacted_Ambulance', 'authorities_contacted_Fire',
    'authorities_contacted_None', 'authorities_contacted_Other',
    'authorities_contacted_Police'
]
def get_contacted(row):
    for col in contacted_columns:
        if row[col] == 1:
            return col.replace('authorities_contacted_', '')
    return 'Unknown'
dataset['authorities contacted'] = dataset.apply(get_contacted, axis=1)
dataset = dataset.drop(columns=contacted_columns)

age_columns = [
    'age_group_15-20', 'age_group_21-25',
    'age_group_26-30', 'age_group_31-35',
    'age_group_36-40', 'age_group_41-45',
    'age_group_46-50', 'age_group_51-55',
    'age_group_56-60', 'age_group_61-65'
]
def get_age(row):
    for col in age_columns:
        if row[col] == 1:
            return col.replace('age_group_', ' ')
    return 'Unknown'
dataset['age'] = dataset.apply(get_age, axis=1)
dataset = dataset.drop(columns=age_columns)

months_columns = [
    'months_as_customer_groups_0-50', 'months_as_customer_groups_101-150',
    'months_as_customer_groups_151-200', 'months_as_customer_groups_201-250',
    'months_as_customer_groups_251-300', 'months_as_customer_groups_301-350',
    'months_as_customer_groups_351-400', 'months_as_customer_groups_401-450',
    'months_as_customer_groups_451-500', 'months_as_customer_groups_51-100'
]
def get_months(row):
    for col in months_columns:
        if row[col] == 1:
            return col.replace('months_as_customer_groups_', ' ')
    return 'Unknown'
dataset['months since joining'] = dataset.apply(get_months, axis=1)
dataset = dataset.drop(columns=months_columns)

policy_columns = [
    'policy_annual_premium_groups_high', 'policy_annual_premium_groups_very high',
    'policy_annual_premium_groups_low', 'policy_annual_premium_groups_very low',
    'policy_annual_premium_groups_medium'
]
def get_policy(row):
    for col in policy_columns:
        if row[col] == 1:
            return col.replace('policy_annual_premium_groups_', ' ')
    return 'Unknown'
dataset['policy annual premium'] = dataset.apply(get_policy, axis=1)
dataset = dataset.drop(columns=policy_columns)

dataset['capital-loss'] = dataset['capital-loss'].apply(lambda x: abs(x) if x < 0 else x)

def predict_fraud_with_accuracy(input_data):

    datasetprocess = dataset
    x = datasetprocess.drop(columns=['fraud reported'])
    y = datasetprocess['fraud reported']

    categorical_cols = x.select_dtypes(include=['object']).columns
    numerical_cols = x.select_dtypes(include=['number']).columns
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model_pipeline.fit(X_train, y_train)
    sample_df = pd.DataFrame(input_data)
    input_processed = preprocessor.transform(sample_df)
    prediction = model_pipeline.predict(sample_df)
    y_pred_test = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    return prediction[0], accuracy

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    hasil = None
    
    if request.method == 'POST':
        toCalculate = {
            'capital-gains': [float(request.form.get('capital_gains'))],
            'capital-loss': [float(request.form.get('capital_loss'))],
            'incident_hour_of_the_day': [int(request.form.get('incident_hour_of_the_day'))],
            'number_of_vehicles_involved': [int(request.form.get('number_of_vehicles_involved'))],
            'witnesses': [int(request.form.get('witnesses'))],
            'total_claim_amount': [float(request.form.get('total_claim_amount'))],
            'insured sex': [request.form.get('insured_sex')],
            'occupation': [request.form.get('occupation')],
            'hobbies': [request.form.get('hobbies')],
            'incident': [request.form.get('incident_type')],
            'collision': [request.form.get('collision_type')],
            'severity': [request.form.get('incident_severity')],
            'authorities contacted': [request.form.get('authorities_contacted')],
            'age': [request.form.get('age_group')],
            'months since joining': [request.form.get('months_as_customer_groups')],
            'policy annual premium': [request.form.get('policy_annual_premium_groups')]
        }

        is_fraud, model_accuracy = predict_fraud_with_accuracy(toCalculate)
        
        hasil = "<p>Hasil Kalkulasi: </p>"
        if is_fraud == 'Yes':
            hasil += f"<p>Penipuan Asuransi</p>"
        else:
            hasil += f"<p>Tidak Penipuan Asuransi</p>"
        hasil += f"<p>Akurasi Model: {model_accuracy * 100:.2f}%</p>"
    
    return render_template('app.html', hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)
