import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import math
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
import random

# Suppress warnings to avoid errors during model training
warnings.filterwarnings('ignore')


# add frontend path
app = FastAPI()

class DisasterData(BaseModel):
    disaster_type: str
    magnitude: float
    population: int
    infrastructure: float

@app.post("/")

async def main(data : DisasterData):
    input1 = data.disaster_type
    input2 = data.magnitude
    # Load dataset
    df = pd.read_csv("updated_disaster_data-2.csv")
    df1 = pd.read_csv("cleaned_table.csv")
    # Entering population and infrastructure rating as integer
    population = data.population
    infrastructure=data.infrastructure
    # Group dataset by Disaster Type
    group = {key: base for key, base in df.groupby("Disaster Type")}

    # Initialize lists for training data
    initialm = []
    initiald = []
    initialc = []
    initialt = []
    initiala = []
    initialp = []
    initialv = []
    # Linear Regression model
    model = LinearRegression()

    def data1():
        if "Storm" in group:
            storm = group["Storm"]
            value = input2 / 63  # Scaling factor
        
            for i in storm["Magnitude"]:
                if math.isclose(i / 63, value, rel_tol=1e-5):  # Avoid floating-point errors
                    initialm.append(i)

                    # Extract damage
                    damage = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "Total Damage (000 US$)"]
                    if not damage.empty:
                        initiald.append(damage.values[0])

                    # Extract city
                    city = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "Location"]
                    if city.values[0] != "unknown":
                        initialc.append(city.values[0])

                    # Extract start year
                    time = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "Start Year"]
                    if not time.empty:
                        initialt.append(time.values[0])
                    #Extract casualties
                    casualties = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "No. Affected"]
                    if not casualties.empty:
                        initiala.append(casualties.values[0])
                    #Extract population
                    pop = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "Population (000s)"]
                    if not pop.empty:
                        initialp.append(pop.values[0]*1000)
                    #Extract infrastructure
                    infra = df.loc[(df["Disaster Type"] == "Storm") & (df["Magnitude"] == i), "Infrastructure Value (M USD)"]
                    if not infra.empty:
                        initialv.append(infra.values[0])
        
    def data2():
        if "Earthquake" in group:
            earthquake = group["Earthquake"]
            value = input2 / 2  # Scaling factor
            for i in earthquake["Magnitude"]:
                if math.isclose(i / 2, value, rel_tol=1e-5):  # Avoid floating-point errors
                    initialm.append(i)
                    # Extract damage
                    damage = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "Total Damage (000 US$)"]
                    if not damage.empty:
                        initiald.append(damage.values[0])

                    # Extract city
                    city = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "Location"]
                    if city.values[0]!= "unknown":
                        initialc.append(city.values[0])

                    # Extract start year
                    time = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "Start Year"]
                    if not time.empty:
                        initialt.append(time.values[0])
                    # Extract casualties
                    casualties = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "No. Affected"]
                    if not casualties.empty:
                        initiala.append(casualties.values[0])
                    # Extract population
                    pop = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "Population (000s)"]
                    if not pop.empty:
                        initialp.append(pop.values[0]*1000)
                    # Extract infrastructure
                    infra = df.loc[(df["Disaster Type"] == "Earthquake") & (df["Magnitude"] == i), "Infrastructure Value (M USD)"]
                    if not infra.empty:
                        initialv.append(infra.values[0])

    def data3():
        if "Flood" in group:
            flood = group["Flood"]
            value = input2 / 10  # Scaling factor
            for i in flood["Magnitude"]:
                if math.isclose(i / 10, value, rel_tol=1e-5):  # Avoid floating-point errors
                    initialm.append(i)
                    # Extract damage
                    damage = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "Total Damage (000 US$)"]
                    if not damage.empty:
                        initiald.append(damage.values[0])

                    # Extract city
                    city = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "Location"]
                    if city.values[0]!= "unknown":
                        initialc.append(city.values[0])

                    # Extract start year
                    time = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "Start Year"]
                    if not time.empty:
                        initialt.append(time.values[0])
                    # Extract casualties
                    casualties = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "No. Affected"]
                    if not casualties.empty:
                        initiala.append(casualties.values[0])
                    # Extract population
                    pop = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "Population (000s)"]
                    if not pop.empty:
                        initialp.append(pop.values[0]*1000)
                    # Extract infrastructure
                    infra = df.loc[(df["Disaster Type"] == "Flood") & (df["Magnitude"] == i), "Infrastructure Value (M USD)"]
                    if not infra.empty:
                        initialv.append(infra.values[0])

    def data4():
        if "Hurricane" in group:
            hurricane = group["Hurricane"]
            value = input2 / 10  # Scaling factor
            for i in hurricane["Magnitude"]:
                if math.isclose(i / 10, value, rel_tol=1e-5):  # Avoid floating-point errors
                    initialm.append(i)
                    # Extract damage
                    damage = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "Total Damage (000 US$)"]
                    if not damage.empty:
                        initiald.append(damage.values[0])

                    # Extract city
                    city = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "Location"]
                    if city.values[0]!= "unknown":
                        initialc.append(city.values[0])

                    # Extract start year
                    time = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "Start Year"]
                    if not time.empty:
                        initialt.append(time.values[0])
                    # Extract casualties
                    casualties = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "No. Affected"]
                    if not casualties.empty:
                        initiala.append(casualties.values[0])
                    # Extract population
                    pop = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "Population (000s)"]
                    if not pop.empty:
                        initialp.append(pop.values[0]*1000)
                    # Extract infrastructure
                    infra = df.loc[(df["Disaster Type"] == "Hurricane") & (df["Magnitude"] == i), "Infrastructure Value (M USD)"]
                    if not infra.empty:
                        initialv.append(infra.values[0])

    def data5():
        if "Typhoon" in group:
            typhoon = group["Typhoon"]
            value = input2 / 10  # Scaling factor
            for i in typhoon["Magnitude"]:
                if math.isclose(i / 10, value, rel_tol=1e-5):  # Avoid floating-point errors
                    initialm.append(i)
                    # Extract damage
                    damage = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "Total Damage (000 US$)"]
                    if not damage.empty:
                        initiald.append(damage.values[0])

                    # Extract city
                    city = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "Location"]
                    if city.values[0]!= "unknown":
                        initialc.append(city.values[0])

                    # Extract start year
                    time = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "Start Year"]
                    if not time.empty:
                        initialt.append(time.values[0])
                    # Extract casualties
                    casualties = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "No. Affected"]
                    if not casualties.empty:
                        initiala.append(casualties.values[0])
                    # Extract population
                    pop = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "Population (000s)"]
                    if not pop.empty:
                        initialp.append(pop.values[0]*1000)
                    # Extract infrastructure
                    infra = df.loc[(df["Disaster Type"] == "Typhoon") & (df["Magnitude"] == i), "Infrastructure Value (M USD)"]
                    if not infra.empty:
                        initialv.append(infra.values[0])
                    
    if input1 == "Storm":
        data1()
    elif input1 == "Earthquake":
        data2()
    elif input1 == "Flood":
        data3()
    elif input1 == "Hurricane":
        data4()
    elif input1 == "Typhoon":
        data5()

    if initialm and initiald and initialv:  #To Ensure there's data before training
        initialm1 = np.array(initialm).reshape(-1, 1)
        initiald = np.array(initiald).reshape(-1, 1)
        initialv1 = np.array(initialv).reshape(-1, 1)

        xtrain=np.hstack((initialm1,initialv1))
        ytrain=initiald
        model.fit(xtrain, ytrain)

        x = np.array([[input2, infrastructure]])
        y = model.predict(x)
    if initiala and initialp and initialm:
        initialm3 = np.array(initialm).reshape(-1, 1)
        initiala1 = np.array(initiala).reshape(-1,1)
        initialp = np.array(initialp).reshape(-1, 1)
        X_train = np.hstack((initialm3, initialp))  # Two features
        y_train = initiala1

        model.fit(X_train, y_train)

        p = np.array([[input2, population]])
        q = model.predict(p)
    for h in y:
       total_damage= int(round(h[0]))
    for i in q:
       affected_population = int(round(i[0]))*1000
    
    # Calculating building wise impact from df1

    df1.columns = df1.columns.str.replace(r'[|.\d]', '', regex=True).str.strip()
    df1 = df1.rename(columns={'Unnamed': 'Index', 'Name': 'Building Name', 'Floors': 'Floors', 'Status': 'Status', 'Year': 'Year Built', 'Drawings': 'Drawings'})
    df1 = df1.drop(columns=['Index'], errors='ignore')

    current_year = 2025 
    df1['Year Built'] = pd.to_numeric(df1['Year Built'], errors='coerce')
    df1['Building Age'] = current_year - df1['Year Built']
    a=df1.loc[(df1['Status']=="built"),"Building Age"]
    f=df1.loc[(df1['Status']=="built"),"Floors"]
    b=df1.loc[(df1['Status']=="built"),"Name"]
    age=[]
    floors=[]
    Name=[]
    for i in a:
        age.append(i)
    for j in f:
        floors.append(j)
    for name in b:
        Name.append(name)
    magnitude=input2
    model1=None
    damage_value=[]
    def e1():
        # getting model
        global model1
        model_path = "earthquake_damage_model.pkl"
        model1 = joblib.load(model_path)

        # calculating damage
        damage = []
        for ages, floor in zip(age,floors):
            def predict_damage(ages, floor, magnitude):
                sample_input = np.array([[ages, floor, magnitude]])
                predicted_damage = model1.predict(sample_input)
                return predicted_damage
            damage.append(predict_damage(ages, floor, magnitude))
        damage_value.append(zip(damage,Name))
    if input1=="Earthquake":
        damge_value=e1()
    
    response = {
        "building_wise_impact": damge_value,
        "total_damage_estimation": total_damage,
        "estimated_affected_population": affected_population,
    }
    return response