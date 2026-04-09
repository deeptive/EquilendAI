from faker import Faker
import pandas as pd
import random

fake = Faker()

data = []

for _ in range(100):
    data.append({
        "name": fake.name(),
        "age": random.randint(18, 70),
        "income": random.randint(20000, 100000),
        "credit_score": random.randint(300, 850)
    })

df = pd.DataFrame(data)
df.to_csv("data/mock_data.csv", index=False)

print("Mock data generated successfully!")