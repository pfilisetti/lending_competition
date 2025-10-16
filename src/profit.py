
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_interest_rate(pd_pred, surplus_rate, eps=1e-6):
    pdc = np.clip(pd_pred, eps, 1 - eps)
    break_even = pdc / (1 - pdc)           
    rate = break_even + surplus_rate
    return np.clip(rate, 0.0, 1.0)         

def compute_preferences(df):
    # make sure n is divisible by 3, or handle remainder as you like
    n= len(df)
    count = n // 3
    remainder = n - 3*count

    vec = np.array([1]*count + [2]*count + [3]*count)
    if remainder > 0:
        vec = np.concatenate([vec, np.random.choice([1, 2, 3], size=remainder, replace=False)])

    np.random.shuffle(vec)
    df["preference"] = vec
    return df

def compute_choice(df):
    df["choice"] = 0
    for row in df.iterrows():
        preference = row["preference"]
        r1 = row["lender_1"]
        r2 = row["lender_2"]
        r3 = row["lender_3"]
        choices = [r1, r2, r3]
        choices[preference-1] -= 0.02
        lender_choice = choices.index(np.argmin(choices)) + 1
        row["lender_choice"] = lender_choice
    return df

def compute_profit(df):
    profit = 0
    for row in df.iterrows():
        choice = row["lender_choice"]
        if choice == 1:
            if row["default"] == 0:
                profit += 10000*row["lender_1"]
            else:
                profit -= 10000
    return profit


def compute_profit(default, pd_pred, surplus_rates):

    df = pd.DataFrame(default, pd_pred)

    for i, surplus_rate in enumerate(surplus_rates):
        df[f"lender_{i}"] = compute_interest_rate(pd_pred, surplus_rate)
    
    df = compute_preferences(df)

    df = compute_choice(df)

    profits = compute_profit(df)
    
    return profits

def compute_profit_distribution(default, pd_pred, min_surplus, max_surplus, num_simulations, seed=42):
    rng = np.random.default_rng(seed)

    # Draw all surplus rates at once: shape (num_simulations, 3)
    m = rng.uniform(min_surplus, max_surplus, size=(num_simulations, 3))

    profits = np.empty(num_simulations, dtype=float)
    for i in range(num_simulations):
        surplus_rates = m[i].tolist()  # [m1, m2, m3]
        profits[i] = compute_profit(default, pd_pred, surplus_rates)

    df = pd.DataFrame({
        "m1": m[:, 0],
        "m2": m[:, 1],
        "m3": m[:, 2],
        "profits": profits
    })

    return df

def plot_profit_distribution(df):
    y = df["profits"]
    x = df["m1"]
    plt.scatter(x, y)
    plt.xlabel("m1")
    plt.ylabel("profits")
    plt.title("Profit Distribution")
    plt.show()
    return df

