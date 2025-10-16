
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_interest_rate_additive(pd_pred, surplus_rate, eps=1e-6):
    pdc = np.clip(pd_pred, eps, 1 - eps)
    break_even = pdc / (1 - pdc)           
    rate = break_even + surplus_rate
    return rate

def compute_interest_rate_multiplicative(pd_pred, surplus_rate, eps=1e-6):
    pdc = np.clip(pd_pred, eps, 1 - eps)
    break_even = pdc / (1 - pdc)           
    rate = pdc * (1 + surplus_rate)
    return rate

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
    df = df.copy()
    df["lender_choice"] = 0   # ensure column exists

    for idx, row in df.iterrows():            # unpack (idx, row)
        preference = int(row["preference"])   # ensure int (1..3)

        r1, r2, r3 = row["lender_1"], row["lender_2"], row["lender_3"]
        choices = [r1, r2, r3]

        # apply 0.02 reduction to the preferred lender (1-based -> 0-based)
        choices[preference - 1] -= 0.02
        for choice in choices:
            if choice > 1.0:
                choice = -1
        if all(choice == -1 for choice in choices):
            lender_choice = 0
        else:
            lender_choice = int(np.argmin(choices)) + 1  # 1..3
        df.at[idx, "lender_choice"] = lender_choice

    return df

def compute_profit(df):
    profit = 0.0
    for idx, row in df.iterrows():                # ✅ unpack
        choice = int(row["lender_choice"])        # 1, 2, or 3
        d = int(row["default"])                   # 0 or 1

        # pick the chosen lender’s rate
        if choice == 1:
            r = row["lender_1"]       
            if d == 0:
                # non-default profit contribution
                profit += 10000*r
            else:
                # default loss contribution
                profit -= 10000
        # -------------------------------------------

    return profit



def compute_profit_pipe(default, pd_preds, surplus_rates, method="additive"):

    df = pd.concat([default, pd_preds], axis=1)
    if method == "additive":
        compute_interest_rate = compute_interest_rate_additive
    elif method == "multiplicative":
        compute_interest_rate = compute_interest_rate_multiplicative
    else:
        raise ValueError(f"Invalid method: {method}")

    for i, surplus_rate in enumerate(surplus_rates):
        pd_pred = pd_preds[f"PD{i+1}"]
        df[f"lender_{i+1}"] = compute_interest_rate(pd_pred, surplus_rate)
    
    df = compute_preferences(df)

    df = compute_choice(df)

    profits = compute_profit(df)
    
    return profits

def compute_profit_distribution(default, pd_preds, min_surplus, max_surplus, num_simulations, seed=42):
    rng = np.random.default_rng(seed)

    # Draw all surplus rates at once: shape (num_simulations, 3)
    m = rng.uniform(min_surplus, max_surplus, size=(num_simulations, 3))

    profits = np.empty(num_simulations, dtype=float)
    for i in range(num_simulations):
        print(f"{i/num_simulations*100}%")
        surplus_rates = m[i].tolist()  # [m1, m2, m3]
        profits[i] = compute_profit_pipe(default, pd_preds, surplus_rates)

    df = pd.DataFrame({
        "m1": m[:, 0],
        "m2": m[:, 1],
        "m3": m[:, 2],
        "profits": profits
    })

    return df



