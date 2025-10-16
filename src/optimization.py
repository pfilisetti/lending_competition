import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_mean23(res_df: pd.DataFrame):
    mean_23 = res_df[["m2", "m3"]].sum(axis=1) / 2
    m1 = res_df["m1"]
    profits = res_df["profits"]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        mean_23,
        m1,
        c=profits,
        cmap="coolwarm",      # "coolwarm" → cold (blue) to hot (red)
        s=50,                 # point size
        edgecolor='k',        # black border around points (optional)
        alpha=0.8             # transparency for overlap
    )

    plt.colorbar(scatter, label="Profit")
    plt.xlabel("Mean of m2 and m3")
    plt.ylabel("m1")
    plt.title("Profit Landscape: m1 vs mean(m2, m3)")
    plt.grid(True, alpha=0.3)

    plt.show()








def split_by_m1_rank(res_df: pd.DataFrame):
    df = res_df.copy()
    # rank of m1 within the row (1 = lowest, 3 = highest)
    df["rank_m1"] = df[["m1", "m2", "m3"]].rank(axis=1, method="first")["m1"].astype(int)
    df["rank_m1"] = 4 - df["rank_m1"]
    df_r1 = df[df["rank_m1"] == 1].copy()
    df_r2 = df[df["rank_m1"] == 2].copy()
    df_r3 = df[df["rank_m1"] == 3].copy()
    return df_r1, df_r2, df_r3



def plot_profit_distributions(df: pd.DataFrame, df_r1: pd.DataFrame, df_r2: pd.DataFrame, df_r3: pd.DataFrame):
    # Define the common x-range from the full dataframe
    min_profit = df["profits"].min()
    max_profit = df["profits"].max()

    plt.figure(figsize=(9, 6))

    # KDE curves for each rank
    sns.kdeplot(df_r1["profits"], label="Rank 1", color="blue",  fill=False, common_norm=False, clip=(min_profit, max_profit))
    sns.kdeplot(df_r2["profits"], label="Rank 2", color="orange", fill=False, common_norm=False, clip=(min_profit, max_profit))
    sns.kdeplot(df_r3["profits"], label="Rank 3", color="red",   fill=False, common_norm=False, clip=(min_profit, max_profit))

    plt.xlabel("Profit")
    plt.ylabel("Density")
    plt.title("Profit Distribution by m1 Rank (smooth curves)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



