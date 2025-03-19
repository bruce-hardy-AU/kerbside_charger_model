import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style for plots
sns.set_style("whitegrid")

# ========================
# 1. Model Parameters
# ========================

# Default parameters that replicate the Inputs sheet from the Excel model.
default_params = {
    "ChargersPerYear": 5000,
    "CapExPerCharger": 6000,
    "OpExPerCharger": 500,
    "AssetLife": 8,
    "WACC1to5": 0.058,     # Post-tax Weighted Average Cost of Capital for Years 1-5
    "WACC6to10": 0.06,     # For Years 6-10
    "WACC11to15": 0.055,   # For Years 11-15
    "CustomerBase": 1800000,
    "ThirdPartyRevenue": 100,
    "SharedAssetOffset": 0,
}

# High-Plus scenario overrides (worse-case assumptions)
high_plus_params = {
    "CapExPerCharger": 9000,
    "OpExPerCharger": 750,
    "WACC1to5": 0.0585,
    # For simplicity, WACC6to10 and WACC11to15 remain unchanged here.
}

# ========================
# 2. Model Calculation Functions
# ========================
def run_ev_charger_model(params):
    """
    Runs the EV Charger Regulatory Asset Base (RAB) model for a given set of parameters.
    
    Financial Concepts:
    - **Phased Rollout:** Assumes a fixed number of chargers are installed each year for the first 6 years.
    - **Straight-line Depreciation:** The capital expenditure (CapEx) for each cohort is depreciated evenly over the asset's life.
    - **RAB Roll-forward:** The Regulatory Asset Base is computed each year as the opening RAB plus new additions minus depreciation.
    - **Revenue & Bill Impact:** The model estimates the revenue requirement based on average RAB, the cost of capital (WACC), 
      operating expenses (OpEx), and then calculates the per-household bill impact based on a customer base.
    
    Parameters:
      params (dict): Dictionary of model parameters.
      
    Returns:
      rollout_df (DataFrame): Yearly rollout details.
      depreciation_df (DataFrame): Depreciation amounts per year.
      rab_df (DataFrame): RAB roll-forward details.
      revenue_df (DataFrame): Revenue and bill impact details.
      summary_stats (dict): Key summary metrics (e.g., peak bill impact, total cumulative cost).
    """
    # A) Build Rollout (Years 1 to 15)
    years = range(1, 16)
    rollout_data = []
    cumulative = 0
    for y in years:
        chargers_installed = params["ChargersPerYear"] if y <= 6 else 0
        capex = chargers_installed * params["CapExPerCharger"]
        cumulative += chargers_installed
        rollout_data.append([y, chargers_installed, capex, cumulative])
    rollout_df = pd.DataFrame(rollout_data, columns=["Year", "ChargersInstalled", "CapEx", "CumulativeChargers"])

    # B) Depreciation Matrix using Straight-line Depreciation
    depreciation_dict = {y: 0 for y in years}
    for cohort in range(1, 7):
        cohort_capex = rollout_df.loc[rollout_df["Year"] == cohort, "CapEx"].values[0]
        # Depreciate the cost evenly over AssetLife years
        for y in range(cohort, cohort + params["AssetLife"]):
            if y in depreciation_dict:
                depreciation_dict[y] += cohort_capex / params["AssetLife"]
    depreciation_data = [[y, depreciation_dict[y]] for y in years]
    depreciation_df = pd.DataFrame(depreciation_data, columns=["Year", "TotalDepreciation"])

    # C) RAB Roll-forward
    # RAB = Opening RAB + Additions – Depreciation.
    rab_data = []
    opening_rab = 0
    for y in years:
        additions = rollout_df.loc[rollout_df["Year"] == y, "CapEx"].values[0]
        depr = depreciation_df.loc[depreciation_df["Year"] == y, "TotalDepreciation"].values[0]
        closing = opening_rab + additions - depr
        rab_data.append([y, opening_rab, additions, depr, closing])
        opening_rab = closing
    rab_df = pd.DataFrame(rab_data, columns=["Year", "OpeningRAB", "Additions", "Depreciation", "ClosingRAB"])

    # D) Revenue & Bill Impact Calculations
    # Return on Capital = Average RAB * appropriate WACC
    rev_data = []
    for y in years:
        # Average RAB in year y (average of opening and closing RAB)
        row_rab = rab_df.loc[rab_df["Year"] == y]
        avg_rab = (row_rab["OpeningRAB"].values[0] + row_rab["ClosingRAB"].values[0]) / 2

        # Use WACC based on the year (reflecting regulatory periods)
        if y <= 5:
            wacc = params["WACC1to5"]
        elif y <= 10:
            wacc = params["WACC6to10"]
        else:
            wacc = params["WACC11to15"]

        return_on_cap = avg_rab * wacc
        cum_chargers = rollout_df.loc[rollout_df["Year"] == y, "CumulativeChargers"].values[0]
        opex = cum_chargers * params["OpExPerCharger"]
        depr = rab_df.loc[rab_df["Year"] == y, "Depreciation"].values[0]
        tpr = cum_chargers * params["ThirdPartyRevenue"]
        offset = cum_chargers * params["SharedAssetOffset"]
        allowed = return_on_cap + opex + depr - tpr - offset
        bill_impact = allowed / params["CustomerBase"]
        rev_data.append([
            y, avg_rab, return_on_cap, opex, depr, tpr, offset, allowed, bill_impact
        ])
    revenue_df = pd.DataFrame(rev_data, columns=[
        "Year", "AvgRAB", "ReturnOnCapital", "Opex", "Depreciation",
        "ThirdPartyRevenue", "SharedAssetOffset", "AllowedRevenue", "BillImpactPerCustomer"
    ])
    revenue_df["CumulativeBillImpact"] = revenue_df["BillImpactPerCustomer"].cumsum()
    revenue_df["TotalCostAllCustomers"] = revenue_df["CumulativeBillImpact"] * params["CustomerBase"]

    # E) Compute Summary Statistics
    avg_bill = revenue_df["BillImpactPerCustomer"].mean()
    peak_bill = revenue_df["BillImpactPerCustomer"].max()
    year_peak = revenue_df.loc[revenue_df["BillImpactPerCustomer"].idxmax(), "Year"]
    total_rev = revenue_df["AllowedRevenue"].sum()
    final_cum_bill = revenue_df["CumulativeBillImpact"].iloc[-1]
    final_total_cost = revenue_df["TotalCostAllCustomers"].iloc[-1]
    summary_stats = {
        "Average Annual Bill Impact": avg_bill,
        "Peak Annual Bill Impact": peak_bill,
        "Year of Peak Impact": year_peak,
        "Total Revenue Requirement": total_rev,
        "Cumulative Bill Impact/Customer": final_cum_bill,
        "Total Cumulative Cost to All Customers": final_total_cost
    }
    return rollout_df, depreciation_df, rab_df, revenue_df, summary_stats


def sensitivity_analysis(param_name, test_values, outcome_key):
    """
    Performs a sensitivity analysis on a given parameter.
    
    For each test value provided for 'param_name', the model is re-run while holding all
    other parameters at their default values. The outcome metric (e.g., Total Cumulative Cost to All Customers,
    Average Annual Bill Impact, or Cumulative Bill Impact per Customer) is recorded.
    
    This function returns a DataFrame with:
      - The test values for the parameter.
      - The resulting outcome metric for each test value.
    
    Parameters:
      param_name (str): Name of the parameter to test.
      test_values (list): List of values to test.
      outcome_key (str): Key from summary_stats to capture.
      
    Returns:
      DataFrame: Two columns with the test values and corresponding outcomes.
    """
    results = []
    for val in test_values:
        temp_params = default_params.copy()
        temp_params[param_name] = val
        _, _, _, _, stats = run_ev_charger_model(temp_params)
        results.append([val, stats[outcome_key]])
    return pd.DataFrame(results, columns=[param_name, outcome_key])


# ========================
# 3. Streamlit Web Interface
# ========================
def main():
    """
    Streamlit web interface for the EV Charger RAB Model.
    
    This dashboard allows non-Python users to interactively adjust key model parameters,
    view the resulting outputs (rollout, RAB evolution, revenue & bill impacts), and explore
    sensitivity analyses using tornado charts.
    
    Financial Concepts Explained:
      - The model simulates a phased rollout of EV chargers (e.g. 5,000 per year for 6 years),
        and calculates how capital expenditures are depreciated over an asset’s life.
      - It tracks the evolution of the Regulatory Asset Base (RAB) which, together with
        operating expenses and financing costs (WACC), determines the revenue requirement.
      - This revenue requirement is then passed through to households as an increased bill.
      - Sensitivity analysis helps identify which parameters (e.g., cost of capital, number of chargers, capital costs)
        most influence the outcomes such as average bill impact or total cost to a household.
    """
    st.title("EV Charger RAB Model Interactive Dashboard")
    st.markdown("### Overview")
    st.write("""
        This dashboard simulates the rollout of EV chargers and calculates the Regulatory Asset Base (RAB), 
        revenue requirements, and subsequent bill impacts on households. Use the sidebar to adjust model inputs,
        and view interactive charts and sensitivity analyses.
    """)

    # Sidebar for user inputs
    st.sidebar.header("Model Inputs")
    chargers_per_year = st.sidebar.number_input("Chargers Installed per Year (Years 1-6)", min_value=1000, max_value=10000, value=default_params["ChargersPerYear"], step=500)
    capex_per_charger = st.sidebar.number_input("CapEx per Charger (in $)", min_value=1000, max_value=10000, value=default_params["CapExPerCharger"], step=500)
    opex_per_charger = st.sidebar.number_input("OpEx per Charger (in $)", min_value=100, max_value=1000, value=default_params["OpExPerCharger"], step=50)
    asset_life = st.sidebar.number_input("Asset Life (Years)", min_value=5, max_value=20, value=default_params["AssetLife"])
    wacc1to5 = st.sidebar.slider("WACC (Years 1-5)", min_value=0.03, max_value=0.1, value=default_params["WACC1to5"], step=0.005)
    wacc6to10 = st.sidebar.slider("WACC (Years 6-10)", min_value=0.03, max_value=0.1, value=default_params["WACC6to10"], step=0.005)
    wacc11to15 = st.sidebar.slider("WACC (Years 11-15)", min_value=0.03, max_value=0.1, value=default_params["WACC11to15"], step=0.005)
    customer_base = st.sidebar.number_input("Customer Base", min_value=500000, max_value=5000000, value=default_params["CustomerBase"], step=100000)

    # Update default parameters from user input
    user_params = default_params.copy()
    user_params.update({
        "ChargersPerYear": chargers_per_year,
        "CapExPerCharger": capex_per_charger,
        "OpExPerCharger": opex_per_charger,
        "AssetLife": asset_life,
        "WACC1to5": wacc1to5,
        "WACC6to10": wacc6to10,
        "WACC11to15": wacc11to15,
        "CustomerBase": customer_base,
    })

    # Run the model
    rollout_df, depr_df, rab_df, rev_df, summary_stats = run_ev_charger_model(user_params)

    st.markdown("### Summary Metrics (Base Scenario)")
    st.write(summary_stats)

    # ---------------------------
    # Visualization Section
    # ---------------------------
    st.markdown("### Model Visualizations")

    # (A) Rollout Chart: Chargers Installed per Year
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Year", y="ChargersInstalled", data=rollout_df, color="steelblue", ax=ax1)
    ax1.set_title("Chargers Installed per Year")
    ax1.set_ylabel("Chargers Installed")
    st.pyplot(fig1)

    # (B) RAB Evolution: Closing RAB vs. Year
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="Year", y="ClosingRAB", data=rab_df, marker="o", color="darkgreen", ax=ax2)
    ax2.set_title("RAB Evolution (Closing RAB)")
    ax2.set_ylabel("Closing RAB ($)")
    st.pyplot(fig2)

    # (C) Revenue & Bill Impact: Annual Bill Impact per Customer
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="Year", y="BillImpactPerCustomer", data=rev_df, marker="o", color="purple", ax=ax3)
    ax3.set_title("Annual Bill Impact per Customer")
    ax3.set_ylabel("Bill Impact ($)")
    st.pyplot(fig3)

    # (D) Total Cumulative Cost to All Households over Time
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="Year", y="TotalCostAllCustomers", data=rev_df, marker="o", color="indigo", ax=ax4)
    ax4.set_title("Total Cumulative Cost to All Households")
    ax4.set_ylabel("Total Cost ($)")
    st.pyplot(fig4)

    # ---------------------------
    # Sensitivity Analysis
    # ---------------------------
    st.markdown("### Sensitivity Analysis")
    st.write("Below are tornado charts showing how changes in key parameters affect household outcomes.")

    # Define sensitivity test values for three parameters
    sensitivity_params = {
        "WACC1to5": [0.0425, 0.0475, 0.0525, 0.0575, 0.0625, 0.0675],
        "ChargersPerYear": [2500, 3750, 5000, 6250, 7500],
        "CapExPerCharger": [4500, 5250, 6000, 6750, 7500]
    }

    # Create a tornado chart for Total Cumulative Cost (Base Scenario)
    base_total_cost = run_ev_charger_model(default_params)[-1]["Total Cumulative Cost to All Customers"]

    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (param, test_vals) in enumerate(sensitivity_params.items()):
        df = sensitivity_analysis(param, test_vals, outcome_key="Total Cumulative Cost to All Customers")
        df["Delta"] = df["Total Cumulative Cost to All Customers"] - base_total_cost
        axes5[idx].barh(df[param].astype(str), df["Delta"], color="teal")
        axes5[idx].axvline(0, color="black", linewidth=1)
        axes5[idx].set_title(f"Sensitivity: {param}")
        axes5[idx].set_xlabel("Delta Total Cost ($)")
        axes5[idx].set_ylabel(param)
    st.pyplot(fig5)

    # Additional Sensitivity Analyses for Household Outcomes:
    # (1) Average Annual Bill Impact per Customer and
    # (2) Cumulative Bill Impact per Customer.
    outcome_metrics = {
        "Average Annual Bill Impact": "Average Annual Bill Impact",
        "Cumulative Bill Impact/Customer": "Cumulative Bill Impact/Customer"
    }
    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
    axes6 = axes6.flatten()
    for j, (outcome_key, outcome_label) in enumerate(outcome_metrics.items()):
        for idx, (param, test_vals) in enumerate(sensitivity_params.items()):
            df = sensitivity_analysis(param, test_vals, outcome_key=outcome_key)
            base_val = run_ev_charger_model(default_params)[-1][outcome_key]
            df["Delta"] = df[outcome_key] - base_val
            ax = axes6[j*3 + idx]
            ax.barh(df[param].astype(str), df["Delta"], color="coral")
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(f"{param}\n({outcome_label})")
            ax.set_xlabel("Delta")
            ax.set_ylabel(param)
    st.pyplot(fig6)


# ========================
# 4. Run the App
# ========================
if __name__ == "__main__":
    main()