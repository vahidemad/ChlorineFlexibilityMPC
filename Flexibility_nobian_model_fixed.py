"""
Chlorine production flexibility at Nobian: rolling-horizon (MPC) simulation.

Run:      streamlit run Flexibility_nobian_model_fixed.py
Packages: pip install streamlit numpy pandas matplotlib openpyxl "pulp<3.0.0" highspy
          (without highspy the app falls back to CBC: same results, but slower)

What changed compared with the bookkeeping version (numbers = the review points)
  1. Closed year. Each scenario runs through the price year again and again, with windows that
     wrap around from the end of the year to its start, until the tank level at the end of a
     year equals the level at its start. The last simulated year is reported. If the yearly
     change stops shrinking (for example with a very large tank), the run stops.
  2. The weighted-average-cost bookkeeping is removed (this also covers point 3). If a year does
     not close, the remaining tank change is valued at that year's average production cost.
  4. One input for the initial tank level. It is only the starting guess for year 1.
  5. Tie-break: the chlorine value gets +0.001 EUR/t. An hour priced exactly at the threshold
     price then runs at P_max, as the definition of N_max assumes. This only chooses between
     plans with the same net cost.
  6. N_max = round(8,760 x 4/7) = round(5,005.7) = 5,006 hours. The app prints it under the table.
  7. Names follow the report:
       C_max -> P_max, L_min_ratio -> L_min, L_min_tons -> P_min, D_const -> D, V_max -> S_max,
       energy_intensity -> E, lookahead -> valuation window (W_V),
       terminal value -> chlorine value (V_Cl), price threshold -> threshold price,
       'Dumb 1h (Static)' -> 'Baseline (static threshold)', storage -> tank level,
       rigid cost -> constant production cost.
  8. Speed: HiGHS runs inside Python (CBC starts a separate program for every hourly LP), every
     LP is checked to be solved to optimality, and results are cached: a scenario that was
     already run with the same settings is not simulated again.

Added: perfect-foresight benchmark. One LP for the whole year with all prices known, in which
the tank ends the year at the level it started with (closed year). It gives the maximum possible
savings with this plant and tank. When it is selected, the table shows every scenario's savings
as a share of this maximum.

Added: realistic MPC. Uses only prices that are known at the moment of decision. The day-ahead
prices of the next day are published once a day (around 12:45), so from 13:00 on they are known
until the end of the next day, and before 13:00 only until the end of the same day. Every day at
13:00 a new plan is made over all known prices (35 hours on a normal day) and carried out hour by
hour until the next day's plan. Its chlorine value comes from the PAST valuation window (the last
W_V days), not from future prices. Days follow the clock in the price file (23 or 25 hours on the
days the clock changes).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import streamlit as st

st.set_page_config(page_title="Nobian MPC Simulation", layout="wide")
st.title("Chlorine Production Flexibility: Forecast Horizon Analysis")

TIE_BREAK = 1e-3  # EUR/t added to the chlorine value (point 5)
BASELINE = 'Baseline (static threshold)'
REALISTIC = 'Realistic MPC'
PERFECT = 'Perfect foresight (whole year)'
PUBLICATION_HOUR = 13  # realistic MPC: from this hour on, the next day's day-ahead prices are known


# ==========================================
# MODEL FUNCTIONS
# ==========================================
def get_solver(solver_name):
    """Return (solver, name of the solver used). HiGHS needs `pip install highspy`."""
    if solver_name == "HiGHS":
        highs = getattr(pulp, "HiGHS", None)
        if highs is not None and highs(msg=False).available():
            return highs(msg=False), "HiGHS"
    cbc = getattr(pulp, "PULP_CBC_CMD", None) or pulp.COIN_CMD  # PULP_CBC_CMD is removed in PuLP 4
    return cbc(msg=0), "CBC"


def solve_plan(t, forecast_prices, S_0, chlorine_value, P_min, P_max, D, S_max, E, solver):
    """Optimizer (Step 3 in the report): the production plan with the lowest net cost Z over the
    forecast horizon. Returns the planned production rate of every hour of the horizon. The MPC
    carries out only the first hour; the realistic MPC carries out the plan until its next plan."""
    W_F = len(forecast_prices)
    model = pulp.LpProblem(f"MPC_{t}", pulp.LpMinimize)
    P = pulp.LpVariable.dicts(f"P_t{t}", range(W_F), lowBound=P_min, upBound=P_max)
    S = pulp.LpVariable.dicts(f"S_t{t}", range(W_F), lowBound=0, upBound=S_max)

    # Net cost Z = electricity cost of the forecast horizon - value of the chlorine in the tank at its end
    model += (pulp.lpSum(P[i] * float(E * forecast_prices[i]) for i in range(W_F))
              - S[W_F - 1] * float(chlorine_value))

    # Tank balance: level at the end of hour i = level at the end of hour i-1 + production - demand
    for i in range(W_F):
        previous_level = float(S_0) if i == 0 else S[i - 1]
        model += S[i] == previous_level + P[i] - D

    model.solve(solver)
    if model.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Hour {t}: the solver returned '{pulp.LpStatus[model.status]}' instead of 'Optimal'.")
    return np.array([P[i].varValue for i in range(W_F)])


def known_price_hours(clock_hours, publication_hour):
    """Realistic MPC: for every hour, the number of hours (from this hour on) whose prices are known.
    Before `publication_hour` the prices are known until the end of the same day, from then on until
    the end of the next day. Days follow the clock in the price file (23 or 25 hours when the clock
    changes). The year is cyclic: after its last hour it continues with its first hour."""
    hours_in_year = len(clock_hours)
    # A new day starts where the clock goes back (23:00 -> 00:00). This also works for a day whose 00:00
    # is missing in the file. Where the year wraps around (00:00 -> 00:00) no new day starts.
    day_starts = np.flatnonzero(clock_hours < np.roll(clock_hours, 1))
    if len(day_starts) == 0:  # no clock information: use 24-hour days
        day_starts = np.arange(0, hours_in_year, 24)
    # Day starts of three years in a row, so that the start of the day after tomorrow always exists
    day_starts = np.concatenate([day_starts, day_starts + hours_in_year, day_starts + 2 * hours_in_year])
    t = np.arange(hours_in_year)
    end_of_today = day_starts[np.searchsorted(day_starts, t, side='right')]
    end_of_tomorrow = day_starts[np.searchsorted(day_starts, end_of_today, side='right')]
    known_until = np.where(clock_hours >= publication_hour, end_of_tomorrow, end_of_today)
    return known_until - t


@st.cache_data(show_spinner=False)
def simulate_scenario(prices, P_max, P_min, D, S_max, E, f_max, baseline_chlorine_value,
                      forecast_horizon, use_dynamic_threshold, valuation_window_hrs,
                      S_initial, tolerance, max_years, solver_name, tie_break,
                      clock_hours=None, publication_hour=PUBLICATION_HOUR):
    """Run the price year again and again until the year closes (point 1).
    Returns the results of the last simulated year.
    With clock_hours (the hour of the day of every price) it runs the realistic MPC."""
    hours_in_year = len(prices)
    solver, solver_used = get_solver(solver_name)
    N_max_window = round(valuation_window_hrs * f_max)  # full-load hours in the valuation window
    k = max(0, N_max_window - 1)                        # position of the threshold price in the sorted window
    realistic = clock_hours is not None
    if realistic:
        known_hours = known_price_hours(clock_hours, publication_hour)
    started = time.perf_counter()

    def simulate_year(S_start, plan):
        prod = np.zeros(hours_in_year)
        tank = np.zeros(hours_in_year)
        cost = np.zeros(hours_in_year)
        S_0 = float(S_start)
        for t in range(hours_in_year):
            if realistic:
                # A new plan when the next day's prices are published (and at the very start)
                if clock_hours[t] == publication_hour or len(plan) == 0:
                    known_prices = prices[(t + np.arange(known_hours[t])) % hours_in_year]
                    # Valuation window W_V: the PAST W_V hours, up to and including this hour
                    window_prices = prices[(t - valuation_window_hrs + 1 + np.arange(valuation_window_hrs)) % hours_in_year]
                    chlorine_value = np.sort(window_prices)[k] * E + tie_break
                    plan = list(solve_plan(t, known_prices, S_0, chlorine_value, P_min, P_max, D, S_max, E, solver))
                P_now = plan.pop(0)  # carry out the plan hour by hour until the next plan is made
            else:
                # Forecast horizon W_F: the next W_F hours. After the last hour of the year it
                # continues with the first hours of the year (wrap-around).
                forecast_prices = prices[(t + np.arange(forecast_horizon)) % hours_in_year]

                if use_dynamic_threshold:
                    # Valuation window W_V: the next W_V hours, also with wrap-around
                    window_prices = prices[(t + np.arange(valuation_window_hrs)) % hours_in_year]
                    threshold_price = np.sort(window_prices)[k]
                    chlorine_value = threshold_price * E + tie_break
                else:
                    chlorine_value = baseline_chlorine_value + tie_break

                P_now = solve_plan(t, forecast_prices, S_0, chlorine_value, P_min, P_max, D, S_max, E, solver)[0]
            prod[t] = P_now
            S_0 = S_0 + P_now - D
            tank[t] = S_0
            cost[t] = P_now * E * prices[t]
        return prod, tank, cost, plan

    level = float(S_initial)
    plan = []  # realistic MPC: the rest of the last plan of a year is carried out at the start of the next year
    previous_change = None
    for year in range(1, max_years + 1):
        S_start = level
        prod, tank, cost, plan = simulate_year(S_start, plan)
        level = float(tank[-1])
        change = level - S_start
        if abs(change) <= tolerance:
            status = "closed"
            break
        if previous_change is not None and abs(change) >= abs(previous_change) - tolerance:
            status = "does not close"  # the change stopped shrinking, so more years will not help
            break
        previous_change = change
    else:
        status = "max years reached"

    # Correction for a year that does not close (point 2): value the tank change at the
    # average production cost of that year. For a closed year the correction is ~0.
    electricity_cost = float(np.sum(cost))
    average_cost_per_t = electricity_cost / float(np.sum(prod))
    correction = -change * average_cost_per_t

    return {
        'prod': prod, 'tank': tank, 'years': year, 'status': status,
        'S_start': S_start, 'S_end': level, 'change': change,
        'electricity_cost': electricity_cost, 'correction': correction,
        'annual_cost': electricity_cost + correction,
        'solver': solver_used, 'seconds': time.perf_counter() - started,
    }


@st.cache_data(show_spinner=False)
def perfect_foresight(prices, P_max, P_min, D, S_max, E, solver_name):
    """Benchmark: one LP for the whole year with all prices known (no rolling horizon, no chlorine value).
    Closed year: hour 1 starts with the tank level at the end of the year, so the tank ends the
    year at the level it started with. The optimizer also chooses this level."""
    hours_in_year = len(prices)
    solver, solver_used = get_solver(solver_name)
    started = time.perf_counter()

    model = pulp.LpProblem("Perfect_foresight", pulp.LpMinimize)
    P = pulp.LpVariable.dicts("P", range(hours_in_year), lowBound=P_min, upBound=P_max)
    S = pulp.LpVariable.dicts("S", range(hours_in_year), lowBound=0, upBound=S_max)

    # Electricity cost of the whole year (the tank ends where it started, so no value for the stock is needed)
    model += pulp.lpSum(P[t] * float(E * prices[t]) for t in range(hours_in_year))

    # Tank balance; for the first hour the previous level is the level at the end of the year
    for t in range(hours_in_year):
        previous_level = S[hours_in_year - 1] if t == 0 else S[t - 1]
        model += S[t] == previous_level + P[t] - D

    model.solve(solver)
    if model.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Perfect foresight: the solver returned '{pulp.LpStatus[model.status]}' instead of 'Optimal'.")

    prod = np.array([P[t].varValue for t in range(hours_in_year)])
    tank = np.array([S[t].varValue for t in range(hours_in_year)])
    electricity_cost = float(np.sum(prod * E * prices))
    return {
        'prod': prod, 'tank': tank, 'years': 1, 'status': "closed (whole-year LP)",
        'S_start': float(tank[-1]), 'S_end': float(tank[-1]), 'change': 0.0,
        'electricity_cost': electricity_cost, 'correction': 0.0, 'annual_cost': electricity_cost,
        'solver': solver_used, 'seconds': time.perf_counter() - started,
    }


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'Price duration curves ETM & heat maps.xlsx'", type=['xlsx'])

st.sidebar.header("2. Plant and Tank")
P_max = st.sidebar.number_input("P_max: maximum production rate (t/h)", value=68.0, step=1.0)
L_avg = st.sidebar.slider("L_avg: average load factor", min_value=0.1, max_value=1.0, value=0.70, step=0.01)
L_min = st.sidebar.slider("L_min: minimum load factor", min_value=0.1, max_value=1.0, value=0.30, step=0.01)
S_max = st.sidebar.number_input("S_max: tank capacity (t)", value=7000.0, step=100.0)
S_initial = st.sidebar.number_input(
    "Initial tank level (t): only the starting guess for year 1",
    min_value=0.0, max_value=float(S_max), value=float(S_max) / 2.0, step=100.0)
E = st.sidebar.number_input("E: specific electricity consumption (MWh/t)", value=2.875, step=0.001, format="%.3f")

st.sidebar.header("3. Model Settings")
valuation_input = st.sidebar.text_input(
    "Valuation windows W_V in days (comma-separated):",
    value="1, 2, 7, 14, 30"
)

# Parse the comma-separated string into a list of whole numbers of days
selected_window_days = []
if valuation_input:
    try:
        selected_window_days = [int(x.strip()) for x in valuation_input.split(',') if x.strip()]
        if any(days <= 0 for days in selected_window_days):
            raise ValueError
    except ValueError:
        selected_window_days = []
        st.sidebar.error("Please enter positive whole numbers separated by commas (e.g., 7, 14, 30)")

solver_name = st.sidebar.selectbox(
    "Solver", ["HiGHS", "CBC"], index=0,
    help="Both find the same optimal plan. HiGHS runs inside Python and is faster (needs highspy in requirements.txt).")
tolerance = st.sidebar.number_input(
    "Closed-year tolerance (t)", min_value=0.0, value=0.5, step=0.1,
    help="A year is closed when the tank level at its end is within this many tons of the level at its start.")
max_years = int(st.sidebar.number_input("Maximum number of simulated years", min_value=1, value=5, step=1))

st.sidebar.header("4. Select Scenarios")
all_scenarios = {
    BASELINE: {'h': 1, 'dyn': False},
    'MPC 1h': {'h': 1, 'dyn': True},
    'MPC 12h': {'h': 12, 'dyn': True},
    'MPC 24h': {'h': 24, 'dyn': True},
    'MPC 48h': {'h': 48, 'dyn': True},
    'MPC 168h': {'h': 168, 'dyn': True},
    REALISTIC: {'h': None, 'dyn': True, 'real': True},
    PERFECT: {'h': None, 'dyn': False, 'pf': True}
}
selected_scenario_names = st.sidebar.multiselect(
    "Choose scenarios to simulate (MPC 24h = forecast horizon W_F of 24 hours):",
    options=list(all_scenarios.keys()),
    default=[BASELINE, 'MPC 24h', REALISTIC, PERFECT],
    help=f"Realistic MPC: only prices known at the time. A new plan every day at {PUBLICATION_HOUR}:00, when the "
         "next day's prices are known; chlorine value from the PAST valuation window. "
         "Perfect foresight: all prices of the year known, which gives the maximum possible savings."
)

run_clicked = st.sidebar.button("Run Simulation")
# At least one run: the baseline and perfect foresight need no valuation window, the MPC scenarios
# (also the realistic MPC) do
ready = (uploaded_file is not None
         and any((not all_scenarios[name]['dyn']) or len(selected_window_days) > 0
                 for name in selected_scenario_names))

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if run_clicked and ready:

    # Derived constants
    D = P_max * L_avg                        # demand (t/h)
    P_min = P_max * L_min                    # minimum production rate (t/h)
    f_max = (L_avg - L_min) / (1.0 - L_min)  # fraction of full-load hours

    # Load Data
    with st.spinner('Loading ENTSO-E Price Data...'):
        df_prices = pd.read_excel(uploaded_file, sheet_name='ENTSOE-E dayahead mid2425')
        df_prices['datetime'] = pd.to_datetime(df_prices['date'].dt.strftime('%Y-%m-%d') + ' ' + df_prices['time'].str[:8])
        df_prices.set_index('datetime', inplace=True)
        df_prices.sort_index(inplace=True)
        prices_eur_mwh = df_prices['price €/MWh'].values.astype(float)
        clock_hours = df_prices.index.hour.values  # hour of the day (local clock), for the realistic MPC
        hours_in_year = len(prices_eur_mwh)

    # Threshold price of the baseline model (point 6: round(8,760 x 4/7) = 5,006)
    N_max_year = round(hours_in_year * f_max)
    baseline_threshold_price = np.sort(prices_eur_mwh)[N_max_year - 1]
    baseline_chlorine_value = baseline_threshold_price * E

    # Generate Run Combinations (Cartesian Product)
    runs_to_execute = []
    for name in selected_scenario_names:
        s = all_scenarios[name]
        if s.get('pf'):
            continue  # added last, below
        if not s['dyn']:
            runs_to_execute.append({'label': name, 'h': s['h'], 'dyn': False, 'W_V_hrs': 336})
        elif s.get('real'):
            for days in selected_window_days:
                label = f"{name} | {days}d past window"
                runs_to_execute.append({'label': label, 'h': None, 'dyn': True, 'W_V_hrs': days * 24, 'real': True})
        else:
            for days in selected_window_days:
                label = f"{name} | {days}d window"
                runs_to_execute.append({'label': label, 'h': s['h'], 'dyn': True, 'W_V_hrs': days * 24})
    if PERFECT in selected_scenario_names:
        runs_to_execute.append({'label': PERFECT, 'pf': True})

    # Execute Scenarios
    results_data = {}
    constant_production_cost = float(np.sum(D * E * prices_eur_mwh))

    # Dynamic color mapping
    cmap = plt.get_cmap('tab10')
    colors = {run['label']: cmap(i % 10) for i, run in enumerate(runs_to_execute)}
    if BASELINE in colors:
        colors[BASELINE] = 'red'
    if PERFECT in colors:
        colors[PERFECT] = 'dimgray'

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, run in enumerate(runs_to_execute):
        label = run['label']
        status_text.text(f"Running {label} ({i + 1} of {len(runs_to_execute)})...")

        if run.get('pf'):
            res = perfect_foresight(prices_eur_mwh, P_max, P_min, D, S_max, E, solver_name)
        else:
            # TIE_BREAK is passed as an argument so that the cache notices when you change it
            res = simulate_scenario(prices_eur_mwh, P_max, P_min, D, S_max, E, f_max, baseline_chlorine_value,
                                    run['h'], run['dyn'], run['W_V_hrs'], S_initial, tolerance, max_years,
                                    solver_name, TIE_BREAK,
                                    clock_hours if run.get('real') else None, PUBLICATION_HOUR)

        res['prices_at_max'] = prices_eur_mwh[res['prod'] >= P_max - 0.1]
        res['prices_at_min'] = prices_eur_mwh[res['prod'] <= P_min + 0.1]
        res['color'] = colors[label]
        results_data[label] = res
        progress_bar.progress((i + 1) / len(runs_to_execute))

    status_text.text("Simulation Complete!")

    if any(res['solver'] != solver_name for res in results_data.values()):
        st.warning("HiGHS is not available (add highspy to requirements.txt), so CBC was used.")

    # Financial Summary Table (closed year)
    st.subheader("Financial Performance (Closed Year)")
    summary_data = []

    # Maximum possible savings (perfect foresight), if that scenario was run
    max_savings = None
    if PERFECT in results_data:
        pf_savings = constant_production_cost - results_data[PERFECT]['annual_cost']
        if pf_savings > 1.0:  # no share column when there is nothing to save (for example S_max = 0)
            max_savings = pf_savings

    for label, res in results_data.items():
        # Tons moved in and out of the tank during the year, including the first hour
        tank_with_start = np.concatenate(([res['S_start']], res['tank']))
        throughput_tons = np.sum(np.abs(np.diff(tank_with_start)))
        savings = constant_production_cost - res['annual_cost']

        row = {
            "Scenario": label,
            "Years Simulated": res['years'],
            "Year Status": res['status'],
            "Tank Level at Start (t)": f"{res['S_start']:,.0f}",
            "Tank Level at End (t)": f"{res['S_end']:,.0f}",
            "Electricity Cost (€)": f"€{res['electricity_cost']:,.0f}",
            "Tank Change Correction (€)": f"€{round(res['correction']):,}",
            "Annual Cost (€)": f"€{res['annual_cost']:,.0f}",
            "Savings vs Constant Production (€)": f"€{savings:,.0f}",
        }
        if max_savings:
            row["Share of Maximum Savings"] = f"{100 * savings / max_savings:.1f}%"
        row["Tank Throughput (t)"] = f"{throughput_tons:,.0f}"
        row["Run Time (s)"] = f"{res['seconds']:,.0f}"
        summary_data.append(row)

    st.table(pd.DataFrame(summary_data))
    st.caption(
        f"The table shows the last simulated year. "
        f"Constant production ({D:.1f} t/h every hour) costs €{constant_production_cost:,.0f}. "
        f"Baseline threshold price: {baseline_threshold_price:.2f} €/MWh."
    )
    scenario_notes = []
    if any(run.get('real') for run in runs_to_execute):
        scenario_notes.append(
            f"Realistic MPC: uses only prices known at the time. Every day at {PUBLICATION_HOUR}:00, when the next "
            "day's day-ahead prices are known, it makes a new plan over all known prices (until the end of the next "
            "day, 35 hours on a normal day) and carries it out until the next day's plan. Its chlorine value comes "
            "from the past valuation window.")
    if max_savings:
        scenario_notes.append(
            "Perfect foresight: one optimization over the whole year with all prices known, ending at its start "
            "level; its savings are the maximum possible with this plant and tank, and the share column compares "
            "every scenario with it. For a year that does not close the share is only approximate, because the "
            "tank change is valued at the average production cost; it can then even exceed 100%.")
    if scenario_notes:
        st.caption(" ".join(scenario_notes))

    # Visualizations
    st.subheader("Comparative Analysis (Last Simulated Year)")
    fig, axes = plt.subplots(4, 1, figsize=(14, 24))

    # Setup blank histogram figure
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))

    hours = np.arange(hours_in_year)
    for label, res in results_data.items():
        c = res['color']
        tank_array = res['tank']

        axes[0].plot(hours, tank_array, color=c, alpha=0.6, label=label)
        axes[1].plot(hours, np.sort(tank_array)[::-1], color=c, linewidth=2, label=label)

        # Overlay histogram data for each scenario
        ax_hist.hist(tank_array, bins=50, color=c, alpha=0.5, label=label, edgecolor='black')

    axes[0].set_title('Tank Level over the Year', fontweight='bold')
    axes[0].set_ylabel('Tank Level (t)')
    axes[0].axhline(S_max, color='black', linestyle='--', label='Tank Capacity (S_max)')
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Tank Level Duration Curve', fontweight='bold')
    axes[1].set_ylabel('Tank Level (t)')
    axes[1].axhline(S_max, color='black', linestyle='--')
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].grid(True, alpha=0.3)

    prod_arrays = [res['prod'] for res in results_data.values()]
    c_list = [res['color'] for res in results_data.values()]
    axes[2].hist(prod_arrays, bins=10, color=c_list, label=list(results_data.keys()), edgecolor='black')
    axes[2].set_title('Hourly Production Frequency Distribution', fontweight='bold')
    axes[2].set_xlabel('Production Rate (t/h)')
    axes[2].set_ylabel('Frequency (Hours)')
    axes[2].legend(loc='upper center', fontsize='small')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(hours, np.sort(prices_eur_mwh)[::-1], color='black', linewidth=3, label='Overall Market Price', zorder=1)

    for label, res in results_data.items():
        c = res['color']
        p_max_prices = res['prices_at_max']
        p_min_prices = res['prices_at_min']

        axes[3].plot(np.arange(len(p_min_prices)), np.sort(p_min_prices)[::-1], color=c, linestyle=':', linewidth=2,
                     alpha=0.8, label=f'{label} ({L_min:.0%} Load)', zorder=2)

        x_offset = hours_in_year - len(p_max_prices)
        axes[3].plot(np.arange(x_offset, hours_in_year), np.sort(p_max_prices)[::-1], color=c, linestyle='-',
                     linewidth=2, label=f'{label} (100% Load)', zorder=2)

    axes[3].set_title('Electricity Price Duration Curves based on Operational State', fontweight='bold')
    axes[3].set_xlabel('Cumulative Hours')
    axes[3].set_ylabel('Electricity Price (€/MWh)')
    axes[3].legend(loc='upper right', fontsize='small', ncol=2)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Render Tank Utilization Histogram
    st.subheader("Tank Capacity Utilization")
    ax_hist.set_title("Tank Level Distribution over the Year", fontsize=14, fontweight='bold')
    ax_hist.set_xlabel("Tank Level (t)", fontsize=12)
    ax_hist.set_ylabel("Frequency (Hours of the Year)", fontsize=12)
    ax_hist.legend(loc='upper right', fontsize='small')
    ax_hist.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig_hist)

elif uploaded_file is None:
    st.info("Please upload the price dataset via the sidebar to begin.")
elif not ready:
    st.warning("Please select at least one scenario and enter at least one valid valuation window.")
else:
    st.info("Press 'Run Simulation' in the sidebar to start.")
