import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

# Constants
T = 24  # Time periods
SOC_min, SOC_max = 0, 100  # Battery capacity bounds (kWh)
SOC_0 = 0  # Initial SOC
eta = 0.9  # Battery efficiency
delta_t = 1  # Time interval (hours)
E_max = 15  # Max energy transfer per period (kWh)
max_grid_supply = 15  # Grid limit (kW)
unmet_demand_penalty = 10000  # Penalty for unmet demand (arbitrary large value)
gamma = 0.9  # Discount factor

#Energy bought on main market
E_bought=5 #kWh for every T period
E_bought_price=0.3 #$/kWh

# Generate random energy prices (IP) with negative and positive values

IP = np.random.uniform(-0.5, 1, size=T)  # Energy price in the imbalance market

# Generate random energy consumption
energy_consumption = np.random.randint(0, 15, size=T)

# Discretize state and action space
SOC_states = np.linspace(SOC_min, SOC_max, 101)  # State space for SOC
actions = np.linspace(-E_max, E_max, 31)  # Actions: charge (-) or discharge (+)
actions = np.append(actions, 0)  # Add the option to do nothing (0 action)

# Initialize value function and policy
V = np.zeros((T + 1, len(SOC_states)))
policy = np.zeros((T, len(SOC_states)))

# Dynamic Programming Algorithm
for t in range(T - 1, -1, -1):  # Iterate backward in time
    for soc_index, SOC_t in enumerate(SOC_states):
        Q_values = []  # Store Q-values for all actions at (t, SOC_t)
        for action_index, a_t in enumerate(actions):
            # Compute SOC_{t+1}
            SOC_next = SOC_t + eta * a_t * delta_t
            energy_transferred = abs(a_t * delta_t)  # Energy charged/discharged

            # Check feasibility of SOC_next and energy constraints
            if SOC_min <= SOC_next <= SOC_max and energy_transferred <= E_max:

                # Step 1: Use pre-purchased energy for household demand
                pre_purchased_used = min(E_bought, energy_consumption[t])
                remaining_demand = energy_consumption[t] - pre_purchased_used

                # Step 2: Battery operation
                if a_t > 0:  # Discharging
                    battery_supply = min(SOC_t, remaining_demand)  # Meet house demand first
                    remaining_demand -= battery_supply

                    # Sell surplus energy to the grid
                    surplus_energy = max(0, a_t - battery_supply)
                    market_profit = surplus_energy * IP[t]  # Profit/cost from selling energy

                elif a_t < 0:  # Charging
                    market_profit = abs(a_t) * -IP[t]  # Profit from charging when IP < 0
                    battery_supply = 0  # No discharging while charging

                else:  # Idle
                    market_profit = 0
                    battery_supply = 0

                # Step 3: Grid usage
                grid_used = min(remaining_demand, max_grid_supply)  # Grid meets remaining demand
                unmet_demand = max(0, remaining_demand - grid_used)  # Remaining unmet demand

                # Step 4: Reward calculation
                energy_bought_cost = E_bought * E_bought_price  # Cost of pre-purchased energy
                unmet_demand_cost = unmet_demand * unmet_demand_penalty  # Penalty for unmet demand
                reward = market_profit - energy_bought_cost - unmet_demand_cost

                # Step 5: Future value
                future_index = int((SOC_next - SOC_min) / (SOC_max - SOC_min) * (len(SOC_states) - 1))
                future_value = gamma * V[t + 1, future_index] if 0 <= future_index < len(SOC_states) else -np.inf

                Q_values.append(reward + future_value)  # Bellman equation
            else:
                Q_values.append(-np.inf)  # Invalid action penalty

        # Determine optimal action and value
        optimal_action_index = np.argmax(Q_values)
        policy[t, soc_index] = optimal_action_index  # Store optimal action
        V[t, soc_index] = Q_values[optimal_action_index]  # Update value function



# Simulate SOC trajectory for SOC_0
SOC_t = SOC_0
SOC_trajectory = [SOC_t]  # Initialize trajectory
battery_supply_used = []  # Track battery usage
grid_supply_used = []  # Track grid usage
grid_profits = []  # Track profits
unmet_demand_list = []  # Track unmet demand
optimal_actions = []  # Track actions taken

for t in range(T):
    # Find the current SOC state index
    soc_index = np.argmin(abs(SOC_states - SOC_t))
    
    # Determine the optimal action for the current state and time
    optimal_action_index = int(policy[t, soc_index])
    action = actions[optimal_action_index]
    optimal_actions.append(action)  # Save the action

    # Step 1: Use pre-purchased energy
    pre_purchased_used = min(E_bought, energy_consumption[t])
    remaining_demand = energy_consumption[t] - pre_purchased_used

    # Step 2: Battery operation
    if action > 0:  # Discharging
        battery_supply = min(SOC_t, remaining_demand)
        remaining_demand -= battery_supply

        # Sell surplus energy to the grid
        surplus_energy = max(0, action - battery_supply)
        market_profit = surplus_energy * IP[t]

    elif action < 0:  # Charging
        market_profit = abs(action) * -IP[t]  # Profit from charging (if IP < 0)
        battery_supply = 0

    else:  # Idle
        market_profit = 0
        battery_supply = 0

    # Step 3: Grid usage
    grid_used = min(remaining_demand, max_grid_supply)
    grid_profits.append(market_profit)  # Track grid profits
    unmet_demand = max(0, remaining_demand - grid_used)

    # Step 4: Update SOC
    SOC_t += eta * action * delta_t  # Update SOC based on action
    SOC_t = max(SOC_min, min(SOC_t, SOC_max))  # Clamp SOC within bounds
    SOC_trajectory.append(SOC_t)

    # Record results
    battery_supply_used.append(battery_supply)
    grid_supply_used.append(grid_used)
    unmet_demand_list.append(unmet_demand)



# Calculate total profit of the day
total_profit_day = sum(grid_profits) - sum(unmet_demand_list) * unmet_demand_penalty - T * E_bought * E_bought_price
print(f"Total profit of the day: ${total_profit_day:.2f}")

# Plot daily schedule
time_of_day = np.arange(T + 1)  # Time steps
plt.figure(figsize=(10, 6))
plt.plot(time_of_day, SOC_trajectory, marker='o', label='SOC Level')
plt.xlabel('Time of Day (hours)', fontsize=12)
plt.ylabel('State of Charge (SOC)', fontsize=12)
plt.title('Daily Battery Schedule with Grid Supply Constraint', fontsize=14)
plt.xticks(time_of_day)
plt.ylim(SOC_min, SOC_max)
plt.grid(True)
plt.legend()
plt.show()

# Plot energy sources and usage
plt.figure(figsize=(12, 8))
time_of_day = np.arange(T)

plt.subplot(2, 1, 1)
plt.plot(time_of_day, energy_consumption, marker='o', label='Energy Consumption')
plt.plot(time_of_day, battery_supply_used, marker='o', label='Battery Supply')
plt.plot(time_of_day, grid_supply_used, marker='o', label='Grid Supply')
plt.xlabel('Time of Day (hours)', fontsize=12)
plt.ylabel('Energy (kWh)', fontsize=12)
plt.title('Energy Consumption and Supply Sources', fontsize=14)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_of_day, optimal_actions, marker='o', label='Optimal Actions')
plt.xlabel('Time of Day (hours)', fontsize=12)
plt.ylabel('Action (kWh)', fontsize=12)
plt.title('Optimal Actions Over Time', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()