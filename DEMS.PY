import numpy as np
import matplotlib.pyplot as plt

# Constants
T = 24  # Time periods
SOC_min, SOC_max = 0, 100  # Battery capacity bounds (kWh)
SOC_0 = 0  # Initial SOC
eta = 0.9  # Battery efficiency
delta_t = 1  # Time interval (hours)
E_max = 15  # Max energy transfer per period (kWh)
max_grid_supply = 15  # Grid limit (kW)
unmet_demand_penalty = 100  # Penalty for unmet demand (arbitrary large value)
gamma = 0.9  # Discount factor

# Generate random energy prices (IP) with negative and positive values
np.random.seed(0)
IP = np.random.uniform(-0.5, 1, size=T)  # Energy price in the imbalance market

# Generate random energy consumption
energy_consumption = np.random.randint(5, 10, size=T)

# Discretize state and action space
SOC_states = np.linspace(SOC_min, SOC_max, 101)  # State space for SOC
actions = np.linspace(-E_max, E_max, 31)  # Actions: charge (-) or discharge (+)

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
            energy_consumed = abs(a_t * delta_t)  # Energy charged/discharged

            # Check feasibility of SOC_next and energy constraints
            if SOC_min <= SOC_next <= SOC_max and energy_consumed <= E_max:
                # Calculate battery supply and grid demand
                battery_supply = min(SOC_t, energy_consumption[t])  # Energy supplied by battery
                grid_demand = energy_consumption[t] - battery_supply  # Remaining demand from grid
                grid_used = min(grid_demand, max_grid_supply)  # Grid limit
                unmet_demand = max(0, grid_demand - max_grid_supply)  # Unmet demand

                # Revenue from selling/buying energy
                grid_profit = -grid_used * IP[t]  # Negative IP means profit
                unmet_demand_cost = unmet_demand * unmet_demand_penalty  # Unmet demand penalty

                # Reward function (profit-based)
                reward = grid_profit - unmet_demand_cost  # Profit - penalties
                future_index = int(SOC_next - SOC_min)  # Index for next SOC state
                future_value = gamma * V[t + 1, future_index]  # Future value
                
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

    # Update SOC based on the action
    SOC_t = SOC_t + eta * action * delta_t  # Update SOC
    SOC_t = max(SOC_min, min(SOC_t, SOC_max))  # Clamp SOC within bounds

    # Meet energy consumption
    battery_supply = min(SOC_t, energy_consumption[t])  # Supply from battery
    grid_demand = energy_consumption[t] - battery_supply  # Remaining demand
    grid_used = min(grid_demand, max_grid_supply)  # Limited by grid capacity
    unmet_demand = max(0, grid_demand - max_grid_supply)  # Unmet demand

    # Calculate profits
    grid_profit = -grid_used * IP[t]  # Negative IP means profit
    grid_profits.append(grid_profit)

    SOC_t -= battery_supply  # Update SOC after supplying energy
    SOC_trajectory.append(SOC_t)

    battery_supply_used.append(battery_supply)
    grid_supply_used.append(grid_used)
    unmet_demand_list.append(unmet_demand)

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


# Plot energy consumption breakdown
time_of_day = np.arange(T)  # Time steps for energy consumption

plt.figure(figsize=(12, 6))

# Stacked bar plot: Battery supply and grid supply
plt.bar(time_of_day, battery_supply_used, label='Battery Supply', color='skyblue', edgecolor='black')
plt.bar(time_of_day, grid_supply_used, bottom=battery_supply_used, label='Grid Supply', color='salmon', edgecolor='black')

# Plot formatting
plt.xlabel('Time of Day (hours)', fontsize=12)
plt.ylabel('Energy (kWh)', fontsize=12)
plt.title('Energy Consumption Breakdown by Source', fontsize=14)
plt.xticks(time_of_day, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# Evaluate total profit
total_profit = sum(grid_profits)
print(f"Total Profit: {total_profit:.2f}")
