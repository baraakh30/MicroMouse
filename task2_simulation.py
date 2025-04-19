import numpy as np
import matplotlib.pyplot as plt

#Baraa Khanfar 1210640
#Osama Zeidan 1210601
#Mohammad Qady 1211099
#Sadeen Khatib 1212164

# Simulation parameters
dt = 0.01  # Time step [s]
sim_time = 50  # Total simulation time [s]
steps = int(sim_time / dt)  # Number of steps

# System parameters
r = 0.02  # Wheel radius [m]
d = 0.1  # Distance between wheels [m]
m = 0.2  # Mass [kg]
I = 0.0005  # Moment of inertia [kg·m²]
km = 0.1  # Motor constant [N·m/V]
b = 0.01  # Linear friction coefficient [N·s/m]
b_theta = 0.001  # Angular friction coefficient [N·m·s/rad]

# Target position
target = np.array([0.5, 0.5])

# Sensor noise standard deviations
sigma_encoder = 0.05  # [rad/s]
sigma_gyro = 0.02  # [rad/s]

# EKF parameters
Q = np.diag([0.001, 0.001, 0.001, 0.001, 0.001])  # Process noise covariance
R = np.diag([sigma_encoder**2, sigma_encoder**2, sigma_gyro**2])  # Measurement noise covariance

# Define walls (each is a segment from (x1,y1) to (x2,y2))
walls = [
    [0.25, 0.1, 0.35, 0.2],    # Wall 1: diagonal from (0.25,0.1) to (0.35,0.2)
    [0.4, 0.3, 0.4, 0.4],      # Wall 2: vertical at x=0.4, from y=0.3 to y=0.4
    [0.2, 0.25, 0.3, 0.25]     # Wall 3: horizontal at y=0.25, from x=0.2 to x=0.3
]

# Wall detection parameters
WALL_THRESHOLD = 0.02  # Distance threshold for wall collision [m]
WALL_AVOID_DISTANCE = 0.1 # Distance to start avoiding walls [m]
WALL_LOOK_AHEAD = 5  # Look ahead multiple timesteps for collision prediction

def continuous_dynamics(state, inputs):
    """Continuous-time dynamics for the micromouse"""
    x, y, theta, v, omega = state
    V_left, V_right = inputs
    
    # Clamp inputs to voltage limits
    V_left = np.clip(V_left, -5, 5)
    V_right = np.clip(V_right, -5, 5)
    
    # State derivatives
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = omega
    v_dot = (km * (V_left + V_right)) / (m * r) - (b / m) * v
    omega_dot = (d * km * (V_right - V_left)) / (2 * I * r) - (b_theta / I) * omega
    
    return np.array([x_dot, y_dot, theta_dot, v_dot, omega_dot])

def discrete_dynamics(state, inputs):
    """Discrete-time dynamics using Euler integration"""
    derivatives = continuous_dynamics(state, inputs)
    return state + dt * derivatives

def sensor_model(state):
    """Calculate sensor measurements from state"""
    _, _, _, v, omega = state
    
    # Calculate wheel angular velocities
    omega_left = (v - omega * d / 2) / r
    omega_right = (v + omega * d / 2) / r
    
    # Ideal measurements (without noise)
    measurements = np.array([omega_left, omega_right, omega])
    
    return measurements

def add_sensor_noise(measurements):
    """Add Gaussian noise to sensor measurements"""
    noise = np.random.normal(0, [sigma_encoder, sigma_encoder, sigma_gyro])
    return measurements + noise

def jacobian_F(state, inputs):
    """Jacobian of state transition w.r.t. state for EKF"""
    x, y, theta, v, omega = state
    
    # Initialize Jacobian matrix
    F = np.zeros((5, 5))
    
    # Fill in the Jacobian elements
    F[0, 2] = -v * np.sin(theta) * dt
    F[0, 3] = np.cos(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt
    F[1, 3] = np.sin(theta) * dt
    F[2, 4] = dt
    F[3, 3] = 1 - (b / m) * dt
    F[4, 4] = 1 - (b_theta / I) * dt
    
    # Add identity matrix (for I + F*dt form)
    F = np.eye(5) + F
    
    return F

def jacobian_H(state):
    """Jacobian of measurement model w.r.t. state for EKF"""
    H = np.zeros((3, 5))
    
    # Derivatives of omega_left = (v - omega * d / 2) / r
    H[0, 3] = 1 / r
    H[0, 4] = -d / (2 * r)
    
    # Derivatives of omega_right = (v + omega * d / 2) / r
    H[1, 3] = 1 / r
    H[1, 4] = d / (2 * r)
    
    # Derivative of gyro measurement (omega)
    H[2, 4] = 1
    
    return H

def extended_kalman_filter(state_est, P, measurements, inputs):
    """EKF for state estimation"""
    # Prediction step
    state_pred = discrete_dynamics(state_est, inputs)
    F = jacobian_F(state_est, inputs)
    P_pred = F @ P @ F.T + Q
    
    # Update step
    H = jacobian_H(state_pred)
    predicted_measurements = sensor_model(state_pred)
    y = measurements - predicted_measurements
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    state_est = state_pred + K @ y
    P = (np.eye(5) - K @ H) @ P_pred
    
    return state_est, P

def distance_to_line_segment(p, line):
    """Calculate the minimum distance from point p to line segment defined by line=[x1,y1,x2,y2]"""
    x1, y1, x2, y2 = line
    A = p[0] - x1
    B = p[1] - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    # If line segment is just a point
    if len_sq == 0:
        return np.sqrt(A * A + B * B), np.array([x1, y1])
    
    # Calculate projection parameter
    param = dot / len_sq
    
    # Find nearest point on the line segment
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    # Calculate distance
    dx = p[0] - xx
    dy = p[1] - yy
    return np.sqrt(dx * dx + dy * dy), np.array([xx, yy])

def normal_vector_from_line(line, position):
    """Calculate unit normal vector pointing away from line segment"""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return np.array([0, 0])
    
    # Calculate both normal vectors
    nx1, ny1 = -dy/length, dx/length  # Counter-clockwise normal
    nx2, ny2 = dy/length, -dx/length  # Clockwise normal
    
    # Determine the closest point on the line
    _, closest_point = distance_to_line_segment(position, line)
    
    # Vector from closest point to current position
    away_vector = position - closest_point
    away_norm = np.linalg.norm(away_vector)
    
    if away_norm < 1e-6:  # If we're almost on the line, pick arbitrary direction
        return np.array([nx1, ny1])
    
    # Normalize
    away_vector = away_vector / away_norm
    
    # Check which normal vector points in similar direction to away_vector
    dot1 = np.dot(away_vector, [nx1, ny1])
    dot2 = np.dot(away_vector, [nx2, ny2])
    
    if dot1 > dot2:
        return np.array([nx1, ny1])
    else:
        return np.array([nx2, ny2])

def predict_trajectory(state, inputs, steps=WALL_LOOK_AHEAD):
    """Predict future trajectory given current state and inputs"""
    trajectory = [state[:2].copy()]  # Start with current position
    current_state = state.copy()
    
    for _ in range(steps):
        current_state = discrete_dynamics(current_state, inputs)
        trajectory.append(current_state[:2].copy())
    
    return np.array(trajectory)

def check_collision(state, inputs):
    """Check if the micromouse collides with any wall, including predicted path"""
    # Predict future trajectory
    trajectory = predict_trajectory(state, inputs)
    
    # Check each point on trajectory
    for point in trajectory:
        for wall in walls:
            dist, _ = distance_to_line_segment(point, wall)
            if dist < WALL_THRESHOLD:
                return True, dist
    
    return False, float('inf')

def controller(state_est, target):
    """Controller for the micromouse"""
    x, y, theta, v, omega = state_est
    position = np.array([x, y])
    
    # Calculate direction to target
    dx = target[0] - x
    dy = target[1] - y
    distance_to_target = np.sqrt(dx**2 + dy**2)
    target_heading = np.arctan2(dy, dx)
    
    # Initialize avoiding_wall flag and wall_dist
    avoiding_wall = False
    wall_dist = float('inf')
    
    # Check all walls for proximity
    for wall in walls:
        dist, _ = distance_to_line_segment(position, wall)
        wall_dist = min(wall_dist, dist)
        if dist < WALL_AVOID_DISTANCE:
            avoiding_wall = True
    
    # Get avoidance vector if needed
    avoidance_vector = np.zeros(2)
    if avoiding_wall:
        for wall in walls:
            dist, closest_point = distance_to_line_segment(position, wall)
            if dist < WALL_AVOID_DISTANCE:
                # Get normal vector pointing away from wall
                normal = normal_vector_from_line(wall, position)
                
                # Weight by inverse distance (closer walls have stronger effect)
                weight = 1.0 - (dist / WALL_AVOID_DISTANCE)
                weight = weight**2  # Square for more aggressive avoidance
                
                # Add to avoidance vector
                avoidance_vector += normal * weight
        
        # Normalize if non-zero
        norm = np.linalg.norm(avoidance_vector)
        if norm > 1e-6:
            avoidance_vector /= norm
            
            # Convert to heading
            avoidance_heading = np.arctan2(avoidance_vector[1], avoidance_vector[0])
            
            # Blend with target heading based on wall proximity
            blend_factor = min(1.0, 1.0 - (wall_dist / WALL_AVOID_DISTANCE))
            blend_factor = blend_factor**2  # More aggressive avoidance
            target_heading = (1 - blend_factor) * target_heading + blend_factor * avoidance_heading
    
    # Calculate heading error
    heading_error = target_heading - theta
    # Normalize to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # Calculate desired speed based on conditions
    if distance_to_target < 0.1:
        # Close to target, slow down
        desired_speed = 0.1 * distance_to_target / 0.1
    elif avoiding_wall:
        # Near wall, reduce speed
        desired_speed = 0.1 + 0.1 * (wall_dist / WALL_AVOID_DISTANCE)
    else:
        # Normal operation
        desired_speed = 0.2
    
    # Reduce speed for sharp turns
    turn_factor = 1.0 - min(0.8, abs(heading_error) / np.pi)
    desired_speed *= max(0.3, turn_factor)
    
    # Calculate speed error
    speed_error = desired_speed - v
    
    # Access global PID variables
    global i, speed_error_sum, heading_error_sum, prev_speed_error, prev_heading_error, prev_speed_der, prev_heading_der
    
    # PID gains for speed - tuned values
    Kp_speed = 3.0      # Proportional gain
    Ki_speed = 0.8      # Integral gain
    Kd_speed = 0.4      # Derivative gain
    
    # PID gains for heading - tuned values
    Kp_heading = 2.0    # Proportional gain 
    Ki_heading = 0.2    # Integral gain
    Kd_heading = 0.5    # Derivative gain
    
    # Calculate integral terms
    # Reset integral when error changes sign (crosses zero) to prevent windup
    if prev_speed_error * speed_error < 0:
        speed_error_sum = 0
    if prev_heading_error * heading_error < 0:
        heading_error_sum = 0
    
    # Update integral sum with limits to prevent excessive buildup
    speed_error_sum += speed_error * dt
    speed_error_sum = np.clip(speed_error_sum, -1.0, 1.0)  # Limit integral term
    
    heading_error_sum += heading_error * dt
    heading_error_sum = np.clip(heading_error_sum, -1.0, 1.0)  # Limit integral term
    
    # Calculate derivative terms
    if i == 0:
        # First iteration - no derivative
        speed_der = 0
        heading_der = 0
        prev_speed_der = 0
        prev_heading_der = 0
    else:
        # Calculate derivatives from current and previous errors
        speed_der = (speed_error - prev_speed_error) / dt
        heading_der = (heading_error - prev_heading_error) / dt
        
        # Apply low-pass filter to derivative to reduce noise
        alpha = 0.2  # Filter coefficient (0-1) - lower values provide more filtering
        speed_der = alpha * speed_der + (1-alpha) * prev_speed_der
        heading_der = alpha * heading_der + (1-alpha) * prev_heading_der
    
    # Save current values for next iteration
    prev_speed_error = speed_error
    prev_heading_error = heading_error
    prev_speed_der = speed_der
    prev_heading_der = heading_der
    
    # Calculate PID control signals
    speed_control = (Kp_speed * speed_error + 
                     Ki_speed * speed_error_sum + 
                     Kd_speed * speed_der)
    
    heading_control = (Kp_heading * heading_error + 
                       Ki_heading * heading_error_sum + 
                       Kd_heading * heading_der)
    
    # Base voltage from speed PID controller
    base_voltage = speed_control
    
    # Differential voltage for steering from heading PID controller
    diff_voltage = heading_control
    
    # Calculate motor voltages
    V_left = base_voltage - diff_voltage
    V_right = base_voltage + diff_voltage
    
    # Prevent extreme differentials that might cause backward motion
    if abs(diff_voltage) > abs(base_voltage) * 0.8 and base_voltage > 0:
        max_diff = abs(base_voltage) * 0.8
        diff_voltage = np.clip(diff_voltage, -max_diff, max_diff)
        V_left = base_voltage - diff_voltage
        V_right = base_voltage + diff_voltage
    
    # Ensure minimum forward voltage to overcome inertia at start
    if i < 100 and v < 0.05:
        min_voltage = 0.5
        V_left = max(V_left, min_voltage)
        V_right = max(V_right, min_voltage)
    
    # Clip voltages
    V_left = np.clip(V_left, -5, 5)
    V_right = np.clip(V_right, -5, 5)
    
    # Ensure the robot never moves backward in the beginning
    if i < 50:
        V_left = max(0.2, V_left)
        V_right = max(0.2, V_right)
    
    return np.array([V_left, V_right]), avoiding_wall

def main():
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Initialize state and estimate [x, y, theta, v, omega]
    true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    state_est = true_state.copy()
    P = np.eye(5) * 0.01  # Initial state covariance
    
    # Storage for plotting
    true_states = []
    estimated_states = []
    control_inputs = []
    avoiding_wall_history = []
    
    # Initialize global variables for PID controller
    global i, speed_error_sum, heading_error_sum, prev_speed_error, prev_heading_error, prev_speed_der, prev_heading_der
    speed_error_sum = 0.0
    heading_error_sum = 0.0
    prev_speed_error = 0.0
    prev_heading_error = 0.0
    prev_speed_der = 0.0
    prev_heading_der = 0.0
    
    # Main simulation loop
    for i in range(steps):
        # Controller
        inputs, avoiding_wall = controller(state_est, target)
        control_inputs.append(inputs.copy())
        avoiding_wall_history.append(avoiding_wall)
        
        # Apply inputs to true system
        true_state = discrete_dynamics(true_state, inputs)
        
        # Generate noisy measurements
        true_measurements = sensor_model(true_state)
        noisy_measurements = add_sensor_noise(true_measurements)
        
        # Update state estimate with EKF
        state_est, P = extended_kalman_filter(state_est, P, noisy_measurements, inputs)
        
        # Save states for plotting
        true_states.append(true_state.copy())
        estimated_states.append(state_est.copy())
        
        # Check if reached target
        distance_to_target = np.linalg.norm(true_state[0:2] - target)
        if distance_to_target < 0.05:
            print(f"Target reached at step {i}, time: {i*dt:.2f}s")
            break
            
        # Check for wall collisions
        for wall in walls:
            dist, _ = distance_to_line_segment(true_state[:2], wall)
            if dist < WALL_THRESHOLD:
                print(f"Collision with wall at step {i}, time: {i*dt:.2f}s")
                break
    
    # Convert to arrays for plotting
    true_states = np.array(true_states)
    estimated_states = np.array(estimated_states)
    control_inputs = np.array(control_inputs)
    avoiding_wall_history = np.array(avoiding_wall_history)
    
    # Plot results
    plot_results(true_states, estimated_states, control_inputs, avoiding_wall_history)

def plot_results(true_states, estimated_states, control_inputs, avoiding_wall_history):
    """Plot simulation results"""
    t = np.arange(len(true_states)) * dt
    
    # Figure 1: Trajectory in 2D space
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label='Estimated')
    plt.plot(target[0], target[1], 'go', markersize=8, label='Target')
    
    # Draw walls
    for wall in walls:
        x1, y1, x2, y2 = wall
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Mark avoiding wall sections
    for i in range(len(avoiding_wall_history)):
        if avoiding_wall_history[i]:
            plt.plot(true_states[i, 0], true_states[i, 1], 'yo', markersize=3)
    
    # Add wall detection zones (faded)
    for wall in walls:
        x1, y1, x2, y2 = wall
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        
        # Add detection zone around walls
        line_len = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_len > 0:  # Prevent division by zero
            nx, ny = (y1-y2)/line_len, (x2-x1)/line_len  # Normal vector
            
            # Create polygon points for detection zone
            polygon_x = [x1+nx*WALL_AVOID_DISTANCE, x2+nx*WALL_AVOID_DISTANCE, 
                        x2-nx*WALL_AVOID_DISTANCE, x1-nx*WALL_AVOID_DISTANCE]
            polygon_y = [y1+ny*WALL_AVOID_DISTANCE, y2+ny*WALL_AVOID_DISTANCE,
                        y2-ny*WALL_AVOID_DISTANCE, y1-ny*WALL_AVOID_DISTANCE]
            
            plt.fill(polygon_x, polygon_y, 'k', alpha=0.1)
    
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Micromouse Trajectory')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    # Draw the robot at several key points along the path
    num_points = min(10, len(true_states))
    indices = np.linspace(0, len(true_states) - 1, num_points, dtype=int)
    
    for idx in indices:
        x, y, theta = true_states[idx, 0:3]
        # Draw a small arrow to represent the robot
        arrow_length = 0.03
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    
    # Figure 2: States over time
    plt.subplot(2, 2, 2)
    plt.plot(t, true_states[:, 0], 'b-', label='True X')
    plt.plot(t, estimated_states[:, 0], 'r--', label='Est X')
    plt.plot(t, true_states[:, 1], 'g-', label='True Y')
    plt.plot(t, estimated_states[:, 1], 'm--', label='Est Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Position vs. Time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(t, true_states[:, 2], 'b-', label='True θ')
    plt.plot(t, estimated_states[:, 2], 'r--', label='Est θ')
    plt.xlabel('Time [s]')
    plt.ylabel('Orientation [rad]')
    plt.title('Orientation vs. Time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(t, true_states[:, 3], 'b-', label='True v')
    plt.plot(t, estimated_states[:, 3], 'r--', label='Est v')
    plt.plot(t, true_states[:, 4], 'g-', label='True ω')
    plt.plot(t, estimated_states[:, 4], 'm--', label='Est ω')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')
    plt.title('Velocities vs. Time')
    plt.grid(True)
    plt.legend()
    
    # Figure 3: Control inputs
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, control_inputs[:, 0], 'b-', label='Left Motor')
    plt.plot(t, control_inputs[:, 1], 'r-', label='Right Motor')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.title('Control Inputs')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    error_x = true_states[:, 0] - estimated_states[:, 0]
    error_y = true_states[:, 1] - estimated_states[:, 1]
    error_theta = true_states[:, 2] - estimated_states[:, 2]
    plt.plot(t, error_x, 'b-', label='X Error')
    plt.plot(t, error_y, 'r-', label='Y Error')
    plt.plot(t, error_theta, 'g-', label='θ Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.title('Estimation Error')
    plt.grid(True)
    plt.legend()
    
    # Calculate and print some statistics
    final_distance = np.linalg.norm(true_states[-1, 0:2] - target)
    print(f"Final distance to target: {final_distance:.4f} m")
    print(f"Total time: {t[-1]:.2f} s")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()