import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import polyregression as pr

# Display numpy floats in regular decimal form
np.set_printoptions(
    suppress=True,
    formatter={"float_kind": lambda x: f"{x:.2f}"},
    linewidth=180
)

# Discrete candidate values for parameter generation.
ROT_VALUES = np.arange(20, 150, 1)              
LIF_VALUES = np.arange(20, 150, 1)              
DUR_VALUES = np.round(np.arange(0.1, 1.5, 0.1), 1) 
KP_VALUES = np.round(np.arange(0.00, 2, 0.01), 2)  
KD_VALUES = np.round(np.arange(0.00, 2, 0.01), 2)   

# Keep bounds for optimization/search helpers.
ROT_BOUNDS = (float(np.min(ROT_VALUES)), float(np.max(ROT_VALUES)))
LIF_BOUNDS = (float(np.min(LIF_VALUES)), float(np.max(LIF_VALUES)))
DUR_BOUNDS = (float(np.min(DUR_VALUES)), float(np.max(DUR_VALUES)))
KP_BOUNDS = (float(np.min(KP_VALUES)), float(np.max(KP_VALUES)))
KD_BOUNDS = (float(np.min(KD_VALUES)), float(np.max(KD_VALUES)))

# Generate experiment params and split into train/validate/test sets.
def gen_params(n_trials, train_ratio=0.7, validate_ratio=0.2, test_ratio=0.1):
    if not np.isclose(train_ratio + validate_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + validate_ratio + test_ratio must equal 1.0")

    params_list = []
    for _ in range(n_trials):
        rot = np.random.choice(ROT_VALUES)
        lif = np.random.choice(LIF_VALUES)
        dur = np.random.choice(DUR_VALUES)
        kp = np.random.choice(KP_VALUES)
        kd = np.random.choice(KD_VALUES)
        params_list.append([rot, lif, dur, kp, kd])

    params = np.array(params_list)
    np.random.shuffle(params)

    train_size = int(n_trials * train_ratio)
    validate_size = int(n_trials * validate_ratio)
    # Keep remainder in test split so all trials are used.
    test_size = n_trials - train_size - validate_size

    train_params = params[:train_size]
    validate_params = params[train_size:train_size + validate_size]
    test_params = params[train_size + validate_size:train_size + validate_size + test_size]
    return train_params, validate_params, test_params

# After we have the model trained to predict the distance and heading given input
# we create a reward function and then find the best model parameter that has the highest reward
def reward_function(params, pred_distance, pred_heading):
    # For simplicity, we define the reward as a weighted sum of distance and heading
    distance_weight = 0.8
    heading_weight = 1.5

    pred_d = float(pred_distance.predict(params).reshape(-1)[0])
    pred_h = float(pred_heading.predict(params).reshape(-1)[0])
    reward = (distance_weight * pred_d) - (heading_weight * abs(pred_h))
    return reward

def plot_loss_function(distance_model, heading_model, show=True, save_path=None):
    distance_loss = np.asarray(getattr(distance_model, "loss_history", []), dtype=float)
    heading_loss = np.asarray(getattr(heading_model, "loss_history", []), dtype=float)

    if distance_loss.size == 0 and heading_loss.size == 0:
        raise ValueError("No loss history found. Train model(s) first using gradient_descent().")

    if distance_loss.size > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(distance_loss, label="Distance Loss", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Cost")
        plt.title("Distance Loss vs Iteration")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_distance.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    if heading_loss.size > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(heading_loss, label="Heading Loss", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Cost")
        plt.title("Heading Loss vs Iteration")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_heading.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()

def plot_reward_function(rewards, show=True, save_path=None):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        raise ValueError("Rewards array is empty.")

    best_so_far = np.maximum.accumulate(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Candidate", alpha=0.35)
    plt.plot(best_so_far, label="Best Reward So Far", linewidth=2)
    plt.xlabel("Candidate Index")
    plt.ylabel("Reward")
    plt.title("Reward Progression")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

def sample_reward_history(pred_distance, pred_heading, bounds, n_candidates=10000, verbose=True):
    candidates = []
    rewards = []

    for _ in range(n_candidates):
        candidate = [np.random.uniform(b[0], b[1]) for b in bounds]
        r = reward_function(candidate, pred_distance, pred_heading)
        candidates.append(candidate)
        rewards.append(r)

        if verbose:
            print(f"Candidate: {candidate}, Reward: {r}")

    candidates = np.array(candidates)
    rewards = np.array(rewards)
    best_idx = int(np.argmax(rewards))
    return candidates, rewards, best_idx

# Generate random parameters within the bound, then find the best combindation of parameters that has the highest reward
def find_best_params(pred_distance, pred_heading, bounds, n_candidates=10000, return_history=False):

    candidates, rewards, best_idx = sample_reward_history(
        pred_distance, pred_heading, bounds, n_candidates=n_candidates, verbose=False
    )
    best_params = candidates[best_idx]
    if return_history:
        return best_params, rewards
    return best_params

def load_dataset(json_path="data/gait_dataset.json"):
    dataset_path = Path(json_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent / dataset_path

    with dataset_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    M_train = np.array(data["M_train"], dtype=float)
    M_validate = np.array(data["M_validate"], dtype=float)
    M_test = np.array(data["M_test"], dtype=float)

    y_distance_train = np.array(data["y_distance_train"], dtype=float)
    y_heading_train = np.array(data["y_heading_train"], dtype=float)
    y_distance_validate = np.array(data["y_distance_validate"], dtype=float)
    y_heading_validate = np.array(data["y_heading_validate"], dtype=float)
    y_distance_test = np.array(data["y_distance_test"], dtype=float)
    y_heading_test = np.array(data["y_heading_test"], dtype=float)

    if len(y_distance_train) != len(y_heading_train):
        raise ValueError(f"Distance and heading data collection has different length ({len(y_distance_train)}, {len(y_heading_train)}).")

    return (
        M_train,
        M_validate,
        M_test,
        y_distance_train,
        y_heading_train,
        y_distance_validate,
        y_heading_validate,
        y_distance_test,
        y_heading_test,
    )

if __name__ == '__main__':
    (
        M_train,
        M_validate,
        M_test,
        y_distance_train,
        y_heading_train,
        y_distance_validate,
        y_heading_validate,
        y_distance_test,
        y_heading_test,
    ) = load_dataset()

    # validate successful loading of params
    print("==Training Params==")
    print(M_train)
    print("==Validating Params==")
    print(M_validate)
    print("==Testing Params==")
    print(M_test)

    # Choose the best degree for the polynomial regression model based on the validation set
    degree_candidates = [1, 2, 3, 4, 5]

    best_deg_dist, best_mod_dist, dist_results = pr.choose_best_degree(
        M_train, y_distance_train,
        M_validate, y_distance_validate,
        degree_candidates, alpha=0.01, iterations=1000
    )

    best_deg_head, best_mod_head, head_results = pr.choose_best_degree(
        M_train, y_heading_train,
        M_validate, y_heading_validate,
        degree_candidates, alpha=0.01, iterations=1000
    )

    

    # final training dataset includes both training and validation data to maximize data for final model training before testing
    M_final = np.vstack([M_train, M_validate])
    y_dist_final = np.concatenate([y_distance_train, y_distance_validate])
    y_head_final = np.concatenate([y_heading_train, y_heading_validate])

    # Create the model for both distance and heading
    model_distance = pr.Polynomial_Regression(degree=best_deg_dist, alpha=0.01, iterations=1000)
    model_heading = pr.Polynomial_Regression(degree=best_deg_head, alpha=0.01, iterations=1000)

    #Training two models
    print("Training model for distance...")
    model_distance.gradient_descent(M_final, y_dist_final)
    print("Training model for heading...")
    model_heading.gradient_descent(M_final, y_head_final)

    #Find the best parameters that has the highest reward
    bounds = np.array([ROT_BOUNDS, LIF_BOUNDS, DUR_BOUNDS, KP_BOUNDS, KD_BOUNDS])
    best_params, rewards = find_best_params(
        model_distance, model_heading, bounds, n_candidates=10000, return_history=True
    )
    #loss plots
    plot_loss_function(model_distance, model_heading)
    plot_reward_function(rewards)

    #Testing output
    print(f"Degree (distance) Results: {dist_results}")
    print(f"Degree (heading) Results: {head_results}")


    print("Best degree (distance):", best_deg_dist)
    print("Best degree (heading):", best_deg_head)
    print(f"Best Parameters: {best_params}")



    
