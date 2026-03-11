import json
from pathlib import Path
import numpy as np
import polyregression as pr
from plot_utils import (
    plot_forward_loss,
    plot_single_loss,
    plot_forward_vs_test,
    plot_reverse_vs_test_params,
    plot_original_vs_reverse_outputs,
    plot_reward_function,
)
from report_utils import create_report_dir, export_lab_report

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
FEATURE_NAMES = ["rot", "lif", "dur", "kp", "kd"]


# Generate experiment params and split.
def gen_params(n_trials, train_ratio=0.7, validate_ratio=0.2, test_ratio=0.1):
    if not np.isclose(train_ratio + validate_ratio + test_ratio, 1.0):
        raise ValueError("ratio sum must equal 1.0")

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

#For learning objective 2, the input and output of objective 1 is reversed
def build_inverse_inputs(y_distance, y_heading):
    y_distance = np.asarray(y_distance, dtype=float)
    y_heading = np.asarray(y_heading, dtype=float)
    if y_distance.shape[0] != y_heading.shape[0]:
        raise ValueError(f"Distance and heading arrays must have same length ({y_distance.shape[0]} vs {y_heading.shape[0]}).")
    # Requested tuple order: (distance, heading)
    return np.column_stack([y_distance, y_heading])


#make sure the output of model from objective 2 is also within the range predefined
def clip_params_to_bounds(params, bounds):
    params = np.asarray(params, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    if params.ndim == 1:
        return np.clip(params, lower, upper)
    if params.ndim == 2:
        return np.clip(params, lower.reshape(1, -1), upper.reshape(1, -1))
    raise ValueError("params must be 1D or 2D array.")

# output the regression metrics of the model comparing to groud truth
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}

# output he evaluation metrics for learning objective 2
def multioutput_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    error = y_pred - y_true
    mse_per_output = np.mean(error ** 2, axis=0)
    mae_per_output = np.mean(np.abs(error), axis=0)
    ss_res = np.sum(error ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2_per_output = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), 0.0)

    return {
        "mse": float(np.mean(mse_per_output)),
        "mae": float(np.mean(mae_per_output)),
        "r2": float(np.mean(r2_per_output)),
        "mse_per_output": mse_per_output,
        "mae_per_output": mae_per_output,
        "r2_per_output": r2_per_output,
    }


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


if __name__ == "__main__":
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
    report_dir = create_report_dir(Path(__file__).resolve().parent)
    print(f"Report directory: {report_dir}")

    degree_candidates = [1, 2, 3, 4, 5]

    # ========= Distance/Heading Model =========
    best_deg_dist, best_mod_dist, dist_results = pr.choose_best_degree(
        M_train,
        y_distance_train,
        M_validate,
        y_distance_validate,
        degree_candidates,
        alpha=0.001,
        iterations=5000,
    )

    best_deg_head, best_mod_head, head_results = pr.choose_best_degree(
        M_train,
        y_heading_train,
        M_validate,
        y_heading_validate,
        degree_candidates,
        alpha=0.001,
        iterations=5000,
    )

    # final training dataset includes both training and validation data to maximize data for final model training before testing
    M_final = np.vstack([M_train, M_validate])
    y_dist_final = np.concatenate([y_distance_train, y_distance_validate])
    y_head_final = np.concatenate([y_heading_train, y_heading_validate])

    # Create the model for both distance and heading
    model_distance = pr.Polynomial_Regression(degree=best_deg_dist, alpha=0.001, iterations=5000)
    model_heading = pr.Polynomial_Regression(degree=best_deg_head, alpha=0.001, iterations=5000)

    # Training two forward models
    print("Training model for distance...")
    model_distance.gradient_descent(M_final, y_dist_final)
    print("Training model for heading...")
    model_heading.gradient_descent(M_final, y_head_final)

    # Find the best parameters that has the highest reward
    bounds = np.array([ROT_BOUNDS, LIF_BOUNDS, DUR_BOUNDS, KP_BOUNDS, KD_BOUNDS])
    best_params, rewards = find_best_params(
        model_distance,
        model_heading,
        bounds,
        n_candidates=10000,
        return_history=True,
    )

    plot_forward_loss(model_distance, model_heading, save_path=str(report_dir / "forward_loss"))
    plot_reward_function(rewards, save_path=str(report_dir / "reward_progression.png"))


    # ========= Reverse Model =========
    # input = (distance, heading), output = gait parameters [rot, lif, dur, kp, kd].
    # Keep test split strictly as held-out baseline.
    X_reverse_train = build_inverse_inputs(y_distance_train, y_heading_train)
    X_reverse_validate = build_inverse_inputs(y_distance_validate, y_heading_validate)
    X_reverse_test = build_inverse_inputs(y_distance_test, y_heading_test)
    Y_reverse_train = M_train
    Y_reverse_validate = M_validate
    Y_reverse_test = M_test

    best_deg_reverse, best_mod_reverse, reverse_results = pr.choose_best_degree(
        X_reverse_train,
        Y_reverse_train,
        X_reverse_validate,
        Y_reverse_validate,
        degree_candidates,
        alpha=0.001,
        iterations=5000,
    )

    # Final reverse training uses train + validation; test remains untouched baseline.
    X_reverse_final = np.vstack([X_reverse_train, X_reverse_validate])
    Y_reverse_final = np.vstack([Y_reverse_train, Y_reverse_validate])

    model_reverse = pr.Polynomial_Regression(degree=best_deg_reverse, alpha=0.001, iterations=5000)
    print("Training reverse model (distance, heading -> gait parameters)...")
    model_reverse.gradient_descent(X_reverse_final, Y_reverse_final)

    reverse_validation_metrics = best_mod_reverse.evaluate(X_reverse_validate, Y_reverse_validate)
    reverse_test_metrics = model_reverse.evaluate(X_reverse_test, Y_reverse_test)

    # ========= Test-baseline comparisons =========
    # 1) Distance/Heading model vs test dataset 
    forward_pred_distance_test = model_distance.predict(M_test).reshape(-1)
    forward_pred_heading_test = model_heading.predict(M_test).reshape(-1)
    forward_distance_metrics = regression_metrics(y_distance_test, forward_pred_distance_test)
    forward_heading_metrics = regression_metrics(y_heading_test, forward_pred_heading_test)

    # 2) Reverse model vs test dataset
    reverse_pred_params_test = model_reverse.predict(X_reverse_test)
    reverse_pred_params_test_clipped = clip_params_to_bounds(reverse_pred_params_test, bounds)
    reverse_param_metrics = multioutput_metrics(M_test, reverse_pred_params_test_clipped)

    # 3) Distance/Heading model vs reverse model (both in output space, aligned on test set)
    reverse_forward_distance_test = model_distance.predict(reverse_pred_params_test_clipped).reshape(-1)
    reverse_forward_heading_test = model_heading.predict(reverse_pred_params_test_clipped).reshape(-1)
    original_vs_reverse_distance_metrics = regression_metrics(forward_pred_distance_test, reverse_forward_distance_test)
    original_vs_reverse_heading_metrics = regression_metrics(forward_pred_heading_test, reverse_forward_heading_test)

    sample_goal = X_reverse_test[0]  # (distance, heading)
    sample_reverse_params = reverse_pred_params_test_clipped[0]
    sample_forward_from_reverse_distance = float(reverse_forward_distance_test[0])
    sample_forward_from_reverse_heading = float(reverse_forward_heading_test[0])

    plot_single_loss(model_reverse, title="Reverse Model Loss vs Iteration", save_path=str(report_dir / "reverse_loss.png"))
    plot_forward_vs_test(
        y_distance_test,
        forward_pred_distance_test,
        y_heading_test,
        forward_pred_heading_test,
        save_path=str(report_dir / "comparison_forward_vs_test.png"),
    )
    plot_reverse_vs_test_params(
        M_test,
        reverse_pred_params_test_clipped,
        feature_names=FEATURE_NAMES,
        save_path=str(report_dir / "comparison_reverse_vs_test_params.png"),
    )
    plot_original_vs_reverse_outputs(
        y_distance_test,
        y_heading_test,
        forward_pred_distance_test,
        forward_pred_heading_test,
        reverse_forward_distance_test,
        reverse_forward_heading_test,
        save_path=str(report_dir / "comparison_original_vs_reverse_outputs.png"),
    )

    # Print all results and metrics
    print("Best degree (distance):", best_deg_dist)
    print("Best degree (heading):", best_deg_head)
    print("Best degree (reverse):", best_deg_reverse)


    print(f"[Comparison 1] Distance vs Test (Distance): {forward_distance_metrics}")
    print(f"[Comparison 1] Heading vs Test (Heading): {forward_heading_metrics}")
    print(f"[Comparison 2] Reverse vs Test (Parameters): {reverse_param_metrics}")
    print(f"[Comparison 3] Distance/heading vs Reverse (Distance outputs): {original_vs_reverse_distance_metrics}")
    print(f"[Comparison 3] Distance/heading vs Reverse (Heading outputs): {original_vs_reverse_heading_metrics}")

    print(f"Sample test goal (distance, heading): {sample_goal}")
    print(f"Sample reverse predicted params (clipped): {sample_reverse_params}")
    print(
        "Sample forward output from reverse params -> "
        f"(distance, heading): [{sample_forward_from_reverse_distance:.2f}, {sample_forward_from_reverse_heading:.2f}]"
    )

    print(f"Best Parameters (forward search): {best_params}")

    export_lab_report(
        report_dir,
        degrees={
            "forward_distance_best_degree": best_deg_dist,
            "forward_heading_best_degree": best_deg_head,
            "reverse_best_degree": best_deg_reverse,
            "degree_candidates": degree_candidates,
            "forward_distance_degree_results": dist_results,
            "forward_heading_degree_results": head_results,
            "reverse_degree_results": reverse_results,
        },
        metrics={
            "forward_vs_test_distance": forward_distance_metrics,
            "forward_vs_test_heading": forward_heading_metrics,
            "reverse_validation": reverse_validation_metrics,
            "reverse_test": reverse_test_metrics,
            "reverse_vs_test_parameters": reverse_param_metrics,
            "original_vs_reverse_distance": original_vs_reverse_distance_metrics,
            "original_vs_reverse_heading": original_vs_reverse_heading_metrics,
        },
        arrays={
            "y_distance_test": y_distance_test,
            "y_heading_test": y_heading_test,
            "forward_pred_distance_test": forward_pred_distance_test,
            "forward_pred_heading_test": forward_pred_heading_test,
            "M_test": M_test,
            "reverse_pred_params_test_clipped": reverse_pred_params_test_clipped,
            "reverse_forward_distance_test": reverse_forward_distance_test,
            "reverse_forward_heading_test": reverse_forward_heading_test,
        },
        sample_case={
            "sample_goal_distance_heading": sample_goal,
            "sample_reverse_params_clipped": sample_reverse_params,
            "sample_forward_output_from_reverse_params": [
                sample_forward_from_reverse_distance,
                sample_forward_from_reverse_heading,
            ],
        },
        feature_names=FEATURE_NAMES,
    )
    print(f"Exported lab report assets to: {report_dir}")
