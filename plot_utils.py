import numpy as np
import matplotlib.pyplot as plt


# Plot the loss function for distance/heading prediction model
def plot_forward_loss(distance_model, heading_model, show=True, save_path=None):
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


# Plot the loss function for learning objective 2
def plot_single_loss(model, title="Model Loss vs Iteration", show=True, save_path=None):
    loss = np.asarray(getattr(model, "loss_history", []), dtype=float)
    if loss.size == 0:
        raise ValueError("No loss history found. Train model first using gradient_descent().")

    plt.figure(figsize=(10, 5))
    plt.plot(loss, label="Loss", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Cost")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_forward_vs_test(y_distance_true, y_distance_pred, y_heading_true, y_heading_pred, show=True, save_path=None):
    idx = np.arange(len(y_distance_true))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(idx, y_distance_true, "o-", label="Ground Truth")
    axes[0].plot(idx, y_distance_pred, "s--", label="Forward Prediction")
    axes[0].set_title("Forward vs Test (Distance)")
    axes[0].set_xlabel("Test Sample Index")
    axes[0].set_ylabel("Distance")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(idx, y_heading_true, "o-", label="Ground Truth")
    axes[1].plot(idx, y_heading_pred, "s--", label="Forward Prediction")
    axes[1].set_title("Forward vs Test (Heading)")
    axes[1].set_xlabel("Test Sample Index")
    axes[1].set_ylabel("Heading")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_reverse_vs_test_params(M_test_true, M_test_pred, feature_names=None, show=True, save_path=None):
    if feature_names is None:
        feature_names = ["rot", "lif", "dur", "kp", "kd"]
    idx = np.arange(M_test_true.shape[0])
    n_params = M_test_true.shape[1]

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.4 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        axes[i].plot(idx, M_test_true[:, i], "o-", label="Ground Truth")
        axes[i].plot(idx, M_test_pred[:, i], "s--", label="Reverse Prediction")
        axes[i].set_ylabel(feature_names[i])
        axes[i].grid(True, linestyle="--", alpha=0.4)
        axes[i].legend(loc="best")

    axes[-1].set_xlabel("Test Sample Index")
    fig.suptitle("Reverse Model vs Test (Parameters)", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_original_vs_reverse_outputs(
    y_distance_true,
    y_heading_true,
    forward_distance_from_test_params,
    forward_heading_from_test_params,
    forward_distance_from_reverse_params,
    forward_heading_from_reverse_params,
    show=True,
    save_path=None,
):
    idx = np.arange(len(y_distance_true))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(idx, y_distance_true, "o-", label="Ground Truth")
    axes[0].plot(idx, forward_distance_from_test_params, "^-", label="Original Forward")
    axes[0].plot(idx, forward_distance_from_reverse_params, "s--", label="Reverse -> Forward")
    axes[0].set_title("Original vs Reverse (Distance)")
    axes[0].set_xlabel("Test Sample Index")
    axes[0].set_ylabel("Distance")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(idx, y_heading_true, "o-", label="Ground Truth")
    axes[1].plot(idx, forward_heading_from_test_params, "^-", label="Original Forward")
    axes[1].plot(idx, forward_heading_from_reverse_params, "s--", label="Reverse -> Forward")
    axes[1].set_title("Original vs Reverse (Heading)")
    axes[1].set_xlabel("Test Sample Index")
    axes[1].set_ylabel("Heading")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
