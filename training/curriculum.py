"""Curriculum learning manager for MARWS training."""
import json
import os
from collections import deque


class CurriculumManager:
    """Manages curriculum stage progression during training.

    Tracks success rate and automatically advances stages when criteria are met.
    """

    STAGE_NAMES = {
        1: "reach",
        2: "grasp",
        3: "lift",
        4: "full",
    }

    def __init__(
        self,
        advancement_threshold=0.80,
        evaluation_window=50,
        min_iterations_per_stage=50,
        stagnation_window=100,
        checkpoint_dir="models/curriculum",
    ):
        """Initialize curriculum manager.

        Args:
            advancement_threshold: Success rate required to advance (0-1)
            evaluation_window: Number of episodes to average for success rate
            min_iterations_per_stage: Minimum iterations before advancement allowed
            stagnation_window: Iterations without improvement before warning
            checkpoint_dir: Directory to save stage checkpoints
        """
        self.advancement_threshold = advancement_threshold
        self.evaluation_window = evaluation_window
        self.min_iterations_per_stage = min_iterations_per_stage
        self.stagnation_window = stagnation_window
        self.checkpoint_dir = checkpoint_dir

        # Current state
        self.current_stage = 1
        self.stage_iterations = 0
        self.total_iterations = 0

        # Success tracking
        self.episode_successes = deque(maxlen=evaluation_window)
        self.best_success_rate = 0.0
        self.iterations_since_improvement = 0

        # History for logging
        self.stage_history = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_current_stage(self):
        """Get current curriculum stage (1-4)."""
        return self.current_stage

    def get_stage_name(self):
        """Get human-readable name of current stage."""
        return self.STAGE_NAMES.get(self.current_stage, "unknown")

    def record_episode(self, success):
        """Record episode result for success rate tracking.

        Args:
            success: Whether the episode was successful
        """
        self.episode_successes.append(1.0 if success else 0.0)

    def get_success_rate(self):
        """Get current success rate over evaluation window."""
        if len(self.episode_successes) == 0:
            return 0.0
        return sum(self.episode_successes) / len(self.episode_successes)

    def update(self, algo, iteration_result):
        """Update curriculum state after a training iteration.

        Args:
            algo: The RLlib algorithm (for saving checkpoints)
            iteration_result: Result dict from algo.train()

        Returns:
            dict with curriculum info for logging
        """
        self.stage_iterations += 1
        self.total_iterations += 1

        # Extract episode successes from result
        env_runners = iteration_result.get("env_runners", {})
        hist = env_runners.get("hist_stats", {})

        # Get success info from episodes
        # RLlib stores custom info in episode data
        episode_infos = iteration_result.get("episodes_this_iter", 0)

        # Track success rate
        current_success_rate = self.get_success_rate()

        # Check for stagnation
        if current_success_rate > self.best_success_rate + 0.01:
            self.best_success_rate = current_success_rate
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

        stagnation_warning = False
        if self.iterations_since_improvement >= self.stagnation_window:
            stagnation_warning = True

        # Check for advancement
        advanced = False
        if self._should_advance():
            self._advance_stage(algo)
            advanced = True

        return {
            "curriculum_stage": self.current_stage,
            "stage_name": self.get_stage_name(),
            "stage_iterations": self.stage_iterations,
            "success_rate": current_success_rate,
            "best_success_rate": self.best_success_rate,
            "advanced": advanced,
            "stagnation_warning": stagnation_warning,
            "iterations_since_improvement": self.iterations_since_improvement,
        }

    def _should_advance(self):
        """Check if criteria for stage advancement are met."""
        # Can't advance beyond stage 4
        if self.current_stage >= 4:
            return False

        # Need minimum iterations
        if self.stage_iterations < self.min_iterations_per_stage:
            return False

        # Need enough episodes for reliable success rate
        if len(self.episode_successes) < self.evaluation_window:
            return False

        # Need to meet success threshold
        return self.get_success_rate() >= self.advancement_threshold

    def _advance_stage(self, algo):
        """Advance to next curriculum stage."""
        # Save checkpoint for current stage
        stage_checkpoint_dir = os.path.join(
            self.checkpoint_dir,
            f"stage{self.current_stage}_{self.get_stage_name()}"
        )
        os.makedirs(stage_checkpoint_dir, exist_ok=True)
        algo.save(stage_checkpoint_dir)

        # Record history
        self.stage_history.append({
            "stage": self.current_stage,
            "name": self.get_stage_name(),
            "iterations": self.stage_iterations,
            "final_success_rate": self.get_success_rate(),
            "total_iterations": self.total_iterations,
        })

        # Advance
        self.current_stage += 1
        self.stage_iterations = 0
        self.episode_successes.clear()
        self.best_success_rate = 0.0
        self.iterations_since_improvement = 0

        print(f"\n{'='*60}")
        print(f"CURRICULUM ADVANCEMENT: Stage {self.current_stage - 1} -> Stage {self.current_stage}")
        print(f"Now training: {self.get_stage_name()}")
        print(f"Checkpoint saved to: {stage_checkpoint_dir}")
        print(f"{'='*60}\n")

    def save_state(self, filepath=None):
        """Save curriculum state to file."""
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "curriculum_state.json")

        state = {
            "current_stage": self.current_stage,
            "stage_iterations": self.stage_iterations,
            "total_iterations": self.total_iterations,
            "best_success_rate": self.best_success_rate,
            "iterations_since_improvement": self.iterations_since_improvement,
            "stage_history": self.stage_history,
            "episode_successes": list(self.episode_successes),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath=None):
        """Load curriculum state from file."""
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "curriculum_state.json")

        if not os.path.exists(filepath):
            return False

        with open(filepath, "r") as f:
            state = json.load(f)

        self.current_stage = state["current_stage"]
        self.stage_iterations = state["stage_iterations"]
        self.total_iterations = state["total_iterations"]
        self.best_success_rate = state["best_success_rate"]
        self.iterations_since_improvement = state["iterations_since_improvement"]
        self.stage_history = state["stage_history"]
        self.episode_successes = deque(state["episode_successes"], maxlen=self.evaluation_window)

        return True

    def get_summary(self):
        """Get summary of curriculum progress."""
        lines = [
            "Curriculum Progress:",
            f"  Current Stage: {self.current_stage} ({self.get_stage_name()})",
            f"  Stage Iterations: {self.stage_iterations}",
            f"  Total Iterations: {self.total_iterations}",
            f"  Success Rate: {self.get_success_rate():.1%}",
            f"  Best Success Rate: {self.best_success_rate:.1%}",
        ]

        if self.stage_history:
            lines.append("  Completed Stages:")
            for entry in self.stage_history:
                lines.append(
                    f"    Stage {entry['stage']} ({entry['name']}): "
                    f"{entry['iterations']} iters, {entry['final_success_rate']:.1%} success"
                )

        return "\n".join(lines)
