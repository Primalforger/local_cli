"""Adaptive ML engine — task classification and model performance tracking.

Uses scikit-learn's MultinomialNB + TfidfVectorizer for task-type
classification. Falls back gracefully to keyword matching when
sklearn is not installed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# ── Graceful sklearn import ───────────────────────────────────

_SKLEARN_AVAILABLE = False
try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


# ── Task Classifier ───────────────────────────────────────────

class TaskClassifier:
    """Scikit-learn wrapper with JSON persistence for task-type classification.

    Uses TfidfVectorizer + MultinomialNB when sklearn is available,
    falls back to keyword matching otherwise.
    """

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._vectorizer: Any = None
        self._classifier: Any = None
        self._is_trained = False
        self._training_texts: list[str] = []
        self._training_labels: list[str] = []
        self._pending_texts: list[str] = []
        self._pending_labels: list[str] = []
        self._retrain_threshold = 10  # Retrain after N new samples

        if _SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
            )
            self._classifier = MultinomialNB(alpha=alpha)

    def train(self, texts: list[str], labels: list[str]) -> bool:
        """Batch fit on accumulated data.

        Args:
            texts: List of prompt texts
            labels: List of task type labels

        Returns:
            True if training succeeded.
        """
        if not _SKLEARN_AVAILABLE:
            return False

        if len(texts) < 2 or len(set(labels)) < 2:
            return False

        try:
            self._training_texts = list(texts)
            self._training_labels = list(labels)
            self._pending_texts.clear()
            self._pending_labels.clear()

            X = self._vectorizer.fit_transform(texts)
            self._classifier.fit(X, labels)
            self._is_trained = True
            return True
        except Exception as e:
            console.print(f"[dim]Classifier training failed: {e}[/dim]")
            return False

    def partial_train(self, text: str, label: str) -> bool:
        """Accumulate a sample, retraining every N new samples.

        Args:
            text: Prompt text
            label: Task type label

        Returns:
            True if retraining was triggered.
        """
        self._pending_texts.append(text)
        self._pending_labels.append(label)

        if len(self._pending_texts) >= self._retrain_threshold:
            all_texts = self._training_texts + self._pending_texts
            all_labels = self._training_labels + self._pending_labels
            return self.train(all_texts, all_labels)
        return False

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict task type with confidence scores.

        Args:
            text: Prompt text

        Returns:
            Tuple of (predicted_label, {label: probability})
        """
        if not _SKLEARN_AVAILABLE or not self._is_trained:
            return self._keyword_fallback(text)

        try:
            X = self._vectorizer.transform([text])
            label = self._classifier.predict(X)[0]
            probas = self._classifier.predict_proba(X)[0]
            classes = self._classifier.classes_
            confidence = {
                cls: round(float(prob), 4)
                for cls, prob in zip(classes, probas)
            }
            return str(label), confidence
        except Exception:
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> tuple[str, dict[str, float]]:
        """Keyword-based classification fallback."""
        try:
            from llm.model_router import detect_task_type
            task_type = detect_task_type(text)
            return task_type, {task_type: 1.0}
        except ImportError:
            return "general", {"general": 1.0}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def sample_count(self) -> int:
        return len(self._training_texts) + len(self._pending_texts)


# ── Adaptive Engine ───────────────────────────────────────────

class AdaptiveEngine:
    """Wraps classifier + model performance tracking.

    Provides adaptive task type detection and model selection
    based on historical performance data.
    """

    def __init__(
        self,
        model_file: Path | None = None,
        min_samples: int = 20,
        alpha: float = 1.0,
    ):
        if model_file is None:
            try:
                from core.config import ADAPTIVE_MODEL_FILE
                self._model_file = ADAPTIVE_MODEL_FILE
            except ImportError:
                self._model_file = Path.home() / ".config" / "localcli" / "adaptive_model.json"
        else:
            self._model_file = model_file

        self._min_samples = min_samples
        self._classifier = TaskClassifier(alpha=alpha)
        self._model_performance: dict[str, dict[str, dict[str, int]]] = {}
        # Structure: {task_type: {model: {"success": N, "total": N}}}
        self._total_samples = 0
        self._last_trained = ""

        self._load()

    def detect_task_type(self, prompt: str) -> tuple[str, float]:
        """Detect task type, using ML when enough data exists.

        Args:
            prompt: User prompt

        Returns:
            Tuple of (task_type, confidence)
        """
        if self._classifier.is_trained and self._total_samples >= self._min_samples:
            label, confidence_dict = self._classifier.predict(prompt)
            confidence = confidence_dict.get(label, 0.0)
            return label, confidence
        else:
            # Fall back to keyword matching
            try:
                from llm.model_router import detect_task_type
                task_type = detect_task_type(prompt)
                return task_type, 1.0
            except ImportError:
                return "general", 1.0

    def learn(
        self,
        prompt: str,
        task_type: str,
        model: str,
        success: bool,
    ) -> None:
        """Record a learning sample.

        Args:
            prompt: The user prompt
            task_type: Detected or confirmed task type
            model: Model that was used
            success: Whether the outcome was successful
        """
        # Update model performance tracking
        if task_type not in self._model_performance:
            self._model_performance[task_type] = {}
        if model not in self._model_performance[task_type]:
            self._model_performance[task_type][model] = {"success": 0, "total": 0}

        self._model_performance[task_type][model]["total"] += 1
        if success:
            self._model_performance[task_type][model]["success"] += 1

        # Feed classifier
        self._total_samples += 1
        self._classifier.partial_train(prompt, task_type)

        # Auto-save periodically
        if self._total_samples % 10 == 0:
            self._save()

    def get_best_model_for_task(
        self,
        task_type: str,
        available_models: list[str],
        fallback: str = "",
    ) -> str | None:
        """Return model with best success rate for this task type.

        Args:
            task_type: Task type to look up
            available_models: List of available model names
            fallback: Default model if no data

        Returns:
            Best model name, or None if insufficient data.
        """
        task_data = self._model_performance.get(task_type, {})
        if not task_data:
            return fallback or None

        best_model = None
        best_rate = -1.0
        min_trials = 3  # Need at least N trials for a recommendation

        for model, stats in task_data.items():
            if model not in available_models:
                continue
            if stats["total"] < min_trials:
                continue
            rate = stats["success"] / stats["total"]
            if rate > best_rate:
                best_rate = rate
                best_model = model

        return best_model or fallback or None

    def get_stats(self) -> dict:
        """Get engine statistics for display.

        Returns:
            Dict with total_samples, model_performance, is_trained, etc.
        """
        return {
            "total_samples": self._total_samples,
            "is_trained": self._classifier.is_trained,
            "sklearn_available": _SKLEARN_AVAILABLE,
            "min_samples": self._min_samples,
            "last_trained": self._last_trained,
            "model_performance": self._model_performance,
            "classifier_samples": self._classifier.sample_count,
        }

    def reset(self) -> None:
        """Clear all learned data."""
        self._classifier = TaskClassifier(alpha=self._classifier._alpha)
        self._model_performance.clear()
        self._total_samples = 0
        self._last_trained = ""

        if self._model_file.exists():
            try:
                self._model_file.unlink()
            except OSError:
                pass

        # Also remove the pkl file
        pkl_file = self._model_file.with_suffix(".pkl")
        if pkl_file.exists():
            try:
                pkl_file.unlink()
            except OSError:
                pass

    def force_retrain(self) -> bool:
        """Force retrain from all accumulated training data.

        Returns:
            True if retrain succeeded.
        """
        all_texts = (
            self._classifier._training_texts + self._classifier._pending_texts
        )
        all_labels = (
            self._classifier._training_labels + self._classifier._pending_labels
        )

        if len(all_texts) < 2:
            return False

        success = self._classifier.train(all_texts, all_labels)
        if success:
            self._last_trained = datetime.now().isoformat()
            self._save()
        return success

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        """Save metadata and model to disk."""
        try:
            self._model_file.parent.mkdir(parents=True, exist_ok=True)

            metadata = {
                "version": 2,
                "total_samples": self._total_samples,
                "training_texts": self._classifier._training_texts,
                "training_labels": self._classifier._training_labels,
                "model_performance": self._model_performance,
                "last_trained": self._last_trained or datetime.now().isoformat(),
            }
            self._model_file.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Save sklearn model if available
            if _SKLEARN_AVAILABLE and self._classifier.is_trained:
                pkl_path = self._model_file.with_suffix(".pkl")
                joblib.dump(
                    {
                        "vectorizer": self._classifier._vectorizer,
                        "classifier": self._classifier._classifier,
                    },
                    pkl_path,
                )
        except Exception as e:
            console.print(f"[dim]Warning: Could not save adaptive model: {e}[/dim]")

    def _load(self):
        """Load metadata and model from disk."""
        if not self._model_file.exists():
            return

        try:
            metadata = json.loads(
                self._model_file.read_text(encoding="utf-8")
            )

            self._total_samples = metadata.get("total_samples", 0)
            self._model_performance = metadata.get("model_performance", {})
            self._last_trained = metadata.get("last_trained", "")

            training_texts = metadata.get("training_texts", [])
            training_labels = metadata.get("training_labels", [])

            # Restore classifier state
            if training_texts and training_labels:
                self._classifier._training_texts = training_texts
                self._classifier._training_labels = training_labels

            # Load sklearn model if available
            pkl_path = self._model_file.with_suffix(".pkl")
            if _SKLEARN_AVAILABLE and pkl_path.exists():
                try:
                    saved = joblib.load(pkl_path)
                    self._classifier._vectorizer = saved["vectorizer"]
                    self._classifier._classifier = saved["classifier"]
                    self._classifier._is_trained = True
                except Exception:
                    # Fall back to retraining from texts
                    if training_texts and len(set(training_labels)) >= 2:
                        self._classifier.train(training_texts, training_labels)

        except (json.JSONDecodeError, OSError, KeyError) as e:
            console.print(f"[dim]Warning: Could not load adaptive model: {e}[/dim]")
