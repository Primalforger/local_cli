"""Adaptive ML package — task classification, outcome tracking, prompt optimization."""


def __getattr__(name):
    if name == "AdaptiveEngine":
        from adaptive.adaptive_engine import AdaptiveEngine
        return AdaptiveEngine
    if name == "OutcomeTracker":
        from adaptive.outcome_tracker import OutcomeTracker
        return OutcomeTracker
    if name == "PromptOptimizer":
        from adaptive.prompt_optimizer import PromptOptimizer
        return PromptOptimizer
    if name == "ResponseValidator":
        from adaptive.response_validator import ResponseValidator
        return ResponseValidator
    raise AttributeError(f"module 'adaptive' has no attribute {name!r}")
