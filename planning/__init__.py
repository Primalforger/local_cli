"""Planning package — planner, builder, project context, reviewer, templates."""


def __getattr__(name):
    if name in ("generate_plan", "display_plan", "save_plan"):
        from planning.planner import generate_plan, display_plan, save_plan
        return locals()[name]
    if name == "build_plan":
        from planning.builder import build_plan
        return build_plan
    if name in ("scan_project", "scan_project_cached"):
        from planning import project_context
        return getattr(project_context, name)
    raise AttributeError(f"module 'planning' has no attribute {name!r}")
