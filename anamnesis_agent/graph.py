from langgraph.graph import StateGraph, END

from .nodes import (
    ask_user_node,
    await_user_node,
    calibrate_node,
    check_consent_node,
    classifier_node,
    detect_language_and_greet_node,
    maybe_translate_node,
    normalize_struct_node,
    parse_and_clean_node,
    planner_node,
    present_results_node,
    red_flags_node,
    safe_goodbye_node,
    set_classifier_target_lang,
    sufficiency_gate_node,
    triage_node,
    update_state_node,
    urgent_exit_node,
)
from .state import AgentState


def build_graph(classifier_target_lang: str = "es"):
    set_classifier_target_lang(classifier_target_lang)
    graph = StateGraph(AgentState)
    graph.add_node("detect_language", detect_language_and_greet_node)
    graph.add_node("check_consent", check_consent_node)
    graph.add_node("red_flags", red_flags_node)
    graph.add_node("planner", planner_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("parse_clean", parse_and_clean_node)
    graph.add_node("update_state", update_state_node)
    graph.add_node("sufficiency_gate", sufficiency_gate_node)
    graph.add_node("normalize_struct", normalize_struct_node)
    graph.add_node("maybe_translate", maybe_translate_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("calibrate", calibrate_node)
    graph.add_node("triage", triage_node)
    graph.add_node("present_results", present_results_node)
    graph.add_node("safe_goodbye", safe_goodbye_node)
    graph.add_node("urgent_exit", urgent_exit_node)
    graph.add_node("await_user", await_user_node)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "check_consent")

    def _consent_route(state: AgentState) -> str:
        data = state.get("data", {})
        if data.get("consent") == "rejected":
            return "stop"
        return "continue"

    graph.add_conditional_edges(
        "check_consent",
        _consent_route,
        {"stop": "safe_goodbye", "continue": "red_flags"},
    )

    def _redflag_route(state: AgentState) -> str:
        data = state.get("data", {})
        if data.get("stop_reason") == "red_flags" or data.get("red_flags", {}).get(
            "present"
        ):
            return "urgent"
        return "continue"

    graph.add_conditional_edges(
        "red_flags",
        _redflag_route,
        {"urgent": "urgent_exit", "continue": "planner"},
    )

    def _planner_route(state: AgentState) -> str:
        internal = state.get("data", {}).get("_internal", {})

        # Emergency loop prevention - check total graph steps
        graph_steps = internal.get("graph_steps", 0)
        if graph_steps > 20:  # Prevent infinite recursion
            return "classify"  # Force exit to classification

        # Check planner-specific loop count
        planner_loops = internal.get("planner_loop_count", 0)
        if planner_loops > 6:  # Aggressive threshold
            return "classify"  # Force exit

        # Check conversation turns
        raw_dialog = state.get("data", {}).get("raw_dialog", [])
        if len(raw_dialog) > 25:  # Too many messages
            return "classify"  # Force exit

        action = internal.get("planner_action")
        if action == "classify":
            return "classify"
        if action == "ask":
            return "ask"
        return "await"

    graph.add_conditional_edges(
        "planner",
        _planner_route,
        {"ask": "ask_user", "await": "await_user", "classify": "normalize_struct"},
    )

    graph.add_edge("ask_user", "parse_clean")
    graph.add_edge("parse_clean", "update_state")
    graph.add_edge("update_state", "sufficiency_gate")

    def _sufficiency_route(state: AgentState) -> str:
        data = state.get("data", {})
        internal = data.get("_internal", {})

        # Emergency loop prevention - multiple safeguards
        graph_steps = internal.get("graph_steps", 0)
        planner_loops = internal.get("planner_loop_count", 0)
        raw_dialog = data.get("raw_dialog", [])

        # Force ready if any emergency condition is met
        emergency_exit = (
            graph_steps > 18
            or planner_loops > 5
            or len(raw_dialog) > 20
            or internal.get("conversation_stuck_exit", False)
        )

        if emergency_exit:
            # Log the emergency exit for debugging
            internal["emergency_exit_reason"] = {
                "graph_steps": graph_steps,
                "planner_loops": planner_loops,
                "dialog_length": len(raw_dialog),
                "stuck_exit": internal.get("conversation_stuck_exit", False),
            }
            return "ready"  # Force classification

        has_suff = internal.get("has_sufficiency")
        return "ready" if has_suff else "need"

    graph.add_conditional_edges(
        "sufficiency_gate",
        _sufficiency_route,
        {"ready": "normalize_struct", "need": "planner"},
    )

    graph.add_edge("normalize_struct", "maybe_translate")
    graph.add_edge("maybe_translate", "classifier")
    graph.add_edge("classifier", "calibrate")
    graph.add_edge("calibrate", "triage")
    graph.add_edge("triage", "present_results")

    graph.add_edge("safe_goodbye", END)
    graph.add_edge("urgent_exit", END)
    graph.add_edge("present_results", END)
    graph.add_edge("await_user", END)

    return graph.compile(
        checkpointer=None, interrupt_before=None, interrupt_after=None, debug=False
    )


__all__ = ["build_graph"]
