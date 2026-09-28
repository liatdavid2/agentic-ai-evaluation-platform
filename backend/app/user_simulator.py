import re


class ScriptedUserSimulator:
    """
    Lightweight deterministic user simulator.

    It does not call another LLM, so it does not add API token cost.
    It is intended to unlock multi-turn benchmark flows such as explicit
    confirmation before cancel/return/exchange/modify actions.

    The simulator uses the original task instructions as its script source.
    """

    def __init__(self, task_text: str, max_turns: int = 3):
        self.task_text = task_text or ""
        self.max_turns = max_turns
        self.turns = 0

    def can_respond(self) -> bool:
        return self.turns < self.max_turns

    def _looks_like_confirmation_request(self, text: str) -> bool:
        t = (text or "").lower()
        markers = [
            "confirm",
            "confirmation",
            "do you want me to",
            "would you like me to",
            "shall i",
            "should i proceed",
            "may i proceed",
            "proceed with",
            "is that okay",
            "is this correct",
        ]
        return any(m in t for m in markers)

    def _extract_scripted_confirmation(self) -> str | None:
        """
        Best-effort extraction of benchmark-authored conditional behavior.
        Examples:
          - 'If the agent asks for confirmation, ...'
          - 'When asked to confirm, ...'
        We preserve the task wording when a short directive can be extracted.
        """
        text = self.task_text

        patterns = [
            r"if (?:the )?agent asks? (?:you )?(?:for )?confirmation[,:\s]+(.+?)(?:\n|$)",
            r"when asked (?:for|to) confirm[^,:\n]*[,:\s]+(.+?)(?:\n|$)",
            r"if asked (?:for|to) confirm[^,:\n]*[,:\s]+(.+?)(?:\n|$)",
        ]

        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if m:
                directive = m.group(1).strip(" .")
                if directive and len(directive) <= 300:
                    # Convert common imperative descriptions into a user reply.
                    low = directive.lower()
                    if low.startswith(("say ", "respond ", "reply ")):
                        directive = re.sub(r"^(say|respond|reply)\s+", "", directive, flags=re.I)
                    return directive

        return None

    def respond(self, agent_text: str) -> str | None:
        if not self.can_respond():
            return None

        if self._looks_like_confirmation_request(agent_text):
            self.turns += 1
            scripted = self._extract_scripted_confirmation()
            if scripted:
                return scripted
            return "Yes, I confirm. Please proceed."

        # Do not invent answers to database questions. Those should be resolved
        # through tools; returning None lets the evaluator terminate the run.
        return None
