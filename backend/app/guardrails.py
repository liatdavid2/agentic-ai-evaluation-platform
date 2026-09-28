from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from app.tooling import MUTATING_TOOLS


AUTH_TOOLS = {
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
}

LOOKUP_TOOLS = {
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "calculate",
}

ORDER_MUTATION_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}

ALL_MUTATION_TOOLS = ORDER_MUTATION_TOOLS | {"modify_user_address"}


@dataclass
class GuardrailState:
    authenticated_user_id: str | None = None
    observed_user_ids: set[str] = field(default_factory=set)
    observed_order_ids: set[str] = field(default_factory=set)
    observed_product_ids: set[str] = field(default_factory=set)
    observed_item_ids: set[str] = field(default_factory=set)
    observed_payment_method_ids: set[str] = field(default_factory=set)
    confirmation_granted: bool = False
    violations: list[dict[str, Any]] = field(default_factory=list)


class GuardrailEngine:
    """
    Deterministic, task-agnostic runtime guardrails.

    The engine does NOT know the benchmark gold trajectory.
    It enforces invariants expected in production systems:
      - identifiers must be grounded in prior tool results
      - mutating tools require explicit confirmation
      - order/user/product ids must come from observed state
      - malformed or ungrounded calls are rejected before execution
      - the model receives a repair message and may retry
    """

    def __init__(self):
        self.state = GuardrailState()

    def mark_confirmation(self):
        self.state.confirmation_granted = True

    def clear_confirmation_after_mutation(self):
        self.state.confirmation_granted = False

    def observe_tool_result(self, tool_name: str, result: dict[str, Any]):
        if not isinstance(result, dict) or not result.get("ok"):
            return

        uid = result.get("user_id")
        if uid:
            uid = str(uid)
            self.state.observed_user_ids.add(uid)
            if tool_name in AUTH_TOOLS:
                self.state.authenticated_user_id = uid

        oid = result.get("order_id")
        if oid:
            self.state.observed_order_ids.add(str(oid))

        pid = result.get("product_id")
        if pid:
            self.state.observed_product_ids.add(str(pid))

        # get_user_details
        user = result.get("user")
        if isinstance(user, dict):
            for oid in user.get("orders", []) or []:
                self.state.observed_order_ids.add(str(oid))
            for pmid in (user.get("payment_methods", {}) or {}).keys():
                self.state.observed_payment_method_ids.add(str(pmid))

        for oid in result.get("order_ids", []) or []:
            self.state.observed_order_ids.add(str(oid))

        payment_methods = result.get("payment_methods", {}) or {}
        if isinstance(payment_methods, dict):
            for pmid in payment_methods.keys():
                self.state.observed_payment_method_ids.add(str(pmid))

        # get_order_details
        order = result.get("order")
        if isinstance(order, dict):
            order_id = order.get("order_id") or order.get("id")
            if order_id:
                self.state.observed_order_ids.add(str(order_id))
            user_id = order.get("user_id")
            if user_id:
                self.state.observed_user_ids.add(str(user_id))
            for item in order.get("items", []) or []:
                iid = item.get("item_id")
                pid = item.get("product_id")
                if iid:
                    self.state.observed_item_ids.add(str(iid))
                if pid:
                    self.state.observed_product_ids.add(str(pid))

        for item in result.get("items", []) or []:
            if isinstance(item, dict):
                iid = item.get("item_id")
                pid = item.get("product_id")
                if iid:
                    self.state.observed_item_ids.add(str(iid))
                if pid:
                    self.state.observed_product_ids.add(str(pid))

        # get_product_details
        product = result.get("product")
        if isinstance(product, dict):
            pid = product.get("product_id") or product.get("id")
            if pid:
                self.state.observed_product_ids.add(str(pid))
            variants = product.get("variants", {}) or {}
            if isinstance(variants, dict):
                for iid in variants.keys():
                    self.state.observed_item_ids.add(str(iid))

        for variant in result.get("variants", []) or []:
            if isinstance(variant, dict) and variant.get("item_id"):
                self.state.observed_item_ids.add(str(variant["item_id"]))

    def _reject(self, tool_name: str, code: str, message: str, args: dict[str, Any]):
        violation = {
            "tool": tool_name,
            "code": code,
            "message": message,
            "arguments": args,
        }
        self.state.violations.append(violation)
        return False, {
            "ok": False,
            "guardrail_rejected": True,
            "guardrail_code": code,
            "error": message,
            "repair_hint": self._repair_hint(code),
        }

    def _repair_hint(self, code: str) -> str:
        hints = {
            "auth_required":
                "Authenticate the user first using an authentication tool.",
            "ungrounded_user_id":
                "Use a user_id returned by an authentication or user lookup tool.",
            "ungrounded_order_id":
                "Use get_user_details or another lookup to obtain a real order_id before calling this tool.",
            "ungrounded_product_id":
                "Use a product_id previously observed in an order or product lookup.",
            "ungrounded_item_id":
                "Use item ids returned by get_order_details or get_product_details.",
            "ungrounded_payment_method":
                "Use a payment_method_id returned by get_user_details.",
            "confirmation_required":
                "Ask the user for explicit confirmation before executing the mutating action.",
        }
        return hints.get(code, "Repair the tool arguments using observed tool results and retry.")

    def validate(self, tool_name: str, args: dict[str, Any]):
        args = args or {}

        # Authentication / identity precondition.
        if tool_name not in AUTH_TOOLS and tool_name != "calculate":
            if not self.state.authenticated_user_id:
                return self._reject(
                    tool_name,
                    "auth_required",
                    "User must be authenticated before accessing personal or mutable retail state.",
                    args,
                )

        # Grounded identifiers.
        if "user_id" in args:
            uid = str(args["user_id"])
            if uid not in self.state.observed_user_ids:
                return self._reject(
                    tool_name,
                    "ungrounded_user_id",
                    f"user_id '{uid}' was not observed in a prior tool result.",
                    args,
                )

        if "order_id" in args:
            oid = str(args["order_id"])
            if oid not in self.state.observed_order_ids:
                return self._reject(
                    tool_name,
                    "ungrounded_order_id",
                    f"order_id '{oid}' was not observed in a prior tool result.",
                    args,
                )

        if "product_id" in args:
            pid = str(args["product_id"])
            if pid not in self.state.observed_product_ids:
                return self._reject(
                    tool_name,
                    "ungrounded_product_id",
                    f"product_id '{pid}' was not observed in a prior tool result.",
                    args,
                )

        for key in ("item_ids", "new_item_ids"):
            if key in args:
                for iid in args.get(key, []) or []:
                    iid = str(iid)
                    if iid not in self.state.observed_item_ids:
                        return self._reject(
                            tool_name,
                            "ungrounded_item_id",
                            f"item_id '{iid}' was not observed in a prior tool result.",
                            args,
                        )

        if "payment_method_id" in args:
            pmid = str(args["payment_method_id"])
            if self.state.observed_payment_method_ids and pmid not in self.state.observed_payment_method_ids:
                return self._reject(
                    tool_name,
                    "ungrounded_payment_method",
                    f"payment_method_id '{pmid}' was not observed in get_user_details.",
                    args,
                )

        # Side-effect gate.
        if tool_name in ALL_MUTATION_TOOLS and not self.state.confirmation_granted:
            return self._reject(
                tool_name,
                "confirmation_required",
                "Explicit user confirmation is required before this mutating action.",
                args,
            )

        return True, None
