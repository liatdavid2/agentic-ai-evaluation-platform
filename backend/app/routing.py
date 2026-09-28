from typing import Iterable

COMMON_POLICY = """
At the beginning of the conversation, authenticate the user via email or via name + zip.
Do not invent information not provided by the user or tools.
Use at most one tool call at a time.
Only help one user per conversation.
Before any database-changing action, explicit user confirmation is required.
"""

POLICY_SNIPPETS = {
    "cancel": """
Cancel pending order:
- Only pending orders may be cancelled.
- Check status first.
- Allowed reasons: no longer needed, ordered by mistake.
""",
    "modify_items": """
Modify pending order items:
- Only pending orders may be modified.
- New item must be an available variant of the same product.
- Collect all item changes before the single modify call.
- User must provide a payment method for price difference.
""",
    "modify_address": """
Modify pending order address:
- Only pending orders may be modified.
- Confirm the destination details before changing the address.
""",
    "modify_payment": """
Modify pending order payment:
- Only pending orders may be modified.
- New payment must differ from original.
- Gift card must have sufficient balance.
""",
    "return": """
Return delivered order:
- Only delivered orders may be returned.
- Confirm order id and item list.
- Refund must go to original payment method or an existing gift card.
""",
    "exchange": """
Exchange delivered order:
- Only delivered orders may be exchanged.
- New item must be an available variant of the same product.
- Collect all requested exchanges before the single exchange call.
- User must provide a payment method for price difference.
""",
    "human": """
Transfer:
- Transfer to a human iff the request cannot be handled within supported actions.
""",
}

TOOL_GROUPS = {
    "auth": ["find_user_id_by_name_zip", "find_user_id_by_email"],
    "lookup": ["get_user_details", "get_order_details", "get_product_details", "calculate"],
    "cancel": ["cancel_pending_order"],
    "modify_items": ["modify_pending_order_items"],
    "modify_address": ["modify_pending_order_address", "modify_user_address"],
    "modify_payment": ["modify_pending_order_payment"],
    "return": ["return_delivered_order_items"],
    "exchange": ["exchange_delivered_order_items"],
    "human": ["transfer_to_human_agents"],
}

def detect_intents(text: str) -> list[str]:
    t = (text or "").lower()
    intents = []
    def has(*terms): return any(x in t for x in terms)
    if has("cancel"): intents.append("cancel")
    if has("return", "refund"): intents.append("return")
    if has("exchange"): intents.append("exchange")
    if has("modify", "change", "update"):
        if has("address", "suite", "street", "zip"): intents.append("modify_address")
        if has("payment", "gift card", "paypal", "credit card"): intents.append("modify_payment")
        if has("item", "shirt", "keyboard", "shoe", "boots", "lamp", "bottle", "variant", "size", "color", "material"):
            intents.append("modify_items")
    if has("human", "representative", "agent for help", "transfer"): intents.append("human")
    # Always permit lookup; most tasks require environment inspection.
    if not intents:
        intents = ["modify_items", "return", "exchange", "cancel", "modify_address", "modify_payment", "human"]
    # de-dupe stable
    out=[]
    for x in intents:
        if x not in out: out.append(x)
    return out

def policy_for_task(text: str, max_chars: int) -> str:
    intents = detect_intents(text)
    parts = [COMMON_POLICY]
    for i in intents:
        if i in POLICY_SNIPPETS:
            parts.append(POLICY_SNIPPETS[i])
    return "\n".join(parts)[:max_chars]

def tool_names_for_task(text: str) -> list[str]:
    intents = detect_intents(text)
    names = TOOL_GROUPS["auth"] + TOOL_GROUPS["lookup"]
    for i in intents:
        names += TOOL_GROUPS.get(i, [])
    out=[]
    for n in names:
        if n not in out: out.append(n)
    return out
