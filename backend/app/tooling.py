import ast
import copy
import operator as op
from typing import Any

_ALLOWED_OPS = {ast.Add:op.add, ast.Sub:op.sub, ast.Mult:op.mul, ast.Div:op.truediv, ast.USub:op.neg, ast.UAdd:op.pos}

def safe_calculate(expression: str):
    def _eval(node):
        if isinstance(node, ast.Expression): return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS: return _ALLOWED_OPS[type(node.op)](_eval(node.left),_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS: return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported calculation")
    return _eval(ast.parse(expression,mode="eval"))

TOOL_SCHEMAS = [
    {"type":"function","name":"find_user_id_by_name_zip","description":"Locate/authenticate a user by name and ZIP.","parameters":{"type":"object","properties":{"first_name":{"type":"string"},"last_name":{"type":"string"},"zip":{"type":"string"}},"required":["first_name","last_name","zip"],"additionalProperties":False}},
    {"type":"function","name":"find_user_id_by_email","description":"Locate/authenticate a user by email.","parameters":{"type":"object","properties":{"email":{"type":"string"}},"required":["email"],"additionalProperties":False}},
    {"type":"function","name":"get_user_details","description":"Get user profile, payment methods and order ids.","parameters":{"type":"object","properties":{"user_id":{"type":"string"}},"required":["user_id"],"additionalProperties":False}},
    {"type":"function","name":"get_order_details","description":"Get one order.","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"],"additionalProperties":False}},
    {"type":"function","name":"get_product_details","description":"Get product variants, availability, options and prices.","parameters":{"type":"object","properties":{"product_id":{"type":"string"}},"required":["product_id"],"additionalProperties":False}},
    {"type":"function","name":"calculate","description":"Evaluate simple arithmetic.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":False}},
    {"type":"function","name":"cancel_pending_order","description":"Cancel a pending order.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"reason":{"type":"string"}},"required":["order_id","reason"],"additionalProperties":False}},
    {"type":"function","name":"modify_pending_order_items","description":"Modify pending order item variants.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"item_ids":{"type":"array","items":{"type":"string"}},"new_item_ids":{"type":"array","items":{"type":"string"}},"payment_method_id":{"type":"string"}},"required":["order_id","item_ids","new_item_ids","payment_method_id"],"additionalProperties":False}},
    {"type":"function","name":"modify_pending_order_address","description":"Change a pending order shipping address.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"address1":{"type":"string"},"address2":{"type":"string"},"city":{"type":"string"},"state":{"type":"string"},"country":{"type":"string"},"zip":{"type":"string"}},"required":["order_id","address1","address2","city","state","country","zip"],"additionalProperties":False}},
    {"type":"function","name":"modify_pending_order_payment","description":"Change payment method for a pending order.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"payment_method_id":{"type":"string"}},"required":["order_id","payment_method_id"],"additionalProperties":False}},
    {"type":"function","name":"return_delivered_order_items","description":"Return items from a delivered order.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"item_ids":{"type":"array","items":{"type":"string"}},"payment_method_id":{"type":"string"}},"required":["order_id","item_ids","payment_method_id"],"additionalProperties":False}},
    {"type":"function","name":"exchange_delivered_order_items","description":"Exchange delivered items for variants of same product.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"item_ids":{"type":"array","items":{"type":"string"}},"new_item_ids":{"type":"array","items":{"type":"string"}},"payment_method_id":{"type":"string"}},"required":["order_id","item_ids","new_item_ids","payment_method_id"],"additionalProperties":False}},
    {"type":"function","name":"modify_user_address","description":"Change default user address.","parameters":{"type":"object","properties":{"user_id":{"type":"string"},"address1":{"type":"string"},"address2":{"type":"string"},"city":{"type":"string"},"state":{"type":"string"},"country":{"type":"string"},"zip":{"type":"string"}},"required":["user_id","address1","address2","city","state","country","zip"],"additionalProperties":False}},
    {"type":"function","name":"transfer_to_human_agents","description":"Transfer to a human agent.","parameters":{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":False}},
]

MUTATING_TOOLS={"cancel_pending_order","modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment","return_delivered_order_items","exchange_delivered_order_items","modify_user_address"}

def filtered_schemas(names):
    s=set(names)
    return [x for x in TOOL_SCHEMAS if x["name"] in s]

class RetailToolExecutor:
    def __init__(self,db):
        self.db=copy.deepcopy(db); self.authenticated_user_id=None; self.calls=[]

    def _record(self,name,args,status,result):
        self.calls.append({"name":name,"arguments":args,"status":status,"result":result,"useful":status=="ok"})

    def _item_product(self,item_id):
        for pid,p in self.db.get("products",{}).items():
            if item_id in p.get("variants",{}): return pid,p["variants"][item_id]
        return None,None

    def _payment_belongs_to_user(self,user_id,payment_id):
        return payment_id in self.db.get("users",{}).get(user_id,{}).get("payment_methods",{})

    def execute(self,name,args):
        try:
            fn=getattr(self,f"_tool_{name}",None)
            if not fn:
                result={"ok":False,"error":f"Unknown tool: {name}"};self._record(name,args,"invalid",result);return result
            result=fn(**args);status="ok" if result.get("ok",True) else "error";self._record(name,args,status,result);return result
        except Exception as e:
            result={"ok":False,"error":f"{type(e).__name__}: {e}"};self._record(name,args,"error",result);return result

    # Compact results by design: never return entire DB objects when unnecessary.
    def _tool_find_user_id_by_name_zip(self,first_name,last_name,zip):
        for uid,u in self.db["users"].items():
            full=str(u.get("name",""))
            # dataset may store name as string
            ok_name=(f"{first_name} {last_name}".lower()==full.lower())
            if isinstance(u.get("name"),dict):
                n=u["name"];ok_name=n.get("first_name","").lower()==first_name.lower() and n.get("last_name","").lower()==last_name.lower()
            if ok_name and str(u.get("address",{}).get("zip"))==str(zip):
                self.authenticated_user_id=uid;return {"ok":True,"user_id":uid}
        return {"ok":False,"error":"User not found"}

    def _tool_find_user_id_by_email(self,email):
        for uid,u in self.db["users"].items():
            if str(u.get("email","")).lower()==email.lower():
                self.authenticated_user_id=uid;return {"ok":True,"user_id":uid}
        return {"ok":False,"error":"User not found"}

    def _tool_get_user_details(self,user_id):
        u=self.db["users"].get(user_id)
        if not u:return {"ok":False,"error":"User not found"}
        return {"ok":True,"user_id":user_id,"email":u.get("email"),"address":u.get("address"),"payment_methods":u.get("payment_methods",{}),"order_ids":u.get("orders",[])}

    def _tool_get_order_details(self,order_id):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        items=[{"item_id":x.get("item_id"),"product_id":x.get("product_id"),"name":x.get("name"),"price":x.get("price"),"options":x.get("options")} for x in o.get("items",[])]
        return {"ok":True,"order_id":order_id,"user_id":o.get("user_id"),"status":o.get("status"),"address":o.get("address"),"items":items,"payment_history":o.get("payment_history",[]),"fulfillments":o.get("fulfillments",[])}

    def _tool_get_product_details(self,product_id):
        p=self.db["products"].get(product_id)
        if not p:return {"ok":False,"error":"Product not found"}
        variants=[{"item_id":iid,"options":v.get("options"),"available":v.get("available"),"price":v.get("price")} for iid,v in p.get("variants",{}).items()]
        return {"ok":True,"product_id":product_id,"name":p.get("name"),"variants":variants}

    def _tool_calculate(self,expression): return {"ok":True,"result":safe_calculate(expression)}

    def _tool_cancel_pending_order(self,order_id,reason):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="pending":return {"ok":False,"error":"Order is not pending"}
        if reason not in {"no longer needed","ordered by mistake"}:return {"ok":False,"error":"Invalid cancellation reason"}
        o["status"]="cancelled";o["cancellation_reason"]=reason;return {"ok":True,"order_id":order_id,"status":"cancelled"}

    def _tool_modify_pending_order_items(self,order_id,item_ids,new_item_ids,payment_method_id):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="pending":return {"ok":False,"error":"Order is not pending"}
        if len(item_ids)!=len(new_item_ids):return {"ok":False,"error":"Item lists length mismatch"}
        by_id={x["item_id"]:x for x in o.get("items",[])}
        replacements={}
        for old,new in zip(item_ids,new_item_ids):
            if old not in by_id:return {"ok":False,"error":f"Item {old} not in order"}
            old_pid=by_id[old]["product_id"]
            new_pid,new_item=self._item_product(new)
            if not new_item or not new_item.get("available"):return {"ok":False,"error":f"New item {new} unavailable"}
            if old_pid!=new_pid:return {"ok":False,"error":"New item must be same product"}
            replacements[old]=(new_pid,new,new_item)

        new_order_items=[]
        for item in o.get("items",[]):
            iid=item.get("item_id")
            if iid not in replacements:
                new_order_items.append(item)
                continue
            pid,new_iid,var=replacements[iid]
            new_order_items.append({
                "name":self.db["products"][pid].get("name"),
                "product_id":pid,
                "item_id":new_iid,
                "price":var.get("price"),
                "options":var.get("options"),
            })
        o["items"]=new_order_items
        o["modified_payment_method_id"]=payment_method_id
        return {"ok":True,"order_id":order_id,"status":o.get("status"),"changed":len(item_ids)}

    def _tool_modify_pending_order_address(self,order_id,address1,address2,city,state,country,zip):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="pending":return {"ok":False,"error":"Order is not pending"}
        o["address"]={"address1":address1,"address2":address2,"city":city,"state":state,"country":country,"zip":zip}
        return {"ok":True,"order_id":order_id,"status":o["status"]}

    def _tool_modify_pending_order_payment(self,order_id,payment_method_id):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="pending":return {"ok":False,"error":"Order is not pending"}
        o["modified_payment_method_id"]=payment_method_id
        return {"ok":True,"order_id":order_id,"payment_method_id":payment_method_id}

    def _tool_return_delivered_order_items(self,order_id,item_ids,payment_method_id):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="delivered":return {"ok":False,"error":"Order is not delivered"}
        order_item_ids={x.get("item_id") for x in o.get("items",[])}
        if not set(item_ids).issubset(order_item_ids):return {"ok":False,"error":"One or more items not in order"}
        o["return_request"]={
            "item_ids":list(item_ids),
            "payment_method_id":payment_method_id,
        }
        return {"ok":True,"order_id":order_id,"status":o.get("status"),"returned_items":len(item_ids)}

    def _tool_exchange_delivered_order_items(self,order_id,item_ids,new_item_ids,payment_method_id):
        o=self.db["orders"].get(order_id)
        if not o:return {"ok":False,"error":"Order not found"}
        if o.get("status")!="delivered":return {"ok":False,"error":"Order is not delivered"}
        if len(item_ids)!=len(new_item_ids):return {"ok":False,"error":"Item lists length mismatch"}
        by_id={x["item_id"]:x for x in o.get("items",[])}
        for old,new in zip(item_ids,new_item_ids):
            if old not in by_id:return {"ok":False,"error":f"Item {old} not in order"}
            old_pid=by_id[old].get("product_id")
            new_pid,new_item=self._item_product(new)
            if not new_item or not new_item.get("available"):return {"ok":False,"error":f"New item {new} unavailable"}
            if old_pid!=new_pid:return {"ok":False,"error":"New item must be same product"}
        o["exchange_request"]={
            "item_ids":list(item_ids),
            "new_item_ids":list(new_item_ids),
            "payment_method_id":payment_method_id,
        }
        return {"ok":True,"order_id":order_id,"status":o.get("status"),"exchanged_items":len(item_ids)}

    def _tool_modify_user_address(self,user_id,address1,address2,city,state,country,zip):
        u=self.db["users"].get(user_id)
        if not u:return {"ok":False,"error":"User not found"}
        u["address"]={"address1":address1,"address2":address2,"city":city,"state":state,"country":country,"zip":zip}
        return {"ok":True,"user_id":user_id}

    def _tool_transfer_to_human_agents(self,summary):
        return {"ok":True,"transferred":True,"summary":summary[:300]}
