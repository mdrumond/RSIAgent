"""Small closed schema vocabulary shared by Pi declarations and host validation."""

STRING = {"type": "string"}
INTEGER = {"type": "integer", "minimum": 0}


def obj(properties=None, required=None):
    properties = properties or {}
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def validate(value, schema):
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "integer": type(value) is int,
        "number": type(value) in (int, float), "boolean": type(value) is bool,
    }
    if kind and not valid[kind]:
        raise ValueError("tool argument has wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("tool argument is outside the allowed values")
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= value.keys():
            raise ValueError("required tool argument is missing")
        if schema.get("additionalProperties") is False and not value.keys() <= properties.keys():
            raise ValueError("unexpected tool argument")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key])
    elif kind == "array":
        if len(value) > schema.get("maxItems", 100000):
            raise ValueError("too many tool arguments")
        for item in value:
            validate(item, schema.get("items", {}))
    elif kind in ("integer", "number"):
        if not schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf")):
            raise ValueError("tool argument is outside bounds")
