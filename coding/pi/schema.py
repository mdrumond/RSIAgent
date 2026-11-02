"""Domain JSON schema declarations; native Pi validates tool arguments."""

STRING = {"type": "string"}
INTEGER = {"type": "integer", "minimum": 0}


def obj(properties=None, required=None):
    properties = properties or {}
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}
