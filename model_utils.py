import re


def make_safe_feature_name(feature: str, category: object) -> str:
    safe_category = re.sub(r"[^A-Za-z0-9_]", "_", str(category))
    return f"{feature}_{safe_category}"
