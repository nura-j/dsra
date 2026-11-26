"""Turning a definition into a prompt, and a prompt plus answer into training text."""


def format_prompt(definition, template, title_case=False):
    """``template`` with the definition filled in (title-cased if asked)."""
    if "{}" not in template:
        raise ValueError(f"template {template!r} has no {{}} for the definition")
    return template.format(definition.title() if title_case else definition)


def format_example(definition, definiendum, template, title_case=False):
    """Prompt followed directly by the definiendum (the fine-tuning text)."""
    return format_prompt(definition, template, title_case) + definiendum
