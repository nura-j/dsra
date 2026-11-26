"""Prompt settings for one model.

``PromptConfig(model_name)`` takes the registered template and title-casing of
the model; either can be overridden, e.g. to trace with another wording.
"""

from dataclasses import dataclass
from typing import Optional

from .format import format_example, format_prompt
from .models import resolve_model
from .templates import get_template, uses_title_case


@dataclass
class PromptConfig:
    model_name: str
    template: Optional[str] = None
    title_case: Optional[bool] = None

    def __post_init__(self):
        self.model_name = resolve_model(self.model_name)
        if self.template is None:
            self.template = get_template(self.model_name)
        if "{}" not in self.template: #todo: add deafualt prompt
            raise ValueError("template must contain {} for the definition")
        if self.title_case is None:
            self.title_case = uses_title_case(self.model_name)

    def prompt(self, definition):
        return format_prompt(definition, self.template, self.title_case)

    def example(self, definition, definiendum):
        return format_example(definition, definiendum, self.template, self.title_case)
