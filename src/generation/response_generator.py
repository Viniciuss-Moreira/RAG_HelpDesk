from src.generation.llm_client import call_llm
from src.generation.prompt_templates import (
    helpdesk_prompt,
    concise_helpdesk_prompt,
    informal_helpdesk_prompt,
    step_by_step_prompt,
    link_suggestion_prompt
)

def build_prompt(query: str, context: str, mode: str = "default") -> str:
    prompt_map = {
        "default": helpdesk_prompt,
        "concise": concise_helpdesk_prompt,
        "informal": informal_helpdesk_prompt,
        "step_by_step": step_by_step_prompt,
        "with_links": link_suggestion_prompt
    }
    template = prompt_map.get(mode, helpdesk_prompt)
    return template.format(context=context, query=query)

def generate_answer(query: str, context: str, mode: str = "default") -> str:
    prompt = build_prompt(query, context, mode)
    return call_llm(prompt)
