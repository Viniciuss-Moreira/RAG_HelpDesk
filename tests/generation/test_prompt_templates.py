from src.generation import prompt_templates

def test_all_templates_render():
    query = "Como reinstalar o Windows?"
    context = "Para reinstalar o Windows, você precisa criar um pendrive de instalação..."

    templates = [
        prompt_templates.helpdesk_prompt,
        prompt_templates.concise_helpdesk_prompt,
        prompt_templates.informal_helpdesk_prompt,
        prompt_templates.step_by_step_prompt,
        prompt_templates.link_suggestion_prompt
    ]

    for template in templates:
        prompt = template.format(query=query, context=context)
        assert query in prompt
        assert context in prompt