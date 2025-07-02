helpdesk_prompt = """
Contexto técnico:
{context}

Com base nisso, responda à seguinte pergunta:
{query}
"""

concise_helpdesk_prompt = """
Contexto técnico:
{context}

Responda de forma breve e objetiva:
{query}
"""

informal_helpdesk_prompt = """
Oi! Aqui está o que você precisa saber com base no contexto:
{context}

Pergunta:
{query}

Resposta descontraída:
"""

step_by_step_prompt = """
Contexto técnico:
{context}

Por favor, explique passo a passo como resolver:
{query}
"""

link_suggestion_prompt = """
Contexto técnico:
{context}

Responda a pergunta: {query}

Se possível, inclua links úteis para mais informações.
"""

RAG_PROMPT_TEMPLATE = """[INST]
Você é um assistente de TI. Sua tarefa é responder à pergunta do usuário em Português, usando apenas as informações do contexto abaixo. Seja direto e conciso.
<CONTEXTO>
{context}
</CONTEXTO>
<PERGUNTA>
{query}
</PERGUNTA>
[/INST]
"""