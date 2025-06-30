import random
import argparse

def generate_helpdesk_qa(num_samples=100000, output_path='base_treinamento.txt'):
    # Question templates with placeholders (formal and informal variations)
    question_templates = [
        "Como {action} {target}?",
        "Como posso {action} {target}?",
        "Qual é a melhor forma de {action} {target}?",
        "Como faço para {action} {target}?",
        "O que devo fazer para {action} {target}?",
        "Preciso de ajuda para {action} {target}",
        "Não consigo {action} {target}, o que fazer?",
        "Está dando erro ao {action} {target}",
        "Por favor, me ajude a {action} {target}",
        "Tenho dificuldade em {action} {target}",
        # Informal variations
        "como {action} {target}?",
        "como posso {action} {target}?",
        "qual a melhor forma de {action} {target}?",
        "como faco para {action} {target}?",
        "o que devo fazer para {action} {target}?",
        "preciso de ajuda para {action} {target}",
        "nao consigo {action} {target}, o que fazer?",
        "ta dando erro ao {action} {target}",
        "por favor, me ajude a {action} {target}",
        "tenho dificuldade em {action} {target}",
        # Casual expressions
        "como que {action} {target}?",
        "me ajuda a {action} {target}",
        "nao sei como {action} {target}",
        "ta bugado, como {action} {target}?",
        "deu ruim aqui, como {action} {target}?"
    ]
    
    # Answer templates with placeholders and follow-ups
    answer_templates = [
        "Use a função de {tool} e {step}. Se não resolver, {fallback}.",
        "Execute {tool} para {step}. Caso persista, {fallback}.",
        "Abra {tool} e realize {step}. Se continuar o problema, {fallback}.",
        "Vá até {tool}, faça {step}. Se não funcionar, {fallback}.",
        "Primeiro, acesse {tool} e {step}. Caso o problema persista, {fallback}.",
        "Recomendo usar {tool} para {step}. Se o erro continuar, {fallback}.",
        "Siga estes passos: abra {tool} e {step}. Em último caso, {fallback}.",
        # Informal variations
        "use a funcao de {tool} e {step}. se nao resolver, {fallback}.",
        "execute {tool} para {step}. caso persista, {fallback}.",
        "abra {tool} e realize {step}. se continuar o problema, {fallback}.",
        "va ate {tool}, faca {step}. se nao funcionar, {fallback}.",
        # Casual expressions
        "tenta usar {tool} e {step}. se nao rolar, {fallback}.",
        "abre o {tool} e faz {step}. se continuar dando pau, {fallback}."
    ]
    
    # Expanded actions for questions (formal and informal)
    actions = [
        # Sistema e performance
        "liberar espaço em disco", "liberar espaco em disco", "limpar cache", 
        "apagar arquivos temporários", "apagar arquivos temporarios",
        "verificar atualizações de sistema", "verificar atualizacoes de sistema",
        "desfragmentar disco", "otimizar performance", "otimizar sistema",
        "acelerar inicialização", "acelerar inicializacao", "limpar registro", 
        "verificar integridade do sistema", "speedar o pc", "deixar o pc mais rapido",
        
        # Usuários e segurança
        "resetar senha de usuário", "resetar senha de usuario", "criar nova conta", 
        "alterar permissões", "alterar permissoes", "desbloquear conta", 
        "configurar autenticação", "configurar autenticacao", "ativar firewall",
        "atualizar antivírus", "atualizar antivirus", "remover malware", 
        "tirar virus", "limpar virus", "configurar backup",
        
        # Rede e conectividade
        "configurar conexão de rede", "configurar conexao de rede", 
        "testar conectividade", "resetar configurações de rede", 
        "resetar configuracoes de rede", "configurar proxy", 
        "conectar impressora", "mapear unidade de rede", "configurar VPN",
        "resolver problemas de DNS", "configurar Wi-Fi", "configurar wifi",
        "conectar na internet", "arrumar a internet",
        
        # Hardware e drivers
        "instalar driver de vídeo", "instalar driver de video", 
        "atualizar drivers", "configurar som", "calibrar monitor", 
        "testar hardware", "verificar temperatura", "configurar impressora", 
        "instalar periféricos", "instalar perifericos", 
        "resolver conflitos de hardware", "arrumar placa de video",
        
        # Software e aplicativos específicos
        "instalar programa", "desinstalar software", "atualizar aplicativo",
        "resolver erro de aplicação", "resolver erro de aplicacao", 
        "recuperar dados", "configurar email", "configurar Outlook",
        "instalar Office", "resolver erro no Excel", "configurar Word",
        "sincronizar arquivos", "restaurar sistema", "fazer backup",
        "instalar Chrome", "configurar Firefox", "atualizar navegador",
        
        # Problemas específicos com códigos
        "corrigir tela azul", "resolver travamento", "acelerar boot",
        "corrigir erro de DLL", "resolver problema de memória", 
        "resolver problema de memoria", "limpar vírus", "limpar virus",
        "resolver erro 0x80070005", "corrigir BSOD 0x0000007B",
        "resolver erro de sistema", "corrigir erro crítico", 
        "corrigir erro critico", "resolver tela preta",
        
        # Gírias técnicas
        "destravar o pc", "arrumar o computador", "consertar o sistema",
        "dar um jeito no pc", "resolver esse pepino", "mexer nas configuracoes"
    ]
    
    # Expanded targets (formal and informal)
    targets = [
        "no Windows", "no windows", "no Windows 10", "no windows 10",
        "no Windows 11", "no windows 11", "no Windows Server", 
        "no Windows Server 2019", "no Windows Server 2022",
        "no macOS", "no macos", "no Mac", "no mac", 
        "no Linux", "no linux", "no Ubuntu", "no ubuntu", 
        "no CentOS", "no centos", "no Debian", "no debian",
        "no computador", "no notebook", "no servidor", "na máquina", 
        "na maquina", "no desktop", "no laptop", "no sistema",
        "na rede corporativa", "no Office 365", "no office 365",
        "no Teams", "no teams", "no Outlook", "no outlook",
        "no Excel", "no excel", "no Word", "no word",
        "no PowerPoint", "no powerpoint", "no Chrome", "no chrome",
        "no Firefox", "no firefox", "no Edge", "no edge",
        "na impressora", "no roteador", "no servidor de email",
        "no pc", "no note", "na workstation"
    ]
    
    # Expanded tools (formal and informal)
    tools = [
        # Windows tools
        "Limpeza de Disco", "limpeza de disco", "Gerenciador de Disco", 
        "gerenciador de disco", "Painel de Controle", "painel de controle",
        "Configurações do Windows", "configuracoes do windows", 
        "Gerenciador de Tarefas", "gerenciador de tarefas",
        "Editor de Registro", "editor de registro", "Prompt de Comando", 
        "prompt de comando", "PowerShell", "powershell",
        "Gerenciador de Dispositivos", "gerenciador de dispositivos",
        "Monitor de Recursos", "monitor de recursos", 
        "Verificador de Arquivos do Sistema", "verificador de arquivos do sistema",
        
        # Network tools
        "Central de Rede", "central de rede", "Diagnóstico de Rede", 
        "diagnostico de rede", "Configurações de Proxy", 
        "configuracoes de proxy", "Gerenciador de Conexões", 
        "gerenciador de conexoes", "Painel de Firewall", "painel de firewall",
        
        # Security tools
        "Windows Defender", "windows defender", "Configurações de Segurança", 
        "configuracoes de seguranca", "Editor de Políticas", 
        "editor de politicas", "Gerenciador de Credenciais", 
        "gerenciador de credenciais", "Backup e Restauração", 
        "backup e restauracao",
        
        # System tools
        "Desfragmentador", "desfragmentador", "Verificação de Disco", 
        "verificacao de disco", "Restauração do Sistema", 
        "restauracao do sistema", "Agendador de Tarefas", 
        "agendador de tarefas", "Visualizador de Eventos", 
        "visualizador de eventos", "Informações do Sistema", 
        "informacoes do sistema",
        
        # Office tools
        "Configurações do Office", "configuracoes do office",
        "Centro de Administração", "centro de administracao",
        "Painel do Outlook", "painel do outlook",
        
        # Casual terms
        "ferramenta de limpeza", "programinha de limpeza",
        "utilitario do sistema", "config do windows"
    ]
    
    # Expanded steps (formal and informal)
    steps = [
        # Cleaning steps
        "apagar arquivos temporários e da lixeira", 
        "apagar arquivos temporarios e da lixeira",
        "excluir cache do sistema", "limpar arquivos de log antigos",
        "remover downloads desnecessários", "remover downloads desnecessarios",
        "esvaziar pasta temp", "limpar cache de navegador",
        "deletar lixo do sistema", "fazer faxina no pc",
        
        # System maintenance
        "executar varredura de vírus", "executar varredura de virus",
        "checar integridade de disco", "verificar atualizações pendentes", 
        "verificar atualizacoes pendentes", "desfragmentar unidades",
        "otimizar configurações de energia", "otimizar configuracoes de energia",
        "reinicializar serviços", "reinicializar servicos", 
        "rodar o antivirus", "scanear o sistema",
        
        # User management
        "redefinir senha do usuário", "redefinir senha do usuario",
        "verificar permissões de conta", "verificar permissoes de conta",
        "desbloquear conta bloqueada", "criar backup de perfil",
        "sincronizar dados do usuário", "sincronizar dados do usuario",
        "resetar a senha", "trocar a senha",
        
        # Network troubleshooting
        "renovar endereço IP", "renovar endereco IP", "limpar cache DNS",
        "testar conectividade", "verificar configurações de proxy", 
        "verificar configuracoes de proxy", "reiniciar adaptador de rede",
        "atualizar drivers de rede", "resetar a rede", "renovar o IP",
        
        # Software maintenance
        "desinstalar programas não utilizados", 
        "desinstalar programas nao utilizados",
        "atualizar drivers desatualizados", "verificar conflitos de software",
        "reparar instalação corrompida", "reparar instalacao corrompida",
        "restaurar configurações padrão", "restaurar configuracoes padrao",
        "executar modo de compatibilidade", "rodar como administrador",
        "dar permissao de admin", "executar em modo seguro",
        
        # Specific error fixes
        "executar sfc /scannow", "rodar chkdsk /f", "executar dism",
        "limpar boot sequence", "reparar MBR", "resetar winsock",
        "fazer restore point", "voltar backup anterior"
    ]
    
    # Expanded fallbacks (formal and informal)
    fallbacks = [
        # Technical escalation
        "verifique atualizações de sistema ou entre em contato com o suporte técnico",
        "verifique atualizacoes de sistema ou entre em contato com o suporte tecnico",
        "reinicie a máquina e tente novamente, se persistir chame o administrador",
        "reinicie a maquina e tente novamente, se persistir chame o administrador",
        "abra um chamado junto ao setor de TI", "consulte a documentação oficial",
        "consulte a documentacao oficial", "verifique logs de erro e reporte ao fornecedor",
        
        # Alternative solutions
        "tente executar em modo de segurança", "tente executar em modo de seguranca",
        "use a ferramenta de solução de problemas", 
        "use a ferramenta de solucao de problemas",
        "verifique se há conflitos com outros programas", 
        "verifique se ha conflitos com outros programas",
        "execute como administrador", "restaure o sistema para um ponto anterior",
        "reinstale o programa problemático", "reinstale o programa problematico",
        
        # Documentation and support
        "consulte o manual do usuário", "consulte o manual do usuario",
        "entre em contato com o fabricante", 
        "procure por atualizações no site oficial", 
        "procure por atualizacoes no site oficial",
        "verifique fóruns de suporte", "verifique foruns de suporte",
        "documente o erro e escale para nível 2", 
        "documente o erro e escale para nivel 2",
        "agende manutenção preventiva", "agende manutencao preventiva",
        
        # Casual solutions
        "tenta reiniciar e ve se resolve", "da uma olhada no Google",
        "chama alguem que entende", "formata tudo de novo",
        "liga pro TI que eles resolvem", "faz backup e reinstala",
        "procura tutorial no YouTube", "pede ajuda no forum"
    ]
    
    # Generate combinations randomly until reaching desired count
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            q_template = random.choice(question_templates)
            action = random.choice(actions)
            target = random.choice(targets)
            question = q_template.format(action=action, target=target)
            
            a_template = random.choice(answer_templates)
            tool = random.choice(tools)
            step = random.choice(steps)
            fallback = random.choice(fallbacks)
            answer = a_template.format(tool=tool, step=step, fallback=fallback)
            
            f.write(f"Pergunta: {question}\nResposta: {answer}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gerar dataset de QA para Helpdesk')
    parser.add_argument('-n', '--num', type=int, default=50000,
                        help='Quantidade de exemplos a gerar (padrão: 50000)')
    parser.add_argument('-o', '--output', type=str, default='base_treinamento.txt',
                        help='Caminho do arquivo de saída')
    
    args = parser.parse_args()
    generate_helpdesk_qa(num_samples=args.num, output_path=args.output)
    print(f"Dataset gerado com {args.num} exemplos em '{args.output}'.")