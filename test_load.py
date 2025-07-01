print("--- INICIANDO TESTE DE CARREGAMENTO DIRETO ---")
try:
    # Tenta importar e chamar a função problemática diretamente.
    from api.rag_chain import get_rag_chain
    print("Importação de 'api.rag_chain' bem-sucedida.")
    print("Chamando get_rag_chain() para verificar os parâmetros...")
    get_rag_chain()
    print("--- TESTE DE CARREGAMENTO CONCLUÍDO ---")
except Exception as e:
    print(f"\n!!!!!! OCORREU UM ERRO DURANTE O TESTE !!!!!!")
    print(f"Erro: {e}")

