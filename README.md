# Conferência Fiscal

Sistema de cruzamento de notas fiscais (entrada, NFC-e e cupons) com o SPED do mês.  
Gera um relatório de divergências e, opcionalmente, solicita um parecer da IA via Cloudflare Workers AI para interpretar os achados.

## O que faz

- Compara notas de entrada, saídas e cupons fiscais contra o SPED mensal.
- Identifica divergências de valor, chave, CFOP e outros campos relevantes.
- Exibe totais por arquivo e a diferença de valor por nota.
- Gera PDF estruturado para envio ao contador.
- Envia opcionalmente um print da tela para análise visual da IA (Llama 3.2 Vision via Cloudflare).
- Testa a conexão com a Cloudflare antes de usar a análise de IA.

## Executável

O repositório não inclui executáveis nem arquivos locais de credenciais. Para gerar o executável no Windows, instale as dependências e o PyInstaller, depois execute `.\build_exe.ps1`. O resultado será criado em `dist\ConferenciaFiscal\`.

## Como executar pelo código-fonte
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

O aplicativo abre no navegador em `http://127.0.0.1:8501`.

## Credenciais da Cloudflare

A análise de IA depende de um **Account ID** e um **API Token** da Cloudflare com permissão para Workers AI.

Informe as credenciais nos campos da interface ou configure as variáveis de ambiente `CLOUDFLARE_ACCOUNT_ID` e `CLOUDFLARE_API_TOKEN`. O projeto não inclui credenciais padrão nem as lê de arquivos locais.

Exemplo no PowerShell:
```
$env:CLOUDFLARE_ACCOUNT_ID = "SEU_ACCOUNT_ID"
$env:CLOUDFLARE_API_TOKEN = "SEU_API_TOKEN"
```

> Não publique nem versione credenciais reais. Se um token for exposto, revogue-o e gere outro.

## Estrutura do projeto

```
├── app.py          # Interface Streamlit e lógica de cruzamento
├── launcher.py     # Inicializador que abre o navegador automaticamente
├── build_exe.ps1   # Script de geração do .exe com PyInstaller
├── requirements.txt
├── .gitignore
└── README.md
```

## Tecnologias

- Python 3.14
- Streamlit (interface web)
- Pandas (manipulação de dados)
- ReportLab (geração de PDF)
- Requests (integragem com Cloudflare API)
- PyInstaller (empacotamento do .exe)

## Licença

Uso interno. Se for adaptar para terceiros, revise os termos de uso da Cloudflare Workers AI e da Meta Llama 3.2.