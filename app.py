import streamlit as st
import pandas as pd
import io
import os
import hashlib
import csv
import base64
import requests
import unicodedata
from datetime import date
from xml.sax.saxutils import escape

# Importações para o ReportLab (Geração do PDF)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuração da página do Streamlit
st.set_page_config(page_title="Conferência Fiscal", layout="wide", page_icon="📊")

st.title("Conferência fiscal")
st.write("Cruze as notas de entrada e NFC-e com o SPED do mês. O sistema confere os documentos e a IA ajuda a interpretar as diferenças.")

with st.container(border=True):
    st.subheader("Conectar Cloudflare Workers AI")
    st.caption(
        "A franquia gratuita é de até 10.000 neurônios por dia. O resumo e o print opcional serão enviados à Cloudflare. "
        "[Criar API Token](https://dash.cloudflare.com/profile/api-tokens)."
    )
    conta_coluna, token_coluna = st.columns(2)
    with conta_coluna:
        account_id_digitado = st.text_input(
            "Cloudflare Account ID",
            value=os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip(),
            placeholder="ID da conta Cloudflare",
            help="Encontre o Account ID no painel Cloudflare, na página inicial da conta.",
        )
    with token_coluna:
        token_digitado = st.text_input(
            "Cloudflare API Token",
            value=os.getenv("CLOUDFLARE_API_TOKEN", "").strip(),
            type="password",
            placeholder="Token com permissão Workers AI",
            help="Crie um token com acesso de execução ao Workers AI. Ele é usado apenas nesta sessão.",
        )
    st.markdown(
        "O aceite abaixo só é necessário se você quiser que a IA analise o print. A análise textual dos arquivos e divergências não depende desse aceite. Modelo visual: "
        "[licença Meta Llama 3.2](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) "
        "e da [política de uso](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)."
    )
    aceitou_licenca = st.checkbox("Aceito a licença para permitir análise de imagem pelo modelo Llama 3.2", value=False)
    st.caption("No primeiro teste, o app envia à Cloudflare a solicitação técnica necessária para ativar o modelo na conta.")

account_id = account_id_digitado.strip() or os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
api_token = token_digitado.strip() or os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
api_configurada = bool(account_id and api_token)
if api_configurada:
    st.success("Cloudflare configurado. O parecer será solicitado após a conferência local.")
else:
    st.info("Informe o Account ID e o token Cloudflare. O cruzamento local continua disponível sem a API.")

MODELO_VISAO_WORKERS_AI = "@cf/meta/llama-3.2-11b-vision-instruct"
MODELO_TEXTO_WORKERS_AI = "@cf/openai/gpt-oss-20b"
TIMEOUT_TESTE_API = (10, 60)
TIMEOUT_ANALISE_API = (10, 120)


def cabecalhos_cloudflare():
    return {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def ativar_modelo_workers_ai():
    if st.session_state.get("workers_ai_model_accepted"):
        return
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODELO_VISAO_WORKERS_AI}"
    resposta = requests.post(url, headers=cabecalhos_cloudflare(), json={"prompt": "agree"}, timeout=(10, 60))
    payload = resposta.json()
    erros = payload.get("errors", [])
    if any(
        erro.get("code") == 5016 and "thank you for agreeing" in erro.get("message", "").lower()
        for erro in erros
    ):
        st.session_state["workers_ai_model_accepted"] = True
        return
    if not resposta.ok or payload.get("success") is False:
        raise RuntimeError(str(erros or payload))
    st.session_state["workers_ai_model_accepted"] = True


def chamar_workers_ai(
    mensagens,
    max_tokens=1200,
    timeout=TIMEOUT_ANALISE_API,
    modelo=MODELO_TEXTO_WORKERS_AI,
):
    if modelo == MODELO_TEXTO_WORKERS_AI:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/responses"
        system_message = next((msg["content"] for msg in mensagens if msg.get("role") == "system"), "")
        user_message = next((msg["content"] for msg in mensagens if msg.get("role") == "user"), "")
        body = {
            "model": modelo,
            "instructions": system_message,
            "input": user_message,
            "max_output_tokens": max_tokens,
            "stream": False,
        }
    else:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
        body = {
            "model": modelo,
            "messages": mensagens,
            "max_tokens": max_tokens,
            "stream": False,
        }

    resposta = requests.post(
        url,
        headers=cabecalhos_cloudflare(),
        json=body,
        timeout=timeout,
    )
    payload = resposta.json()
    if not resposta.ok or payload.get("success") is False:
        raise RuntimeError(str(payload.get("errors") or payload))

    if modelo == MODELO_TEXTO_WORKERS_AI:
        saida_direta = payload.get("output_text")
        if isinstance(saida_direta, str) and saida_direta.strip():
            return saida_direta.strip()
        blocos_texto = []
        for item in payload.get("output", []):
            for bloco in item.get("content", []):
                if bloco.get("type") in {"output_text", "text"} and bloco.get("text"):
                    blocos_texto.append(bloco["text"])
        return "\n".join(blocos_texto).strip()

    escolhas = payload.get("choices", [])
    if not escolhas:
        raise RuntimeError("A Cloudflare não retornou uma resposta de chat.")
    conteudo = escolhas[0].get("message", {}).get("content", "")
    if isinstance(conteudo, list):
        conteudo = "\n".join(parte.get("text", "") for parte in conteudo if isinstance(parte, dict))
    return conteudo or ""


if st.button("Testar conexão com Cloudflare", disabled=not api_configurada):
    with st.spinner("Verificando a conexão; o modelo pode levar até 60 segundos para responder..."):
        try:
            teste = chamar_workers_ai(
                [{"role": "user", "content": "Responda apenas: conexão funcionando."}],
                max_tokens=32,
                timeout=TIMEOUT_TESTE_API,
            )
            st.success(teste or "A API respondeu sem texto; tente o botão de análise para obter o parecer.")
        except requests.Timeout:
            st.error("A Cloudflare não respondeu em 60 segundos. Tente novamente ou confira a conexão e o status do serviço.")
        except Exception as erro:
            st.error(f"A conexão com Cloudflare falhou: {erro}")

# --- FUNÇÃO GERADORA DE PDF ---
def gerar_pdf_contador(resumo_erros, parecer_ia):
    """
    Gera um arquivo PDF estruturado contendo as pendências
    e as recomendações analíticas para enviar ao contador.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    style_titulo = ParagraphStyle(
        'TituloLaudo',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    style_sub = ParagraphStyle(
        'SubTituloLaudo',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=8
    )
    
    style_texto = ParagraphStyle(
        'TextoLaudo',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748")
    )

    # 1. Cabeçalho do Documento
    story.append(Paragraph("LAUDO DE AUDITORIA DIGITAL - SPED FISCAL", style_titulo))
    story.append(Paragraph("<b>Destinatário:</b> Setor Contábil / Contador Responsável", style_texto))
    story.append(Paragraph("<b>Objetivo:</b> Conferência de documentos fiscais e divergências", style_texto))
    story.append(Spacer(1, 15))
    story.append(Paragraph("1. Resultado do cruzamento automático", style_sub))

    tabela_dados = [["Pendências e divergências"]]
    for linha in resumo_erros.split("\n"):
        if linha.strip():
            tabela_dados.append([Paragraph(escape(linha), style_texto)])
    tabela = Table(tabela_dados, colWidths=[doc.width])
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(tabela)
    story.append(Spacer(1, 15))
    story.append(Paragraph("2. Parecer da IA", style_sub))
    
    # Divide o parecer da IA em parágrafos separados para evitar LayoutError no ReportLab
    blocos = parecer_ia.split("\n\n") if "\n\n" in parecer_ia else parecer_ia.split("\n")
    for bloco in blocos:
        bloco_limpo = bloco.strip()
        if not bloco_limpo:
            continue
        if bloco_limpo.startswith(("1.", "2.", "3.", "4.", "5.", "#", "---")):
            story.append(Paragraph(escape(bloco_limpo), style_sub))
        else:
            story.append(Paragraph(escape(bloco_limpo).replace("\n", "<br/>"), style_texto))
        story.append(Spacer(1, 5))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- ARQUIVOS ---
st.subheader("1. Selecione o mês e os arquivos")
col_mes, col_ano = st.columns(2)
with col_mes:
    mes_referencia = st.selectbox(
        "Mês de referência",
        range(1, 13),
        index=date.today().month - 1,
        format_func=lambda mes: [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
        ][mes - 1],
    )
with col_ano:
    ano_referencia = st.number_input(
        "Ano de referência",
        min_value=2000,
        max_value=2100,
        value=date.today().year,
        step=1,
    )
periodo_referencia = f"{mes_referencia:02d}/{ano_referencia}"
st.caption("Use arquivos do mesmo mês. O período selecionado identifica a conferência e a análise da IA.")
st.caption("Envie os arquivos disponíveis. O SPED, as entradas e os cupons podem ser comparados em conjunto; nenhum campo precisa ser configurado manualmente.")
col1, col2, col3 = st.columns(3)
with col1:
    sped_file = st.file_uploader("SPED Fiscal", type=["txt"], key="sped", help="Arquivo fiscal do mês, com registros C100.")
with col2:
    entradas_file = st.file_uploader("Notas de entrada", type=["txt", "csv"], key="entradas", help="Exportação em texto ou CSV, separada por |.")
with col3:
    cupons_file = st.file_uploader("Cupons NFC-e", type=["txt", "csv"], key="cupons", help="Exportação de cupons em texto ou CSV, separada por |.")

with st.expander("Adicionar outras fontes ou contexto"):
    saidas_file = st.file_uploader("Notas de saída adicionais", type=["txt", "csv"], key="saidas")
    incluir_print = st.checkbox("Incluir imagem de pendências do mês anterior")
    print_file = st.file_uploader("Imagem do mês anterior", type=["png", "jpg", "jpeg"]) if incluir_print else None


def normalizar_nome(valor):
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    return "".join(caractere.lower() for caractere in texto if caractere.isalnum())


def normalizar_chave(valor):
    if pd.isna(valor) or valor is None:
        return ""
    digitos = "".join(caractere for caractere in str(valor) if caractere.isdigit())
    if len(digitos) >= 44:
        return digitos[-44:]
    return "".join(caractere for caractere in str(valor).strip().upper() if caractere.isalnum())


def converter_valor(valor):
    if pd.isna(valor) or valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip().replace("R$", "").replace(" ", "").replace("\xa0", "")
    if not texto:
        return None
    if "." in texto and "," in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        texto = texto.replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return None


def formatar_data(valor):
    if pd.isna(valor) or valor is None:
        return ""
    texto = str(valor).strip()
    if len(texto) == 8 and texto.isdigit():
        return f"{texto[:2]}/{texto[2:4]}/{texto[4:]}"
    if len(texto) >= 10 and "-" in texto[:10]:
        partes = texto[:10].split("-")
        if len(partes) == 3 and len(partes[0]) == 4:
            return f"{partes[2]}/{partes[1]}/{partes[0]}"
    return texto


def resumir_valores(tabela, coluna_valor, coluna_chave=None):
    if coluna_valor is None or tabela.empty:
        return {
            "total": None,
            "valores_validos": 0,
            "valores_nao_numericos": len(tabela),
            "documentos_unicos": 0,
            "tem_repetidos": False,
        }
    if coluna_chave and coluna_chave in tabela.columns:
        # Se houver itens repetidos por nota, calcula o total sobre as notas únicas
        tabela_docs = tabela.drop_duplicates(subset=[coluna_chave])
        valores = [converter_valor(valor) for valor in tabela_docs[coluna_valor]]
        valores_validos = [valor for valor in valores if valor is not None]
        return {
            "total": round(sum(valores_validos), 2),
            "valores_validos": len(valores_validos),
            "valores_nao_numericos": len(tabela_docs) - len(valores_validos),
            "documentos_unicos": len(tabela_docs),
            "tem_repetidos": len(tabela) > len(tabela_docs),
        }
    else:
        valores = [converter_valor(valor) for valor in tabela[coluna_valor]]
        valores_validos = [valor for valor in valores if valor is not None]
        return {
            "total": round(sum(valores_validos), 2),
            "valores_validos": len(valores_validos),
            "valores_nao_numericos": len(tabela) - len(valores_validos),
            "documentos_unicos": len(tabela),
            "tem_repetidos": False,
        }


def ler_arquivo(upload, nome):
    conteudo = upload.getvalue()
    texto = conteudo.decode("latin-1", errors="replace")

    if nome == "SPED":
        participantes = {}
        empresa = {"nome": "", "cnpj": ""}
        documentos = []
        for linha in texto.splitlines():
            campos_linha = [campo.strip() for campo in linha.split("|")]
            if len(campos_linha) > 2:
                reg = campos_linha[1]
                if reg == "0000" and len(campos_linha) > 7:
                    empresa["nome"] = campos_linha[6]
                    empresa["cnpj"] = campos_linha[7]
                elif reg == "0150" and len(campos_linha) > 5:
                    cod = campos_linha[2]
                    participantes[cod] = {
                        "nome": campos_linha[3],
                        "cnpj": campos_linha[5] if len(campos_linha) > 5 and campos_linha[5] else (campos_linha[6] if len(campos_linha) > 6 else ""),
                    }
                elif reg == "C100" and len(campos_linha) > 12 and campos_linha[9]:
                    ind_oper = campos_linha[2]
                    ind_emit = campos_linha[3]
                    cod_part = campos_linha[4]
                    part = participantes.get(cod_part, {})
                    emit_nome = empresa["nome"] if ind_emit == "0" else part.get("nome", "")
                    emit_cnpj = empresa["cnpj"] if ind_emit == "0" else part.get("cnpj", "")

                    sit_cod = campos_linha[6]
                    situacao_desc = "Regular" if sit_cod == "00" else ("Cancelado" if sit_cod in ("02", "03") else f"Situação {sit_cod}")

                    val_num = converter_valor(campos_linha[12])
                    val_str = f"{val_num:.2f}".replace(".", ",") if val_num is not None else campos_linha[12]

                    documentos.append({
                        "Chave": normalizar_chave(campos_linha[9]),
                        "Documento": campos_linha[8],
                        "Número": campos_linha[8],
                        "Série": campos_linha[7],
                        "Data Emissão": formatar_data(campos_linha[10]),
                        "Data de emissão": formatar_data(campos_linha[10]),
                        "Data Entrada": formatar_data(campos_linha[11]) if len(campos_linha) > 11 else "",
                        "Valor Total": val_str,
                        "Valor": val_str,
                        "Status": sit_cod,
                        "Situação": situacao_desc,
                        "Emitente/Fornecedor": emit_nome,
                        "CNPJ Emitente": emit_cnpj,
                        "Operação": "Entrada" if ind_oper == "0" else "Saída",
                        "Modelo": campos_linha[5],
                    })
        if documentos:
            return pd.DataFrame(documentos)
        raise ValueError("não encontrei registros C100 com chave de documento neste SPED")

    primeira_linha = texto.splitlines()[0] if texto.splitlines() else ""
    delimitador = "|" if "|" in primeira_linha else (";" if ";" in primeira_linha else ",")
    leitor = csv.reader(io.StringIO(texto), delimiter=delimitador)
    cabecalho = next(leitor, [])
    linhas = [linha for linha in leitor if linha]
    if not cabecalho:
        return pd.DataFrame()

    largura = max([len(cabecalho), *(len(linha) for linha in linhas)], default=len(cabecalho))
    colunas_originais = [str(coluna).replace("\ufeff", "").strip() for coluna in cabecalho]
    colunas_originais.extend(f"Campo_extra_{indice}" for indice in range(1, largura - len(colunas_originais) + 1))
    linhas = [linha + [""] * (largura - len(linha)) for linha in linhas]
    df = pd.DataFrame(linhas, columns=colunas_originais, dtype=str)

    # Identifica colunas-chave e padroniza para formato consistente
    col_map = {}
    for col in df.columns:
        norm = normalizar_nome(col)
        if ("chave" in norm or "accesskey" in norm) and "Chave" not in col_map:
            col_map["Chave"] = col
        elif norm in {"numero", "numeronota", "numerodocumento", "documento"} and "Documento" not in col_map:
            col_map["Documento"] = col
        elif norm in {"serie", "seriedocumento"} and "Série" not in col_map:
            col_map["Série"] = col
        elif ("dataemissao" in norm or norm in {"data", "dtdoc"}) and "Data Emissão" not in col_map:
            col_map["Data Emissão"] = col
        elif ("dataentrada" in norm or "dt_e_s" in norm or "dataentsai" in norm) and "Data Entrada" not in col_map:
            col_map["Data Entrada"] = col
        elif (norm in {"valor", "vldoc", "valordocumento"} or "valortotal" in norm) and "Valor Total" not in col_map:
            col_map["Valor Total"] = col
        elif ("razaosocialemit" in norm or "fornecedor" in norm or norm == "emitente") and "Emitente/Fornecedor" not in col_map:
            col_map["Emitente/Fornecedor"] = col
        elif ("cnpjemit" in norm or "cpfcnpjemit" in norm) and "CNPJ Emitente" not in col_map:
            col_map["CNPJ Emitente"] = col
        elif ("situacao" in norm or "status" in norm) and "Status" not in col_map:
            col_map["Status"] = col

    # Cria colunas padronizadas no início se as colunas originais forem encontradas
    colunas_padrao_criadas = {}
    if "Chave" in col_map:
        colunas_padrao_criadas["Chave"] = df[col_map["Chave"]].map(normalizar_chave)
    if "Documento" in col_map:
        colunas_padrao_criadas["Documento"] = df[col_map["Documento"]]
        colunas_padrao_criadas["Número"] = df[col_map["Documento"]]
    if "Série" in col_map:
        colunas_padrao_criadas["Série"] = df[col_map["Série"]]
    if "Data Emissão" in col_map:
        colunas_padrao_criadas["Data Emissão"] = df[col_map["Data Emissão"]].map(formatar_data)
        colunas_padrao_criadas["Data de emissão"] = df[col_map["Data Emissão"]].map(formatar_data)
    if "Data Entrada" in col_map:
        colunas_padrao_criadas["Data Entrada"] = df[col_map["Data Entrada"]].map(formatar_data)
    else:
        colunas_padrao_criadas["Data Entrada"] = ""
    if "Valor Total" in col_map:
        colunas_padrao_criadas["Valor Total"] = df[col_map["Valor Total"]]
        colunas_padrao_criadas["Valor"] = df[col_map["Valor Total"]]
    if "Status" in col_map:
        colunas_padrao_criadas["Status"] = df[col_map["Status"]]
        colunas_padrao_criadas["Situação"] = df[col_map["Status"]]
    if "Emitente/Fornecedor" in col_map:
        colunas_padrao_criadas["Emitente/Fornecedor"] = df[col_map["Emitente/Fornecedor"]]
    if "CNPJ Emitente" in col_map:
        colunas_padrao_criadas["CNPJ Emitente"] = df[col_map["CNPJ Emitente"]]

    if colunas_padrao_criadas:
        df_padrao = pd.DataFrame(colunas_padrao_criadas)
        demais_colunas = [c for c in df.columns if c not in col_map.values() and c not in df_padrao.columns]
        df = pd.concat([df_padrao, df[demais_colunas]], axis=1)

    return df


uploads = {
    "SPED": sped_file,
    "Entradas": entradas_file,
    "Saídas": saidas_file,
    "Cupons": cupons_file,
}
tabelas = {}
for nome, arquivo in uploads.items():
    if arquivo is not None:
        try:
            tabelas[nome] = ler_arquivo(arquivo, nome)
        except Exception as erro:
            st.error(f"Não foi possível ler {nome}: {erro}")

campos = {}
for nome, tabela in tabelas.items():
    st.caption(f"{nome}: {len(tabela):,} linhas carregadas".replace(",", "."))
    with st.expander(f"Visualizar notas importadas: {nome} ({len(tabela):,})".replace(",", ".")):
        st.dataframe(tabela, use_container_width=True, hide_index=True, height=420)
    opcoes = [None, *tabela.columns.tolist()]
    chave_sugerida = next(
        (coluna for coluna in tabela.columns if "chave" in normalizar_nome(coluna) or "accesskey" in normalizar_nome(coluna)),
        None,
    )
    valor_sugerido = next(
        (
            coluna for coluna in tabela.columns
            if normalizar_nome(coluna) in {"valor", "vldoc", "valordocumento"}
            or "valortotal" in normalizar_nome(coluna)
        ),
        None,
    )
    chave_indice = opcoes.index(chave_sugerida) if chave_sugerida in opcoes else 0
    valor_indice = opcoes.index(valor_sugerido) if valor_sugerido in opcoes else 0
    with st.expander(f"Conferir campos detectados: {nome}"):
        st.dataframe(tabela.head(5), use_container_width=True, hide_index=True)
        chave = st.selectbox(
            f"Campo identificador de {nome} (opcional)",
            opcoes,
            index=chave_indice,
            format_func=lambda opcao: "Não usar" if opcao is None else str(opcao),
            key=f"chave_{nome}",
        )
        valor = st.selectbox(
            f"Campo de valor de {nome} (opcional)",
            opcoes,
            index=valor_indice,
            format_func=lambda opcao: "Não usar" if opcao is None else str(opcao),
            key=f"valor_{nome}",
        )
    campos[nome] = {"chave": chave, "valor": valor}


def formatar_moeda_br(valor, texto_vazio="Não calculado"):
    if valor is None or pd.isna(valor):
        return texto_vazio
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metadados_documento(linha):
    metadados = {}
    for coluna, valor in linha.items():
        nome = normalizar_nome(coluna)
        texto = "" if pd.isna(valor) else str(valor).strip()
        if not texto:
            continue
        if nome in {"documento", "numero", "numeronota", "numerodocumento"}:
            metadados.setdefault("Documento", texto)
            metadados.setdefault("Número", texto)
        elif "serie" in nome:
            metadados.setdefault("Série", texto)
        elif "dataemissao" in nome or nome in {"data", "dtdoc"}:
            metadados.setdefault("Data Emissão", texto)
            metadados.setdefault("Data", texto)
        elif "dataentrada" in nome or "dt_e_s" in nome or "dataentsai" in nome:
            metadados.setdefault("Data Entrada", texto)
        elif "razaosocialemit" in nome or "fornecedor" in nome or nome == "emitente":
            metadados.setdefault("Emitente/Fornecedor", texto)
        elif "cnpjemit" in nome or "cpfcnpjemit" in nome:
            metadados.setdefault("CNPJ Emitente", texto)
        elif "situacao" in nome or "status" in nome:
            metadados.setdefault("Status", texto)
            metadados.setdefault("Situação", texto)
    return metadados


def comparar_arquivos(tabelas, campos):
    elegiveis = [nome for nome in tabelas if campos[nome]["chave"] is not None]
    if len(elegiveis) < 2:
        return {
            "referencia": None,
            "mensagem": "Envie pelo menos dois arquivos com uma chave de documento detectada para fazer o cruzamento automático.",
            "comparacoes": [],
            "detalhes": [],
            "valores_documentos": [],
        }

    referencia = "SPED" if "SPED" in elegiveis else elegiveis[0]
    comparacoes = []
    detalhes = []
    valores_documentos = []
    for nome in elegiveis:
        if nome == referencia:
            continue

        tabela_referencia = tabelas[referencia]
        if referencia == "SPED" and "Operação" in tabela_referencia.columns:
            if nome == "Entradas":
                tabela_referencia = tabela_referencia[tabela_referencia["Operação"] == "Entrada"]
            elif nome == "Cupons":
                tabela_referencia = tabela_referencia[
                    (tabela_referencia["Operação"] == "Saída") & (tabela_referencia["Modelo"] == "65")
                ]
            elif nome == "Saídas":
                tabela_referencia = tabela_referencia[
                    (tabela_referencia["Operação"] == "Saída") & (tabela_referencia["Modelo"] != "65")
                ]

        coluna_referencia = campos[referencia]["chave"]
        coluna_chave = campos[nome]["chave"]
        coluna_valor = campos[nome]["valor"]
        valor_referencia = campos[referencia]["valor"]
        indice_referencia = {}
        for linha in tabela_referencia.to_dict(orient="records"):
            chave = normalizar_chave(linha.get(coluna_referencia))
            if chave and chave not in indice_referencia:
                indice_referencia[chave] = linha

        indice_arquivo = {}
        for linha in tabelas[nome].to_dict(orient="records"):
            chave = normalizar_chave(linha.get(coluna_chave))
            if chave and chave not in indice_arquivo:
                indice_arquivo[chave] = linha

        ausentes_na_referencia = set(indice_arquivo) - set(indice_referencia)
        ausentes_no_arquivo = set(indice_referencia) - set(indice_arquivo)
        divergentes = 0
        detalhes_comparacao = []
        resumo_arquivo = resumir_valores(tabelas[nome], coluna_valor, coluna_chave)
        resumo_referencia = resumir_valores(tabela_referencia, valor_referencia, coluna_referencia)
        diferenca_total = (
            round(resumo_arquivo["total"] - resumo_referencia["total"], 2)
            if resumo_arquivo["total"] is not None and resumo_referencia["total"] is not None
            else None
        )
        for chave in sorted(ausentes_na_referencia):
            linha_arq = indice_arquivo[chave]
            valor_arquivo = (
                converter_valor(linha_arq.get(coluna_valor))
                if coluna_valor is not None
                else None
            )
            meta = metadados_documento(linha_arq)
            detalhe = {
                "Fonte": nome,
                "Categoria": f"Faltando no {referencia} (Só no Arquivo)",
                "Situação": f"Só em {nome}",
                "Chave": chave,
                "Documento": meta.get("Documento") or meta.get("Número") or "—",
                "Data Emissão": meta.get("Data Emissão") or meta.get("Data") or "—",
                "Data Entrada": meta.get("Data Entrada") or "—",
                "Valor Total": round(valor_arquivo, 2) if valor_arquivo is not None else None,
                "Valor no arquivo": round(valor_arquivo, 2) if valor_arquivo is not None else None,
                "Valor no SPED": None,
                "Diferença (arquivo - SPED)": None,
                "Status": meta.get("Status") or "00",
                "Emitente/Fornecedor": meta.get("Emitente/Fornecedor") or "—",
            }
            detalhe.update(meta)
            detalhes_comparacao.append(detalhe)

        for chave in sorted(ausentes_no_arquivo):
            linha_ref = indice_referencia[chave]
            valor_sped = (
                converter_valor(linha_ref.get(valor_referencia))
                if valor_referencia is not None
                else None
            )
            meta = metadados_documento(linha_ref)
            detalhe = {
                "Fonte": nome,
                "Categoria": f"Faltando no ATF / ERP (Só no {referencia})",
                "Situação": f"Só em {referencia}",
                "Chave": chave,
                "Documento": meta.get("Documento") or meta.get("Número") or "—",
                "Data Emissão": meta.get("Data Emissão") or meta.get("Data") or "—",
                "Data Entrada": meta.get("Data Entrada") or "—",
                "Valor Total": round(valor_sped, 2) if valor_sped is not None else None,
                "Valor no arquivo": None,
                "Valor no SPED": round(valor_sped, 2) if valor_sped is not None else None,
                "Diferença (arquivo - SPED)": None,
                "Status": meta.get("Status") or "00",
                "Emitente/Fornecedor": meta.get("Emitente/Fornecedor") or "—",
            }
            detalhe.update(meta)
            detalhes_comparacao.append(detalhe)

        if coluna_valor is not None and valor_referencia is not None:
            for chave in sorted(set(indice_arquivo).intersection(indice_referencia)):
                linha_arq = indice_arquivo[chave]
                linha_ref = indice_referencia[chave]
                valor_arquivo = converter_valor(linha_arq.get(coluna_valor))
                valor_sped = converter_valor(linha_ref.get(valor_referencia))
                diferenca = (
                    round(valor_arquivo - valor_sped, 2)
                    if valor_arquivo is not None and valor_sped is not None
                    else None
                )
                meta_arq = metadados_documento(linha_arq)
                meta_ref = metadados_documento(linha_ref)
                is_divergente = diferenca is not None and abs(diferenca) > 0.05
                sit_label = "Valor divergente" if is_divergente else ("Valores iguais" if diferenca is not None else "Valor não disponível")

                documento = {
                    "Fonte": nome,
                    "Categoria": "Divergência de Valor" if is_divergente else "OK",
                    "Situação": sit_label,
                    "Chave": chave,
                    "Documento": meta_arq.get("Documento") or meta_ref.get("Documento") or "—",
                    "Data Emissão": meta_arq.get("Data Emissão") or meta_ref.get("Data Emissão") or meta_arq.get("Data") or "—",
                    "Data Entrada": meta_ref.get("Data Entrada") or meta_arq.get("Data Entrada") or "—",
                    "Valor Total": round(valor_arquivo, 2) if valor_arquivo is not None else round(valor_sped, 2),
                    "Valor no arquivo": round(valor_arquivo, 2) if valor_arquivo is not None else None,
                    "Valor no SPED": round(valor_sped, 2) if valor_sped is not None else None,
                    "Diferença (arquivo - SPED)": diferenca,
                    "Status": meta_arq.get("Status") or meta_ref.get("Status") or "00",
                    "Emitente/Fornecedor": meta_arq.get("Emitente/Fornecedor") or meta_ref.get("Emitente/Fornecedor") or "—",
                }
                valores_documentos.append(documento)
                if is_divergente:
                    divergentes += 1
                    detalhes_comparacao.append(documento)

        comparacoes.append({
            "Fonte": nome,
            "Documentos únicos": len(indice_arquivo),
            "Documentos no SPED": len(indice_referencia),
            "Só na fonte": len(ausentes_na_referencia),
            "Só na referência": len(ausentes_no_arquivo),
            "Valores divergentes": divergentes,
            "Total no arquivo": resumo_arquivo["total"],
            "Total no SPED": resumo_referencia["total"],
            "Diferença total (arquivo - SPED)": diferenca_total,
        })
        detalhes.extend(detalhes_comparacao)

    return {
        "referencia": referencia,
        "mensagem": None,
        "comparacoes": comparacoes,
        "detalhes": detalhes,
        "valores_documentos": valores_documentos,
    }


def formatar_resumo(resultado):
    if resultado["mensagem"]:
        return resultado["mensagem"]
    linhas = [f"Base de referência: {resultado['referencia']}."]
    for comparacao in resultado["comparacoes"]:
        valores = []
        for chave, valor in comparacao.items():
            if chave.startswith("Total") or chave.startswith("Diferença total"):
                valor = formatar_moeda_br(valor)
            elif valor is None:
                valor = "não calculado"
            valores.append(f"{chave}: {valor}")
        linhas.append(", ".join(valores))
    if resultado["detalhes"]:
        total_detalhes = len(resultado["detalhes"])
        linhas.append(f"Detalhes encontrados (total de {total_detalhes}):")
        limite_ia = 60
        for detalhe in resultado["detalhes"][:limite_ia]:
            valores = []
            for chave, valor in detalhe.items():
                if chave in {"Valor no arquivo", "Valor no SPED", "Diferença (arquivo - SPED)"}:
                    valor = formatar_moeda_br(valor, "não disponível")
                valores.append(f"{chave}: {valor}")
            linhas.append("; ".join(valores))
        if total_detalhes > limite_ia:
            linhas.append(f"... e mais {total_detalhes - limite_ia} divergências disponíveis na tabela do relatório.")
    else:
        linhas.append("Nenhuma divergência encontrada nos documentos e valores comparados.")
    return "\n".join(linhas)


def gerar_parecer_ia(resumo, imagem=None, analise_geral=False):
    if analise_geral:
        instrucao = (
            "Responda em português do Brasil para a pessoa responsável por gerar o SPED e encaminhar os arquivos ao contador. "
            "O relatório contém comparações independentes das fontes fiscais importadas, todas do mesmo mês de referência. "
            "Interprete os resultados em conjunto: identifique padrões entre fontes, diferenças que se repetem e prioridades, "
            "sem somar documentos que possam aparecer em mais de uma comparação. O mês informado é o período analisado; "
            "não conclua tendências entre meses sem dados de outros períodos.\n\n"
            "Organize o parecer nestas seções:\n"
            "1. Conclusão geral do mês e se os arquivos parecem prontos para envio.\n"
            "2. Interpretação por comparação e padrões observados entre as fontes.\n"
            "3. O que revisar primeiro, com passos verificáveis para conferir período, operação, chave, modelo, total da nota e escrituração.\n"
            "4. O que encaminhar ao contador e quais decisões precisam de validação.\n"
            "5. Limitações: diferencie fatos calculados de hipóteses; não invente causa, imposto, CFOP ou risco jurídico. O parecer não substitui a validação do contador.\n\n"
            "Use todos os resultados fornecidos, não os trate como amostra. Se houver print, interprete-o apenas como contexto adicional."
        )
    else:
        instrucao = (
            "Responda em português do Brasil para a pessoa responsável por gerar o SPED e encaminhar os arquivos ao contador. "
            "O cruzamento local comparou todos os documentos identificáveis dos arquivos enviados e listou TODOS os achados abaixo. "
            "Use essa lista integral como base; não a trate como amostra. Faça uma interpretação geral e prática, sem apenas repetir a tabela.\n\n"
            "Organize o parecer nestas seções:\n"
            "1. Conclusão geral: diga em linguagem direta se os arquivos parecem prontos para envio ou se há pontos a revisar.\n"
            "2. O que foi encontrado: explique cada tipo de divergência, ausência em cada fonte e diferença de valor; priorize o que pode afetar o SPED.\n"
            "3. O que eu devo fazer antes de enviar: passos verificáveis no sistema que gerou o SPED, como conferir período, operação, chave, modelo, total da nota e escrituração; não mande alterar dados sem validar o documento original.\n"
            "4. O que encaminhar ao contador: lista objetiva dos arquivos, documentos/valores divergentes e perguntas que precisam de decisão contábil.\n"
            "5. Limitações: diferencie fatos calculados pelo comparador de hipóteses; não invente causa, imposto, CFOP ou risco jurídico que os dados não provem. O parecer é apoio e não substitui a validação do contador.\n\n"
            "A lista de achados local é completa para as chaves e campos de valor comparados. Se houver print, interprete-o como contexto adicional e aponte qualquer informação visual que não coincida com os dados estruturados."
        )
    prompt = f"{instrucao}\n\nRELATÓRIO COMPLETO DO CRUZAMENTO LOCAL:\n{resumo}"
    mensagens_texto = [
        {
            "role": "system",
            "content": "Você é um analista de pré-conferência fiscal, cuidadoso, objetivo e prático. Não invente dados nem dê parecer jurídico definitivo.",
        },
        {"role": "user", "content": prompt},
    ]

    if imagem is None:
        parecer = chamar_workers_ai(mensagens_texto, modelo=MODELO_TEXTO_WORKERS_AI, max_tokens=2200)
        return parecer or "A API respondeu sem texto. O cruzamento local continua disponível; tente gerar o parecer novamente."

    ativar_modelo_workers_ai()
    mensagens_visao = [
        mensagens_texto[0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{imagem.type or 'image/jpeg'};base64,{base64.b64encode(imagem.getvalue()).decode('ascii')}"
                    },
                },
            ],
        },
    ]
    try:
        parecer = chamar_workers_ai(mensagens_visao, modelo=MODELO_VISAO_WORKERS_AI, max_tokens=2200)
        if parecer:
            return parecer
    except requests.Timeout:
        pass
    except RuntimeError as erro:
        if "3030" not in str(erro):
            raise

    parecer = chamar_workers_ai(mensagens_texto, modelo=MODELO_TEXTO_WORKERS_AI, max_tokens=2200)
    if not parecer:
        return "A API não retornou texto, nem no modelo de visão nem no modelo textual. O cruzamento local foi concluído; tente novamente mais tarde."
    return "A análise textual completa foi concluída. A Cloudflare não conseguiu interpretar o print nesta tentativa.\n\n" + parecer


def gerar_resumo_consolidado(resultados, periodo):
    linhas = [f"Mês de referência: {periodo}.", "Comparações independentes com o SPED:"]
    for fonte in ("Entradas", "Saídas", "Cupons"):
        if fonte in resultados:
            linhas.append(f"\n--- {fonte} x SPED ---")
            linhas.append(formatar_resumo(resultados[fonte]))
    return "\n".join(linhas)


def criar_contexto_comparacao(periodo, uploads, campos):
    partes = [periodo]
    for nome, arquivo in uploads.items():
        if arquivo is not None:
            conteudo = hashlib.sha256(arquivo.getvalue()).hexdigest()
            partes.append(f"{nome}:{arquivo.name}:{conteudo}")
    for nome, selecao in sorted(campos.items()):
        partes.append(f"{nome}:{selecao['chave']}:{selecao['valor']}")
    return hashlib.sha256("\n".join(partes).encode("utf-8")).hexdigest()


def executar_analise_ia(resumo, imagem=None, analise_geral=False):
    try:
        return gerar_parecer_ia(resumo, imagem, analise_geral)
    except requests.Timeout:
        return "A Cloudflare não respondeu em 120 segundos. O cruzamento local foi concluído; tente gerar o parecer novamente."
    except Exception as erro:
        return f"Não foi possível obter o parecer da Cloudflare: {erro}"


def exibir_comparacao(resultado):
    if resultado["mensagem"]:
        st.info(resultado["mensagem"])
        return
    st.caption(f"Base de referência: {resultado['referencia']}")
    tabela_comparacoes = pd.DataFrame(resultado["comparacoes"])
    for coluna in ("Total no arquivo", "Total no SPED", "Diferença total (arquivo - SPED)"):
        if coluna in tabela_comparacoes:
            tabela_comparacoes[coluna] = tabela_comparacoes[coluna].map(formatar_moeda_br)
    st.dataframe(tabela_comparacoes, use_container_width=True, hide_index=True)

    detalhes = resultado.get("detalhes", [])
    valores_docs = resultado.get("valores_documentos", [])

    so_sped = [
        d for d in detalhes
        if "Faltando no ATF" in d.get("Categoria", "") or "Só em " + str(resultado["referencia"]) in d.get("Situação", "")
    ]
    so_arquivo = [
        d for d in detalhes
        if "Faltando no SPED" in d.get("Categoria", "") or ("Só em " in d.get("Situação", "") and d not in so_sped)
    ]
    divergentes = [
        d for d in detalhes
        if d.get("Categoria") == "Divergência de Valor" or d.get("Situação") == "Valor divergente"
    ]
    docs_ok = [
        d for d in valores_docs
        if d.get("Categoria") == "OK" or d.get("Situação") == "Valores iguais"
    ]

    aba_erros, aba_ok, aba_todos = st.tabs(["🔴 ERROS / PENDÊNCIAS", "🟢 OK (BATIDOS)", "📋 TODOS"])

    with aba_erros:
        if not detalhes:
            st.success("Os documentos e valores conferidos estão 100% consistentes. Nenhuma pendência encontrada!")
        else:
            opcoes_erros = []
            if so_sped:
                opcoes_erros.append(f"FALTANDO NO ATF / ERP (Só no SPED) ({len(so_sped)})")
            if so_arquivo:
                opcoes_erros.append(f"FALTANDO NO SPED (Só no Arquivo) ({len(so_arquivo)})")
            if divergentes:
                opcoes_erros.append(f"Divergência de Valor ({len(divergentes)})")
            opcoes_erros.append(f"Todas as Pendências ({len(detalhes)})")

            escolha = st.radio("Filtrar grupo de pendências:", opcoes_erros, horizontal=True)

            if "FALTANDO NO ATF" in escolha:
                lista_grupo = so_sped
            elif "FALTANDO NO SPED" in escolha:
                lista_grupo = so_arquivo
            elif "Divergência" in escolha:
                lista_grupo = divergentes
            else:
                lista_grupo = detalhes

            total_grupo = sum(
                converter_valor(item.get("Valor Total") or item.get("Valor no SPED") or item.get("Valor no arquivo")) or 0.0
                for item in lista_grupo
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Quantidade", f"{len(lista_grupo)} notas")
            with col2:
                st.metric("Total", formatar_moeda_br(total_grupo))

            df_grupo = pd.DataFrame(lista_grupo)
            cols_desejadas = [
                "Chave", "Documento", "Data Emissão", "Data Entrada",
                "Valor Total", "Valor no arquivo", "Valor no SPED", "Diferença (arquivo - SPED)",
                "Status", "Emitente/Fornecedor", "Situação"
            ]
            cols_finais = [c for c in cols_desejadas if c in df_grupo.columns]
            df_render = df_grupo[cols_finais].copy()
            for col_moeda in ("Valor Total", "Valor no arquivo", "Valor no SPED", "Diferença (arquivo - SPED)"):
                if col_moeda in df_render.columns:
                    df_render[col_moeda] = df_render[col_moeda].map(lambda v: formatar_moeda_br(v, "—"))
            st.dataframe(df_render, use_container_width=True, hide_index=True)

    with aba_ok:
        if not docs_ok:
            st.info("Nenhum documento com valores exatamente iguais encontrado.")
        else:
            total_ok = sum(converter_valor(item.get("Valor Total")) or 0.0 for item in docs_ok)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Quantidade", f"{len(docs_ok)} notas")
            with col2:
                st.metric("Total", formatar_moeda_br(total_ok))

            df_ok = pd.DataFrame(docs_ok)
            cols_desejadas = [
                "Chave", "Documento", "Data Emissão", "Data Entrada",
                "Valor Total", "Status", "Emitente/Fornecedor"
            ]
            cols_finais = [c for c in cols_desejadas if c in df_ok.columns]
            df_render_ok = df_ok[cols_finais].copy()
            if "Valor Total" in df_render_ok.columns:
                df_render_ok["Valor Total"] = df_render_ok["Valor Total"].map(formatar_moeda_br)
            st.dataframe(df_render_ok, use_container_width=True, hide_index=True)

    with aba_todos:
        todos_itens = list(valores_docs) + [d for d in detalhes if d not in valores_docs]
        if todos_itens:
            df_todos = pd.DataFrame(todos_itens)
            cols_desejadas = [
                "Chave", "Documento", "Data Emissão", "Data Entrada",
                "Valor Total", "Situação", "Status", "Emitente/Fornecedor"
            ]
            cols_finais = [c for c in cols_desejadas if c in df_todos.columns]
            df_render_todos = df_todos[cols_finais].copy()
            if "Valor Total" in df_render_todos.columns:
                df_render_todos["Valor Total"] = df_render_todos["Valor Total"].map(formatar_moeda_br)
            st.dataframe(df_render_todos, use_container_width=True, hide_index=True, height=420)


def exibir_totais_importados(tabelas, campos):
    linhas = []
    for nome, tabela in tabelas.items():
        chave_col = campos.get(nome, {}).get("chave")
        resumo = resumir_valores(tabela, campos[nome]["valor"], chave_col)
        linhas.append({
            "Arquivo": nome,
            "Registros (linhas)": len(tabela),
            "Notas únicas": resumo.get("documentos_unicos", len(tabela)),
            "Campo de valor": campos[nome]["valor"] or "Não selecionado",
            "Total das notas": formatar_moeda_br(resumo["total"]),
            "Valores válidos": resumo["valores_validos"],
            "Sem valor numérico": resumo["valores_nao_numericos"],
        })
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
    st.caption(
        "Os totais são consolidados pelas notas únicas (primeira ocorrência de cada chave) "
        "para garantir a conferência fiscal correta, mesmo em arquivos detalhados por itens."
    )


contexto_comparacao = criar_contexto_comparacao(periodo_referencia, uploads, campos)
if st.session_state.get("contexto_comparacao") != contexto_comparacao:
    st.session_state["contexto_comparacao"] = contexto_comparacao
    st.session_state["comparacoes_mes"] = {}
    st.session_state["pareceres_comparacoes"] = {}
    st.session_state["parecer_geral_mes"] = None

st.subheader("2. Totais das notas importadas")
if tabelas:
    exibir_totais_importados(tabelas, campos)
else:
    st.info("Os totais aparecerão após importar os arquivos e selecionar os campos de valor.")

st.subheader("3. Comparações por tipo")
st.caption(f"Período selecionado: {periodo_referencia}. Cada aba compara a fonte selecionada com o SPED.")
nomes_fontes = ("Entradas", "Saídas", "Cupons")
rotulos_fontes = {"Entradas": "Entrada", "Saídas": "Saída", "Cupons": "Cupom"}
rotulos_abas = ("Comparar Entrada", "Comparar Saída", "Comparar Cupom")
abas = st.tabs(rotulos_abas)

for aba, nome in zip(abas, nomes_fontes):
    with aba:
        pronto_para_comparar = (
            "SPED" in tabelas
            and nome in tabelas
            and campos["SPED"]["chave"] is not None
            and campos[nome]["chave"] is not None
        )
        if "SPED" not in tabelas:
            st.info("Importe o SPED do mês para comparar esta fonte.")
        elif nome not in tabelas:
            st.info(f"Importe o arquivo de {nome.lower()} para habilitar esta comparação.")
        elif not pronto_para_comparar:
            st.info(f"Selecione as colunas de chave do SPED e de {nome} em “Conferir campos detectados”.")

        if st.button(
            f"Comparar {rotulos_fontes[nome]} e analisar com IA",
            type="primary",
            disabled=not pronto_para_comparar,
            key=f"comparar_{normalizar_nome(nome)}",
        ):
            resultado = comparar_arquivos(
                {"SPED": tabelas["SPED"], nome: tabelas[nome]},
                {"SPED": campos["SPED"], nome: campos[nome]},
            )
            st.session_state["comparacoes_mes"][nome] = resultado
            st.session_state["pareceres_comparacoes"].pop(nome, None)
            st.session_state["parecer_geral_mes"] = None
            if not resultado["mensagem"] and api_configurada:
                resumo_individual = f"Mês de referência: {periodo_referencia}.\n{formatar_resumo(resultado)}"
                with st.spinner("A IA está analisando esta comparação..."):
                    st.session_state["pareceres_comparacoes"][nome] = executar_analise_ia(resumo_individual)

        resultado_atual = st.session_state["comparacoes_mes"].get(nome)
        if resultado_atual is not None:
            exibir_comparacao(resultado_atual)
            if not resultado_atual["mensagem"]:
                parecer_individual = st.session_state["pareceres_comparacoes"].get(nome)
                if parecer_individual is None and api_configurada:
                    if st.button("Gerar análise da IA", key=f"analisar_{normalizar_nome(nome)}"):
                        resumo_individual = f"Mês de referência: {periodo_referencia}.\n{formatar_resumo(resultado_atual)}"
                        with st.spinner("A IA está analisando esta comparação..."):
                            parecer_individual = executar_analise_ia(resumo_individual)
                            st.session_state["pareceres_comparacoes"][nome] = parecer_individual
                elif parecer_individual is None:
                    st.warning("A comparação local foi concluída. Configure a API Cloudflare para gerar a análise da IA.")

                if parecer_individual:
                    st.subheader("Análise da IA")
                    if parecer_individual.startswith(("Não foi possível", "A Cloudflare não respondeu")):
                        st.error(parecer_individual)
                    else:
                        st.write(parecer_individual)

fontes_carregadas = [nome for nome in nomes_fontes if nome in tabelas]
comparacoes_mes = st.session_state["comparacoes_mes"]
fontes_concluidas = [
    nome for nome in fontes_carregadas
    if nome in comparacoes_mes and not comparacoes_mes[nome]["mensagem"]
]
todas_comparacoes_concluidas = bool(fontes_carregadas) and len(fontes_concluidas) == len(fontes_carregadas)

st.subheader("4. Interpretação geral do mês")
if not fontes_carregadas:
    st.info("Importe ao menos uma fonte (entrada, saída ou cupom) junto com o SPED.")
elif not todas_comparacoes_concluidas:
    faltantes = [nome for nome in fontes_carregadas if nome not in fontes_concluidas]
    st.info(f"Conclua as comparações disponíveis antes da análise geral: {', '.join(faltantes)}.")
elif not api_configurada:
    st.warning("Todas as comparações locais foram concluídas. Configure a API Cloudflare para gerar a interpretação geral com IA.")

if todas_comparacoes_concluidas:
    resumo_consolidado = gerar_resumo_consolidado(comparacoes_mes, periodo_referencia)
    imagem_para_ia = print_file if aceitou_licenca else None
    if print_file is not None and not aceitou_licenca:
        st.warning("A análise geral será textual; o print não será enviado porque a licença do modelo visual não foi aceita.")

    if st.button(
        "Analisar todas as comparações com IA",
        type="primary",
        disabled=not api_configurada,
        key="analisar_geral_mes",
    ):
        with st.spinner("A IA está interpretando em conjunto as comparações do mês..."):
            st.session_state["parecer_geral_mes"] = executar_analise_ia(
                resumo_consolidado,
                imagem_para_ia,
                analise_geral=True,
            )

    parecer_geral = st.session_state.get("parecer_geral_mes")
    if parecer_geral:
        st.subheader("Análise geral da IA")
        if parecer_geral.startswith(("Não foi possível", "A Cloudflare não respondeu")):
            st.error(parecer_geral)
        else:
            st.write(parecer_geral)
    elif api_configurada:
        st.caption("Clique em “Analisar todas as comparações com IA” para gerar a interpretação geral.")
    else:
        parecer_geral = "Parecer não gerado: configure o Account ID e o token da Cloudflare para habilitar a análise por IA."

    try:
        pdf = gerar_pdf_contador(resumo_consolidado, parecer_geral or "Análise geral da IA ainda não solicitada.")
        st.download_button(
            "Baixar laudo consolidado em PDF",
            data=pdf,
            file_name=f"laudo_auditoria_{ano_referencia}_{mes_referencia:02d}.pdf",
            mime="application/pdf",
            key="baixar_laudo_consolidado",
        )
    except Exception as erro_pdf:
        st.warning(f"Não foi possível gerar o PDF consolidado: {erro_pdf}")