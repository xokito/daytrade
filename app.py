import streamlit as st
import pandas as pd
import plotly.express as px
import locale
import numpy as np
import re
import unicodedata # Para normalização de caracteres (remover acentos)

st.set_page_config(layout="wide", page_title="Controle de Trades")
# Configura o locale para formato numérico brasileiro
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    st.warning("Não foi possível configurar o locale 'pt_BR.UTF-8'. As formatações numéricas podem não ser as esperadas.")
    st.warning("Em alguns sistemas (ex: Windows), pode ser necessário usar 'Portuguese_Brazil.1252' ou instalar o pacote de idioma.")




st.title("📊 Controle de Trades Day Trade")

st.sidebar.write(
    "Faça o upload do seu relatório de operações do Nelogica (.csv) e visualize sua performance."
)

# --- Upload do Arquivo ---
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=";",
            decimal=",",
            thousands=".",
            encoding="latin-1",
            on_bad_lines="skip",
            skiprows=5
        )

        # --- Limpeza e Padronização dos Nomes das Colunas ---
        def clean_column_names(df):
            new_columns = []
            for col in df.columns:
                # Normaliza para remover acentos e caracteres especiais, depois remove caracteres não-ASCII
                normalized_col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8')
                
                # Mapeamento explícito para colunas conhecidas do Nelogica
                if normalized_col == 'Res. Operacao':
                    new_columns.append('res_operacao')
                elif normalized_col == 'Qtd Compra':
                    new_columns.append('qtd_compra')
                elif normalized_col == 'Qtd Venda':
                    new_columns.append('qtd_venda')
                elif normalized_col == 'Tempo Operacao':
                    new_columns.append('tempo_operacao')
                elif normalized_col == 'Preco Compra':
                    new_columns.append('preco_compra')
                elif normalized_col == 'Preco Venda':
                    new_columns.append('preco_venda')
                elif normalized_col == 'Preco de Mercado':
                    new_columns.append('preco_de_mercado')
                elif normalized_col == 'Res. Intervalo':
                    new_columns.append('res_intervalo')
                elif normalized_col == 'Res. Intervalo (%)':
                    new_columns.append('res_intervalo_perc')
                elif normalized_col == 'Res. Operacao (%)':
                    new_columns.append('res_operacao_perc')
                else:
                    # Limpeza genérica para outras colunas: remove caracteres não alfanuméricos,
                    # substitui espaços por underscores e converte para minúsculas
                    cleaned_col = re.sub(r'[^\w\s]', '', normalized_col)
                    cleaned_col = cleaned_col.replace(' ', '_')
                    cleaned_col = cleaned_col.lower()
                    new_columns.append(cleaned_col)
            df.columns = new_columns
            return df

        df = clean_column_names(df)



        # Verifica se as colunas essenciais existem após a limpeza
        required_columns = ['ativo', 'res_operacao', 'qtd_compra', 'qtd_venda']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"Não foi possível encontrar as colunas essenciais: {', '.join(missing_cols)}. Verifique o cabeçalho do seu CSV após as 5 linhas puladas.")
            st.info("Por favor, verifique a barra lateral à esquerda para ver os nomes exatos das colunas após o processamento.")
            st.stop()

        # --- Exibir/Esconder a Planilha (primeiras linhas) ---
        #with st.expander("Mostrar/Esconder Dados da Planilha (Primeiras 5 linhas)"):
        #    st.dataframe(df.head())
        #    st.info("Para ver a planilha completa, desmarque o 'Mostrar/Esconder Dados da Planilha' na barra lateral.")

        # --- Entrada do Capital Inicial (Continua como input) ---
        st.sidebar.header("Configurações da Carteira")
        initial_capital = st.sidebar.number_input(
            "Capital Inicial (R$)",
            min_value=0.0,
            value=3730.0,
            step=100.0,
            format="%.2f",
        )
        ## função para tratar negativos 
    
 
        def colored_metric(label, value, dark_theme=True):
            """
            Exibe uma métrica com cor condicional, compatível com tema claro e escuro.

            Args:
                label (str): O rótulo da métrica.
                value (float or int): O valor numérico da métrica.
                dark_theme (bool): Se True, usa a paleta de cores para tema escuro.
            """
            
            # --- Define a paleta de cores baseada no tema ---
            if dark_theme:
                text_color_positive = "#FAFAFA"  # Um branco/off-white para texto principal
                text_color_negative = "#FF4B4B"  # Vermelho padrão do Streamlit para erros/negativos
                label_color = "#FAFAFA"          # Cinza para texto secundário (funciona bem em ambos)
                border_color = "#262730"         # Borda sutil do tema escuro
                background_color = "transparent" # Usa o fundo do próprio tema
            else: # Tema claro
                text_color_positive = "black"
                text_color_negative = "red"
                label_color = "#FAFAFA"
                border_color = "rgba(49, 51, 63, 0.2)"
                background_color ="#FAFAFA"

            # --- Lógica da cor do valor ---
            color = text_color_negative if value < 0 else text_color_positive
            
            # Formata o valor usando o locale para o formato de moeda
            formatted_value = locale.format_string('%10.2f',value, grouping=True)

            # --- Cria o HTML com o estilo dinâmico ---
            html_string = f"""
            <div style="border: 1px solid {border_color}; border-radius: 0.5rem; padding: 1rem; background-color: {background_color}; margin-bottom: 1rem;">
                <div style="font-size: 0.9rem; color: {label_color}; margin-bottom: 0.2rem;">{label}</div>
                <div style="font-size: 1.75rem; color: {color}; font-weight: 600;">{formatted_value}</div>
            </div>
            """
            
            st.markdown(html_string, unsafe_allow_html=True)
    
        # --- Processamento dos Dados ---
        st.subheader("Resultados e Análises")
      
        # Garante que qtd_compra e qtd_venda sejam numéricas
        df["qtd_compra"] = pd.to_numeric(df["qtd_compra"], errors="coerce").fillna(0)
        df["qtd_venda"] = pd.to_numeric(df["qtd_venda"], errors="coerce").fillna(0)


        # Converte a coluna de data para o formato datetime
        date_column_name = None
        for col in ["abertura", "fechamento", "data"]:
            if col in df.columns:
                try:
                    # Tenta converter com infer_datetime_format para maior flexibilidade
                    df[col] = pd.to_datetime(
                        df[col], 
                        errors="coerce", 
                        dayfirst=True,
                        #infer_datetime_format=True
                    )
                    if not df[col].isnull().all():
                        date_column_name = col
                        break
                except Exception as date_e: # Captura exceções específicas de data
                    st.warning(f"Erro ao tentar converter coluna '{col}' para data: {date_e}")
                    continue

        if date_column_name is None:
            st.error(
                "Não foi possível encontrar uma coluna de data válida ('Abertura', 'Fechamento' ou 'Data') no seu arquivo CSV. Verifique se o formato é 'DD/MM/YYYY HH:MM' ou similar e se os dados estão limpos."
            )
            st.stop()
        # periodo da tabela
        data_inicial = pd.to_datetime(df["abertura"].min()).to_pydatetime()
        data_final = pd.to_datetime(df["abertura"].max()).to_pydatetime()
        
        intevalo_datas = st.sidebar.slider("Periodo:",min_value=data_inicial,max_value=data_final,value=((data_inicial,data_final)))
        start_date = intevalo_datas[0]
        end_date = intevalo_datas[1]
        df = df.loc[(df["abertura"] >= start_date) & (df["abertura"] <= end_date)]
        # Agrupa os resultados por dia
        df["data_operacao"] = df[
            date_column_name
        ].dt.normalize()

        # Calcula a soma dos resultados por dia
        daily_results_sum = (
            df.groupby("data_operacao")["res_operacao"].sum().reset_index()
        )
        daily_results_sum.columns = ["Data", "Resultado_Diario"]

        # Calcula os contratos operados por dia
        df["contratos_operados_trade"] = df[["qtd_compra", "qtd_venda"]].max(axis=1)
        contratos_por_dia = (
            df.groupby("data_operacao")["contratos_operados_trade"].sum().reset_index()
        )
        contratos_por_dia.columns = ["Data", "Contratos_Operados"]

        # Junta os resultados diários e contratos por dia
        daily_summary = pd.merge(daily_results_sum, contratos_por_dia, on="Data", how="left").fillna(0)

        # Garante que todas as datas entre a primeira e a última estejam presentes
        min_date = daily_summary["Data"].min()
        max_date = daily_summary["Data"].max()
        all_dates = pd.DataFrame(
            pd.date_range(start=min_date, end=max_date, freq="D"), columns=["Data"]
        )
        daily_summary = pd.merge(all_dates, daily_summary, on="Data", how="left").fillna(0)
        daily_summary = daily_summary.sort_values("Data")


        # Calcula o capital acumulado
        daily_summary["Capital_Acumulado"] = (
            initial_capital + daily_summary["Resultado_Diario"].cumsum()
        )

        # --- Métricas Chave ---
        total_profit_loss = daily_summary["Resultado_Diario"].sum()
        percentage_return = (
            (total_profit_loss / initial_capital) * 100 if initial_capital > 0 else 0
        )

        current_capital = initial_capital + total_profit_loss


        # Cálculos para as novas métricas
        winning_trades_df = df[df["res_operacao"] > 0]
        losing_trades_df = df[
            df["res_operacao"] < 0
        ]

        total_trades = df["res_operacao"].count()
        winning_trades_count = len(winning_trades_df)
        losing_trades_count = len(losing_trades_df)

        win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0

        avg_winning_trade = (
            winning_trades_df["res_operacao"].mean() if winning_trades_count > 0 else 0
        )
        avg_losing_trade = (
            losing_trades_df["res_operacao"].mean() if losing_trades_count > 0 else 0
        )

        gross_profit = winning_trades_df["res_operacao"].sum()
        gross_loss = losing_trades_df["res_operacao"].sum()

        if gross_loss == 0:
            profit_factor_display = "N/A (Sem Prejuízo)"
        else:
            profit_factor = abs(gross_profit / gross_loss)
            profit_factor_display = f"{profit_factor:.2f}"

        max_profit_trade = df["res_operacao"].max() if total_trades > 0 else 0

        if not losing_trades_df.empty:
            max_loss_trade = losing_trades_df["res_operacao"].min()
        else:
            max_loss_trade = "N/A (Sem Prejuízo)"

        # Exibição das métricas
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Capital Inicial",
                value=locale.format_string('%10.2f',initial_capital,grouping=True),
                border=True,
            )
            st.metric(
                label="Lucro/Prejuízo Total",
                #value=locale.currency(total_profit_loss, grouping=True),
                value=locale.format_string('%10.2f',total_profit_loss, grouping=True),
                border=True,
            )
            st.metric(
                label="Retorno Percentual", 
                value=f"{percentage_return:.2f}%", 
                border=True)
            st.metric(
                label="Capital Atual",
                value=locale.format_string('%10.2f',current_capital, grouping=True),
                border=True,
            )
        with col2:
            st.metric(label="Total de Operações", value=f"{total_trades}", border=True)
            st.metric(label="Operações Vencedoras", value=f"{winning_trades_count}", border=True)
            st.metric(label="Operações Perdedoras", value=f"{losing_trades_count}", border=True)
            st.metric(
                label="Contratos Operados (Total)",
                value=f"{df['contratos_operados_trade'].sum()}",
                border=True,
            )
        with col3:
            st.metric(label="Taxa de Acerto", value=f"{win_rate:.2f}%", border=True)
            st.metric(label="Fator de Lucro", value=profit_factor_display, border=True)
            st.metric(
                label="Lucro Médio/Operação",
                value=locale.format_string('%10.2f',avg_winning_trade, grouping=True), border=True,
            )
            # 'avg_losing_trade' deve ser a sua variável com o número, por exemplo: -25.50
            locale.format_string('%10.2f',avg_losing_trade, grouping=True)
            colored_metric("Prejuízo Médio/Operação", avg_losing_trade)
            #st.metric(
            #    label="Prejuízo Médio/Operação",
            #    value="",  # Deixa o valor principal vazio
            #    delta=locale.currency(avg_losing_trade, grouping=True),
            #    delta_color="normal"  # 'normal' faz com que valores negativos fiquem vermelhos
            #)
                
            
            
            
        with col4:
            st.metric(
                label="Maior Lucro (Trade)",
                value=locale.format_string('%10.2f',max_profit_trade, grouping=True),
                
                
                border=True
            )

            if isinstance(max_loss_trade, str):
                st.metric(label="Maior Prejuízo (Trade)", value=max_loss_trade, border=True)
                
            else:
                #st.metric(
                #    label="Maior Prejuízo (Trade)",
                #    value=locale.currency(max_loss_trade, grouping=True),
                #    border=True,
                #)
                locale.format_string('%10.2f',max_loss_trade, grouping=True)
                colored_metric("Maior Prejuízo (Trade)", max_loss_trade)
        st.markdown("---")

        # --- Gráfico da Curva de Capital ---
        #st.subheader("Curva de Capital ao Longo do Tempo")
        fig = px.line(
            daily_summary,
            x="Data",
            y="Capital_Acumulado",
            title="Evolução do Capital",
            labels={"Data": "Data da Operação", "Capital_Acumulado": "Capital Acumulado (R$)"},
        )
        fig.update_xaxes(tickformat="%d/%m/%Y")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- Gráfico do Resultado Diário e Contratos Operados ---
        #st.subheader("Resultado Diário e Contratos Operados por Dia")
        
        fig_daily = px.bar(
            daily_summary,
            x="Data",
            y="Resultado_Diario",
            title="Resultado Diário das Operações (R$) e Contratos Operados por Dia",
            labels={"Data": "Data da Operação", "Resultado_Diario": "Resultado Diário (R$)"},
            color_discrete_sequence=["green"],
        )
        fig_daily.update_traces(
            marker_color=["red" if x < 0 else "green" for x in daily_summary["Resultado_Diario"]]
        )

        fig_daily.add_trace(
            px.line(daily_summary, x="Data", y="Contratos_Operados", color_discrete_sequence=["blue"])
            .data[0]
        )
        
        fig_daily.update_layout(
            yaxis2=dict(
                title="Contratos Operados",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            hovermode="x unified"
        )
        fig_daily.data[1].update(yaxis="y2")
        fig_daily.update_xaxes(tickformat="%d/%m/%Y")
        st.plotly_chart(fig_daily, use_container_width=True)


        # --- Análise por Ativo ---
        #st.subheader("Análise por Ativo")
        df["ativo"] = df["ativo"].astype(str)

        profit_loss_by_asset = (
            df.groupby("ativo")["res_operacao"].sum().reset_index()
        )
        
        col_asset_chart1, col_asset_chart2 = st.columns(2)

        with col_asset_chart1:
            fig_asset_pl_bar = px.bar(
                profit_loss_by_asset,
                x="ativo",
                y="res_operacao",
                title="Lucro/Prejuízo Total por Ativo (Barras)",
                labels={"ativo": "Ativo", "res_operacao": "Lucro/Prejuízo Total (R$)"},
            )
            fig_asset_pl_bar.update_traces(
                marker_color=["red" if x < 0 else "green" for x in profit_loss_by_asset["res_operacao"]]
            )
            st.plotly_chart(fig_asset_pl_bar, use_container_width=True)

        with col_asset_chart2:
            fig_asset_pl_pie = px.pie(
                profit_loss_by_asset,
                values="res_operacao",
                names="ativo",
                title="Proporção de Lucro/Prejuízo por Ativo (Pizza)",
                hole=0.3,
            )
            st.plotly_chart(fig_asset_pl_pie, use_container_width=True)


        # Número de Operações por Ativo
        trades_by_asset = df["ativo"].value_counts().reset_index()
        trades_by_asset.columns = ["Ativo", "Num_Operacoes"]
        fig_asset_trades = px.bar(
            trades_by_asset,
            x="Ativo",
            y="Num_Operacoes",
            title="Número de Operações por Ativo",
            labels={"Ativo": "Ativo", "Num_Operacoes": "Número de Operações"},
        )
        st.plotly_chart(fig_asset_trades, use_container_width=True)

        #st.subheader("Distribuição dos Resultados por Operação")
        fig_hist = px.histogram(
            df,
            x="res_operacao",
            nbins=30,
            title="Distribuição dos Resultados por Operação (Lucro/Prejuízo)",
            labels={"res_operacao": "Resultado da Operação (R$)"},
        )
        # Ajustando o zoom inicial
        
        st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(
            f"Ocorreu um erro ao processar o arquivo. Verifique se o formato está correto e tente novamente. Erro: {e}"
        )
        st.info("Dica: Se o erro persistir, o problema pode estar no formato das datas, na codificação ou no cabeçalho após as 5 linhas iniciais. Tente abrir o CSV em um editor de texto simples (como Notepad++) e verificar a codificação (geralmente indicada na barra de status) ou o formato exato das datas nas colunas 'Abertura'/'Fechamento'.")

#st.sidebar.info(
#    "Este aplicativo é para visualização de performance de trades. "
#    "Os dados não são salvos persistentemente, são processados a cada upload."
#)
