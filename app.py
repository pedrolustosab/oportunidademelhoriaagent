import streamlit as st
import base64
from pathlib import Path
import os
from dotenv import load_dotenv
from process_analyse import BusinessAnalystAgent
import pandas as pd
from io import BytesIO

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def stylable_container(key, css_styles):
    """
    Create a stylable container with custom CSS.
    """
    st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child({key}) {{
            {css_styles}
        }}
        </style>
    """, unsafe_allow_html=True)
    return st.container()

def add_bg_from_local(image_file):
    """
    Add a background image to the Streamlit app from a local file.
    """
    with Path(image_file).open("rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_css(css_file):
    """
    Load and apply CSS from a file to the Streamlit app.
    """
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_button_style(button_class):
    """
    Get the button style based on its class.
    """
    if button_class == "current":
        return "background-color: #01374C; color: white; font-weight: bold; font-size: 24px;"
    elif button_class == "previous":
        return "background-color: #4B4843; color: white; font-weight: normal; font-size: 24px;"
    else:
        return "background-color: #4B484340; color: #333333; font-size: 18px;"

def setup_navigation():
    """
    Set up the navigation sidebar for the app.
    """
    st.sidebar.markdown("<h1 style='text-align: center; color: #AC8D61;'>Navegação</h1>", unsafe_allow_html=True)
    
    #Pages
    pages = ["🔍 Oportunidade de melhorias", "📋 Planilha Final"]
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    for i, page in enumerate(pages):
        button_class = "current" if i == st.session_state.current_page else "previous" if i == st.session_state.current_page - 1 else ""

        if st.sidebar.button(page, key=f"nav_{i}", use_container_width=True, disabled=(i == st.session_state.current_page)):
            st.session_state.current_page = i
            st.rerun()

        st.markdown(f"""
            <style>
            div.row-widget.stButton > button[key="nav_{i}"] {{
                {get_button_style(button_class)}
            }}
            </style>
            """, unsafe_allow_html=True)

    return pages

def convert_df_to_excel(df):
    """
    Convert DataFrame to Excel file
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Oportunidade de melhorias', index=False)
        # Auto-adjust columns' width
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Oportunidade de melhorias'].column_dimensions[chr(65 + col_idx)].width = column_width + 2
    
    return output.getvalue()

def render_diagnostico():
    """
    Render the "Oportunidade de Melhoria" page with input preservation.
    """
    # Initialize session state for form inputs if not exists
    if 'form_inputs' not in st.session_state:
        st.session_state.form_inputs = {
            'ramo_empresa': '',
            'direcionadores': '',
            'nome_processo': '',
            'atividade': '',
            'evento': '',
            'causa': ''
        }

    # Create a form for user input with preserved values
    with st.form(key='oportunidade_melhoria_form'):
        # Input fields with preserved values
        ramo_empresa = st.text_input(
            "Ramo da empresa", 
            value=st.session_state.form_inputs['ramo_empresa'], 
            placeholder="Digite o ramo da empresa",
            key='input_ramo_empresa'
        )
        direcionadores = st.text_input(
            "Direcionadores", 
            value=st.session_state.form_inputs['direcionadores'], 
            placeholder="Digite os direcionadores de negócios",
            key='input_direcionadores'
        )
        nome_processo = st.text_input(
            "Nome do processo", 
            value=st.session_state.form_inputs['nome_processo'], 
            placeholder="Digite o nome do processo",
            key='input_nome_processo'
        )
        atividade = st.text_input(
            "Atividade", 
            value=st.session_state.form_inputs['atividade'], 
            placeholder="Digite a atividade",
            key='input_atividade'
        )
        evento = st.text_input(
            "Evento", 
            value=st.session_state.form_inputs['evento'], 
            placeholder="Digite o evento",
            key='input_evento'
        )
        causa = st.text_input(
            "Causa", 
            value=st.session_state.form_inputs['causa'], 
            placeholder="Digite a causa",
            key='input_causa'
        )

        # Submit button
        submit_button = st.form_submit_button(label='Obter Oportunidade de melhorias')

    # Process form submission
    if submit_button:
        # Update session state inputs
        st.session_state.form_inputs = {
            'ramo_empresa': ramo_empresa,
            'direcionadores': direcionadores,
            'nome_processo': nome_processo,
            'atividade': atividade,
            'evento': evento,
            'causa': causa
        }

        if ramo_empresa and direcionadores and nome_processo and atividade and evento and causa:              
            
            #Apply AI
            with st.spinner('Oportunidade de melhorias em andamento...'):  
                analyst = BusinessAnalystAgent()
                resultados = analyst.analyze_business_scenario(ramo_empresa, direcionadores, nome_processo, atividade, evento, causa)               
            st.success("Oportunidade de melhorias obtidas com sucesso.")
            
            # Store in the session state
            st.session_state.resultados = resultados

            # Convert to Excel and allow download
            excel_file = convert_df_to_excel(resultados)
            st.download_button(
                label="📥Baixar Excel",
                data=excel_file,
                file_name='oportunidade_melhoria.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )            
        else:
            st.warning("Por favor, preencha todos os campos antes da obtenção das Oportunidade de melhorias.")

def render_planilha_final():
    """
    Render the "Planilha Final" page showing all opportunities 
    with editable forms for each.
    """
    # Retrieve 'resultados' from session state
    if 'resultados' not in st.session_state:
        st.warning("Não foi executado a obtenção das Oportunidade de melhorias.")
        return
    
    resultados = st.session_state.resultados
    st.session_state.resultados_dict = resultados.to_dict('records')

    # Create a container to hold all opportunity forms
    st.write("## Todas as Oportunidades de Melhoria")

    # Track if any changes were made
    changes_made = False

    # Iterate through each opportunity and create an editable form
    for idx, row in enumerate(st.session_state.resultados_dict):
        with st.form(key=f'opportunity_form_{idx}'):
            st.write(f"### Oportunidade {idx + 1}")
            
            # Display each field in a text area for editing
            oportunidade_de_melhoria = st.text_area(
                "Oportunidade de Melhoria", 
                value=row['Oportunidade de Melhoria'], 
                key=f"oportunidade_{idx}", 
                height=100
            )
            solucao = st.text_area(
                "Solução", 
                value=row['Solução'], 
                key=f"solucao_{idx}", 
                height=100
            )
            backlog_de_atividades = st.text_area(
                "Backlog de Atividades", 
                value=row.get('Backlog de Atividades', ''), 
                key=f"backlog_{idx}", 
                height=100
            )
            investimento = st.text_area(
                "Investimento", 
                value=row.get('Investimento', ''), 
                key=f"investimento_{idx}", 
                height=100
            )
            ganhos = st.text_area(
                "Ganhos", 
                value=row.get('Ganhos', ''), 
                key=f"ganhos_{idx}", 
                height=100
            )
            riscos = st.text_area(
                "Riscos", 
                value=row.get('Riscos', ''), 
                key=f"riscos_{idx}", 
                height=100
            )

            # Submit button for each form
            submit_button = st.form_submit_button(label=f"Salvar Edição para Oportunidade {idx + 1}")

            # Process form submission
            if submit_button:
                # Update the row in the dictionary
                row.update({
                    'Oportunidade de Melhoria': oportunidade_de_melhoria,
                    'Solução': solucao,
                    'Backlog de Atividades': backlog_de_atividades,
                    'Investimento': investimento,
                    'Ganhos': ganhos,
                    'Riscos': riscos
                })
                changes_made = True
                st.success(f"Edição salva para Oportunidade {idx + 1}")

    # If changes were made, update the DataFrame and provide download
    if changes_made:
        # Update DataFrame with edited rows
        edited_df = pd.DataFrame(st.session_state.resultados_dict)
        st.session_state.resultados = edited_df

        # Convert updated DataFrame to Excel for download
        excel_file = convert_df_to_excel(edited_df)
        st.download_button(
            label="📥 Baixar Excel com Todas as Edições",
            data=excel_file,
            file_name='Oportunidade_de_melhorias_final.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Oportunidade de Melhoria", layout="wide")
    add_bg_from_local('background.png')
    load_css('style.css')  # Load the external CSS file
    
    pages = setup_navigation()
    
    # Calculate progress value
    progress_value = (st.session_state.current_page + 1) / len(pages)
    
    # Create layout with title on the left and logo on the right
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<p class="big-font">{pages[st.session_state.current_page]}</p>', unsafe_allow_html=True)
    with col2:
        st.image('logo.png', width=200)
    
    # Add progress bar
    st.progress(progress_value)

    # Create a container for the main content
    main_container = st.container()
    with main_container:
        # Render appropriate page content based on current page
        if pages[st.session_state.current_page] == "🔍 Oportunidade de melhorias":
            render_diagnostico()  
        elif pages[st.session_state.current_page] == "📋 Planilha Final":
            render_planilha_final()
        else:
            st.write(f"This is {pages[st.session_state.current_page]} page. Add your content here.")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.current_page > 0:
            if st.button("Anterior", key="prev_button"):
                st.session_state.current_page -= 1
                st.rerun()
    with col3:
        if st.session_state.current_page < len(pages) - 1:
            if st.button("Próximo", key="next_button"):
                st.session_state.current_page += 1
                st.rerun()
        elif st.session_state.current_page == len(pages) - 1:
            if st.button("Finalizar", key="finish_button"):
                st.success("Processo finalizado com sucesso!")

if __name__ == "__main__":
    main()