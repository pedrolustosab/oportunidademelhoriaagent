import os
import logging
import json
import ast
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import load_tools
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.hub import pull as hub_pull
from typing import Dict, Any, Optional, List

class BusinessAnalystAgent:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4o", 
                 temperature: float = 0.1, 
                 logging_level: int = logging.INFO):
        """
        Initialize the Business Analyst Agent with configurable parameters.
        
        Args:
            api_key: OpenAI API key. Defaults to environment variable.
            model: OpenAI model to use.
            temperature: Creativity/randomness of model.
            logging_level: Logging level.
        """
        # Load environment variables
        load_dotenv()
        
        # Set up logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
        # Validate API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or pass directly.")
        
        # Initialize components
        self.llm = self._initialize_llm(model, temperature)
        self.agent = self._create_react_agent()

    def _initialize_llm(self, model: str, temperature: float) -> ChatOpenAI:
        """Initialize the Language Model with specified parameters."""
        try:
            return ChatOpenAI(
                temperature=temperature,
                model=model,
                api_key=self.api_key,
                verbose=True
            )
        except Exception as e:
            self.logger.error(f"LLM initialization error: {e}")
            raise

    def _create_react_agent(self) -> AgentExecutor:
        """Create a ReAct agent with Python REPL tool and React prompt."""
        try:
            # Prepare tools and prompt
            tools = [PythonREPLTool()]
            prompt = hub_pull("hwchase17/react")
            
            # Create agent executor
            return AgentExecutor(
                agent=create_react_agent(llm=self.llm, tools=tools, prompt=prompt),
                tools=tools, 
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            self.logger.error(f"ReAct agent creation error: {e}")
            raise

    def _construct_analysis_prompt(self, 
                                   ramo_empresa: str, 
                                   direcionadores: str, 
                                   nome_processo: str, 
                                   atividade: str, 
                                   evento: str, 
                                   causa: str) -> str:
        """Construct a detailed analysis prompt from scenario details."""
        
        company_metrics = f"**{direcionadores}**: {direcionadores}, **{nome_processo}**: {nome_processo}, **{atividade}**: {atividade}, **{evento}**: {evento}, **{causa}**: {causa}"
        
        return f"""
Você é um consultor sênior altamente qualificado, especializado em análise de desempenho organizacional para grandes empresas, com foco em **{ramo_empresa}**. Sua tarefa é analisar dados da empresa e fornecer recomendações objetivas de melhoria usando as melhores práticas de consultoria.

**OBJETIVO**:
Usando as métricas do DataFrame **{company_metrics}**, identifique problemas críticos e oportunidades de melhoria na empresa.

**INSTRUÇÕES**:
1. Avalie o impacto de **{causa}** nas operações e considere ferramentas usadas, como Ordem de Serviço, DRE, Ordem de Compras e CRM.
2. Priorize recomendações com alto **ROI** e detalhe a **quantificação de benefícios**, com métricas estimadas de impacto, como percentuais ou valores financeiros.
3. Inclua **possíveis riscos de implementação** para cada solução e estratégias breves de mitigação para aumentar as chances de sucesso.

**METODOLOGIA**:
1. Use metodologias como **Six Sigma, Lean** e **DMAIC** para estruturar a análise de problemas e melhorias.
2. Fundamente suas recomendações com **casos de sucesso do setor**, quando relevante, e proponha melhorias escalonadas para as recomendações que exigem maior investimento inicial.
3. Priorize soluções que sejam **práticas e de implementação viável** dentro das operações atuais da empresa.

**PASSOS PARA ANÁLISE**:
1. Revise a situação atual usando os indicadores do DataFrame.
2. Identifique, com base nos **{direcionadores}**, ao menos **5** problemas que impactam o desempenho em relação a **{evento}**.
3. Para cada problema, proponha uma oportunidade de melhoria incluindo os seguintes elementos:

   - **Problema Mapeado**: Descrição objetiva do problema identificado.
   - **Oportunidade de Melhoria**: Sugestão clara da oportunidade de melhoria para o problema.
   - **Solução**: Descrição detalhada da solução, incluindo especificação de etapas escalonadas, se relevante.
   - **Backlog de Atividades**: Lista de tarefas específicas que devem ser adicionadas ao backlog de implementação.
   - **Investimento**: Estimativa de horas e recursos para desenvolvimento e implantação.
   - **Ganhos Quantificados**: Benefícios esperados em termos percentuais.
   - **Riscos e Mitigação**: Descrição breve dos riscos de implementação e estratégia para mitigá-los.

**FORMATO DE RESPOSTA**:
- Utilize **PythonREPLTool** para gerar uma lista de dicionários Python, conforme o exemplo:
[
    {{
        "Problema Mapeado": "Descrição do problema",
        "Oportunidade de Melhoria": "Descrição da oportunidade de melhoria",
        "Solução": "Descrição da solução proposta",
        "Backlog de Atividades": "  - Atividade 1  - Atividade 2  - Atividade 3",
        "Investimento": "  - Investimento 1  - Investimento 2  - Investimento 3",
        "Ganhos": "  - Ganhos 1  - Ganhos 2  - Ganhos 3",
        "Riscos": "  - Risco 1  - Risco 2  - Risco 3"
    }},
    // outras oportunidades de melhoria
]
- Retorne apenas a lista de dicionários sem aspas adicionais, sem a palavra "'''python" ou "'''json", sem variável e sem explicações.
- Caso seja gerado como string ou variável, transforme diretamente em uma lista de dicionários.
"""

    def analyze_business_scenario(self, 
                                  ramo_empresa: str, 
                                  direcionadores: str, 
                                  nome_processo: str, 
                                  atividade: str, 
                                  evento: str, 
                                  causa: str) -> pd.DataFrame:
        """
        Analyze a business scenario using the ReAct agent.
        
        Returns:
            Analysis results as a DataFrame.
        """
        try:
            # Construct prompt and run analysis
            question = self._construct_analysis_prompt(
                ramo_empresa, direcionadores, nome_processo, 
                atividade, evento, causa
            )
            
            # Invoke agent and parse response
            response = self.agent.invoke({"input": question})
            analysis_data = self._parse_agent_response(response.get('output', ''))
            
            return pd.DataFrame(analysis_data)
        
        except Exception as e:
            self.logger.error(f"Business scenario analysis error: {e}")
            return pd.DataFrame()

    def _parse_agent_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the agent's response, ensuring it's a list of dictionaries.
        
        Args:
            response: Raw response string from the agent
        
        Returns:
            List of dictionaries with analysis results
        """
        try:
            # Clean up the response - remove leading/trailing whitespace
            response = response.strip()
            
            # Try multiple parsing strategies
            parsing_strategies = [
                # 1. Direct JSON parsing
                lambda r: json.loads(r),
                
                # 2. Literal evaluation (safer than eval)
                lambda r: ast.literal_eval(r),
                
                # 3. Remove Python list representation artifacts
                lambda r: json.loads(r.replace("'", '"')),
                
                # 4. Try extracting JSON-like content
                lambda r: json.loads(r[r.index('['):r.rindex(']')+1].replace("'", '"'))
            ]
            
            for strategy in parsing_strategies:
                try:
                    parsed_data = strategy(response)
                    
                    # Validate the parsed data
                    if isinstance(parsed_data, list) and all(isinstance(item, dict) for item in parsed_data):
                        return parsed_data
                    elif isinstance(parsed_data, dict):
                        return [parsed_data]
                except (json.JSONDecodeError, SyntaxError, ValueError, IndexError):
                    continue
            
            # If all parsing attempts fail
            self.logger.error(f"Could not parse response: {response}")
            return []
        
        except Exception as e:
            self.logger.error(f"Unexpected parsing error: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Example scenario inputs
    ramo_empresa = 'Construção'
    direcionadores = 'Redução de Custo'
    nome_processo = 'Suprir materiais'
    atividade = 'Realizar compra'
    evento = 'Valor gasto é maior que foi orçado para a área'
    causa = 'Ordem de compra não é vinculada ao orçamento'
    
    # Initialize and analyze
    analyst = BusinessAnalystAgent()
    results = analyst.analyze_business_scenario(ramo_empresa, direcionadores, nome_processo, atividade, evento, causa)
    print(results)