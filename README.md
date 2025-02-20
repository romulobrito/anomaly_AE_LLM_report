# Sistema AE + LLM para Análise de Pavimentos

Sistema que combina Autoencoder (AE) e Large Language Model (LLM) para detecção e análise de anomalias em pavimentos rodoviários.

## Estrutura do Projeto
- `src/`: Códigos fonte
  - `models/`: Modelos de ML
  - `utils/`: Funções utilitárias
- `notebooks/`: Jupyter notebooks
- `data/`: Arquivos de dados
- `requirements.txt`: Dependências

## Instalação
```bash
pip install -r requirements.txt
```


## Configuração

1. Copie o arquivo `.env.example` para `.env`:
    ```bash
    cp .env.example .env
    ```
2. Edite o arquivo `.env` e adicione sua API key da OpenAI:
    ```bash
    OPENAI_API_KEY=sua-chave-aqui
    ```

## Dependências Principais
- PyTorch
- Lightning
- Polars
- Pandas
- NumPy
- Matplotlib
- Plotly
- TQDM
- Scikit-learn
- OpenAI


## Uso
1. Prepare os dados no formato adequado
2. Execute o notebook principal `iri_AE.ipynb`
3. Analise os resultados gerados



