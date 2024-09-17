# Implementação NumPy do Modelo de Linguagem Probabilístico Neural

Este repositório contém uma implementação do Modelo de Linguagem Probabilístico Neural proposto por Yoshua Bengio et al. no artigo "A Neural Probabilistic Language Model" (2003). A implementação é feita usando apenas NumPy para fins educacionais e de compreensão profunda da arquitetura do modelo. Esse artigo foi muito importante na introdução da embedding layer. Minha implementação foi treinada no texto Dom Casmurro de Machado de Assis.

## Uso

Primeiramente deve ser feito o treinamento do modelo rodando o arquivo `train.py`.Depois, o modelo pode ser utilizado rodando o arquivo `generate.py` e o contexto pode ser alterado na linha tokenizer.encode("José Dias amava ").

## Resultados

Os resultados podem ser encontrados no arquivo `output.txt` na raiz do projeto.

## Referências

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. Journal of Machine Learning Research, 3(Feb), 1137-1155.
- Assis, Machado de. Dom Casmurro. Rio de Janeiro: Garnier, 1899.