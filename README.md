# ADA
Proyecto Bintec ADA (Abastecimiento Digital Avanzado)
![alt text](imagenes/baner.png)

# ADA (Abastecimiento Digital Avanzado)

ADA es la iniciativa de abastecimiento para generar un ecosistema donde se centralice con calidad toda la información y se disponibilice a través de analítica avanzada como NLP y ML.

## Ejecución
1. Crear y activar ambiente virtual.

   ```bash
   python -m venv .venv
   ```
2. Instalar dependencias.
   ```bash
   pip install -r requirements.txt
   ```
3. Para crear la vector store será necesario colocar los documentos `pdf` dentro de una carpeta llamada `documents` en la raíz del proyecto. 
<br>Del mismo modo, crear una carpeta llamada `data`, y en ella guardar un `.xlsx` con la tabla de requerimientos. Finalmente deberá ejecutar:
   ```bash
   python notebooks/Embedding.py
   ```
4. Ejecutar proyecto:
   ```bash
   chainlit run main.py -w
   ```
## Arquitectura de la solución

ADA se basa en la función de `Agentes` que ofrece `LangChain`, soportado a su vez por un LLM de OpenAI. Este GPT tiene dos funciones principales:

1. Responder preguntas respecto a información de proveedores, su gasto y sus contratos en la cadena de abastecimiento, usando la funcionalidad de `create_pandas_dataframe_agent` de `LangChain`.

2. Consultar a cerca de información de abastecimiento por medio de la herramienta `RetrivalQA` de `LangChain`.

Los modelos usados, tanto en el agente principal, como en las tools que este usa, son previamente entrenados. Se hizo uso de _context learning_ para dotar a los mencionados con la capacidad de resolver de este ámbito. 

Con el objetivo de disminuir las alucinaciones en los modelos usados, se estableció su temperatura en `0`, y se limitaron sus funcionalidades en los _prompt templates_.

![alt text](imagenes/arquitectura.png)