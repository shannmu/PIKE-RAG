# .env File Format

Please follow below environment configuration variable names to create your *.env* file, we suggest you put it under
`PIKE-RAG/env_configs/` which has already been added to *.gitignore* file:

## For Azure OpenAI Client

```sh
AZURE_OPENAI_ENDPOINT = "YOUR-ENDPOINT(https://xxx.openai.azure.com/)"
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2023-07-01-preview"
```

*Note that the way to access GPT API with key is disabled in Azure now.*

To access GPT resource from Azure, please remember to login to Azure CLI using your *SC-* account:

```sh
# Install Azure-CLI and other dependencies. Sudo permission is required.
bash scripts/install_az.sh

# Login Azure CLI using device code.
bash scripts/login_az.sh
```

## For Azure Meta LlaMa Client

Since the endpoint and API keys varied among different LlaMa models, you can add multiple
(`llama_endpoint_name`, `llama_key_name`) pairs you want to use into the *.env* file, and specify the names when
initializing `AzureMetaLlamaClient` (you can modify the llm client args in the YAML files). If `null` is set to be the
name, the (`LLAMA_ENDPOINT`, `LLAMA_API_KEY`) would be used as the default environment variable name.

```sh
# Option 1: Set only one pair in one time, update these variables every time you want to change the LlaMa model.
LLAMA_ENDPOINT = "YOUR-LLAMA-ENDPOINT"
LLAMA_API_KEY = "YOUR-API-KEY"

# Option 2: Add multiple pairs into the .env file, for example:
LLAMA3_8B_ENDPOINT = "..."
LLAMA3_8B_API_KEY = "..."

LLAMA3_70B_ENDPOINT = "..."
LLAMA3_70B_API_KEY = "..."
```

### Ways to Get the Available Azure Meta LLaMa **Endpoints**, **API Keys** and **Model Names**

The way we have implemented the LLaMa model so far involves requesting the deployed model on the GCR server. You can
find the available settings follow the steps below:

1. Open [Azure Machine Learning Studio](https://ml.azure.com/home), sign in may be required;
2. Click *Workspaces* on the left side (expand the menu by clicking the three horizontal lines in the top left corner if
you cannot find it);
3. Choose and click on a valid workspace, e.g., *gcrllm2ws*;
4. Click *Endpoints* on the left side (expand the menu by clicking the three horizontal lines in the top left corner if
you cannot find it), You can find the available model list in this page;
5. Choose and click the model you want to use, e.g., *gcr-llama-3-8b-instruct*:
    - **model** name: in tab "Details", scroll to find "Deployment summary", the *Live traffic allocation* string (e.g.,
        *meta-llama-3-8b-instruct-4*) is the model name you need to set up in your YAML file;
    - **LLAMA_ENDPOINT** & **LLAMA_API_KEY**: can be found in tab "Consume".

### Handling the Issue "Specified deployment could not be found"

If you get error message "Specified deployment could not be found", it indicates that the GCR team has changed the
server deployment location. In this case, you need to check the available model list in
[Azure Machine Learning Studio](https://ml.azure.com/home) and update the YAML config again.

*Return to the main [README](https://github.com/microsoft/PIKE-RAG/blob/main/README.md)*
