from langchain_community.llms.llamacpp import LlamaCpp
from langchain_openai import ChatOpenAI
import dotenv


dotenv.load_dotenv()


def get_llama_cpp_chat_model(temperature=0.8, top_p=0.95, use_quantized_model=False):
    """ llama max token length is 4096 """
    llama_model_path = '/mnt/nas1/models/llama/gguf/llama2-7b-chat.gguf'
    if use_quantized_model:
        llama_model_path = '/mnt/nas1/models/llama/gguf/llama2-7b-chat.q4_k_m.gguf'
    model = LlamaCpp(
        model_path=llama_model_path,
        max_tokens=2000,
        n_gpu_layers=-1,
        n_ctx=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=100,
        # callback_manager=callback_manager,
        # verbose=True,  # Verbose is required to pass to the callback manager
    )
    return model


def get_chatglm_api():
    # default endpoint_url for a local deployed ChatGLM api server
    openai_api_base = "http://127.0.0.1:8000/v1"

    llm = ChatOpenAI(model="chatglm3-6b", openai_api_base=openai_api_base)
    return llm


if __name__ == "__main__":
    pass
