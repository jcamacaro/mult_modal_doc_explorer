from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from langchain.llms import HuggingFacePipeline
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import torch
import os


def initialize_llm(llm_model, ll_model_type=None):
    if ll_model_type == 'local':
        model_id = llm_model
        if llm_model in  ["gpt2-medium", 'distilgpt2']:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            task = "text-generation"
        elif llm_model in ['tiiuae/falcon-40b-instruct']:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            task = "text-generation"
            model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    offload_folder="offload",
                    trust_remote_code=True,
                    device_map="auto",
                    # cache_dir="model_dir",
                    )
        elif llm_model in ['facebook/blenderbot-1B-distill',
                           'MBZUAI/LaMini-Flan-T5-783M',
                           'google/flan-t5-large',
                           'facebook/blenderbot-1B-distill'
                           ]:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            task = "text2text-generation"
        elif llm_model in ['chavinlo/alpaca-native',
                           "chainyo/alpaca-lora-7b"]:
            tokenizer = LlamaTokenizer.from_pretrained(llm_model)
            model = LlamaForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto',
                # device_map={"": 'cpu'}
            )
            task = "text-generation"

        pipe = pipeline(task,
                        model=model,
                        tokenizer=tokenizer,
                        max_length=4000,
                        temperature=0.6,
                        top_p=0.95,
                        repetition_penalty=0.5
                        )
        the_llm = HuggingFacePipeline(pipeline=pipe)
    else:
        if llm_model in ['gpt-4', 'gpt-3.5-turbo']:
            the_llm = ChatOpenAI(
                temperature=0.7,
                model_name=llm_model,
                max_tokens=1000,
                top_p=0.95,
                presence_penalty=0.5
            )
        elif llm_model == 'openai':
            the_llm = OpenAI()

    return the_llm
