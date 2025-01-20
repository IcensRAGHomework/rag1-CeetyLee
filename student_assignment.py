import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def get_openAI_llm():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

def get_hw01_tmp_response(llm: BaseChatModel, question: str):
    holiday_response_schemas = [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=holiday_response_schemas)
    holiday_format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions},只需回答問題內容就好，所有答案放進同個list中"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=holiday_format_instructions)
    tmp_response = llm.invoke(prompt.format(question=question)).content
    return tmp_response

def get_tmp_result(llm: BaseChatModel, result: str):
    format_response_schemas = [
        ResponseSchema(
            name="Result",
            description="json的格式內容",
        )
    ]

    json_output_parser = StructuredOutputParser(response_schemas=format_response_schemas)
    json_format_instructions = json_output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將提供的json內容整理為指定的json內容做輸出, {format_instructions}, 不要增加額外資訊"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=json_format_instructions)
    tmp_result = llm.invoke(prompt.format(question=result)).content
    return tmp_result

def generate_hw01(question):
    llm = get_openAI_llm()
    tmp_response = get_hw01_tmp_response(llm, question)
    #print({tmp_response})
    tmp_result = get_tmp_result(llm, tmp_response)
    #print({tmp_result})
    examples = [
        {"input": """```json
                    {
                        "Result": [ 
                            content 
                        ]
                    }   
                    ```""",
        "output": """{
                        "Result": [ 
                            content 
                        ]
                    }"""},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

#    print(few_shot_prompt.invoke({}).to_messages())

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "依照我提供的文字內容仔細比對範例進行處理，去除開頭與結尾應該不需要的字串"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(final_prompt.format(input={tmp_result})).content
    return response

    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
# 測試環境用
# print(demo("你好，使用繁體中文").content)

question = "2025年台灣4月紀念日有哪些?"
answer = generate_hw01(question)
print(answer)