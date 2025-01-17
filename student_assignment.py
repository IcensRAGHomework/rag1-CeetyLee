import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    hw1_response_schemas = [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=hw1_response_schemas)
    hw1_format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions}"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=hw1_format_instructions)
    tmp_response = llm.invoke(prompt.format(question=question)).content

    sample_string1 = '''
                \"data\": \"2024-04-04\",
                \"name\": \"兒童節\"    
                '''
    sample_string12 = '''
                \"data\": \"2024-04-05\",
                \"name\": \"清明節\"    
                '''

    examples = [
        {"input": "``` json {sample_string1} ```", "output": "\"Result\": [ {sample_string1} ]"},
        {"input": "``` json {sample_string2} ```", "output": "\"Result\": [ {sample_string2} ]"},
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
            ("system", "你是能夠將json格式變更的人才"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(final_prompt.format(input={tmp_response})).content
    return response

    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
# def demo(question):
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