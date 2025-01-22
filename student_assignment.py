import json
import traceback
import requests
import base64

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from mimetypes import guess_type

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

calendarific_api = "ZMfZRia1MUjMfgqoOxZLRsQeLG2u24dM"
history = ChatMessageHistory()

def get_openAI_llm():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

def get_holiday_tmp_response(llm: BaseChatModel, question: str):
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

def get_format_result(llm: BaseChatModel, result: str):
    format_response_schemas = [
        ResponseSchema(
            name="Result",
            description="json的格式內容",
            type="list"
        )
    ]

    json_output_parser = StructuredOutputParser(response_schemas=format_response_schemas)
    json_format_instructions = json_output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將提供的json內容整理為指定的json內容做輸出, {format_instructions}, 不要增加額外資訊"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=json_format_instructions)
    tmp_result = llm.invoke(prompt.format(question=result)).content

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
            ("system", "依照我提供的文字內容仔細比對範例進行處理，記得去除開頭與結尾應該不需要的字串"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(final_prompt.format(input=tmp_result)).content
    return response

def get_hw_format_result(llm: BaseChatModel, result: str):
    format_response_schemas = [
        ResponseSchema(
            name="Result",
            description="json的格式內容",
            type="list"
        )
    ]

    json_output_parser = StructuredOutputParser(response_schemas=format_response_schemas)
    json_format_instructions = json_output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將提供的json內容整理為指定的json內容做輸出, {format_instructions}, 不要增加額外資訊"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=json_format_instructions)
    tmp_result = llm.invoke(prompt.format(question=result)).content

    examples = [
        {"input": """```json
                    {
                        "Result": [ 
                            content 
                        ]
                    }   
                    ```""",
        "output": """{
                        "Result": 
                            content 
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
            ("system", "依照我提供的文字內容仔細比對範例進行處理，記得保留格式以及去除不需要的字串或符號"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(final_prompt.format(input=tmp_result)).content
    return response

def get_agent(llm: BaseChatModel):
    def get_holiday_value(year: int, month: int) -> str:
        url = f"https://calendarific.com/api/v2/holidays?&api_key={calendarific_api}&country=tw&year={year}&month={month}"
        response = requests.get(url)
        response = response.json()
        response = response.get('response')
        return response
    
    class GetHolidayValue(BaseModel):
        year: int = Field(description="specific year of holiday")
        month: int = Field(description="specific month of holiday")

    tool = StructuredTool.from_function(
        name="get_holiday_value",
        description="get holiday form calendarific api",
        func=get_holiday_value,
        args_schema=GetHolidayValue,
    )

    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    # agent_executor.invoke is for hw02
    # response = agent_executor.invoke({"input": question}).get('output')
    # below is for hw02/hw03
    def get_history() -> ChatMessageHistory:
        return history
    
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def generate_hw01(question):
    llm = get_openAI_llm()
    tmp_response = get_holiday_tmp_response(llm, question)
    # print(tmp_response)
    response = get_format_result(llm, tmp_response)
    return response
    
def generate_hw02(question):
    llm = get_openAI_llm()
    agent = get_agent(llm)
    response = agent.invoke({"input": question}).get('output')

    # below is hw01
    tmp_response = get_holiday_tmp_response(llm, response)
    response = get_format_result(llm, tmp_response)
    return response

    
def generate_hw03(question2, question3):
    generate_hw02(question2)
    llm = get_openAI_llm()
    agent = get_agent(llm)

    response_schemas = [
        ResponseSchema(
            name="add",
            description="表示是否需要將節日新增到節日清單中, 根據問題判斷該節日是否存在於清單中, 如果不存在, 則為true, 否則為false",
            type="boolean"
        ),
        ResponseSchema(
            name="reason",
            description="描述為什麼需要或不需要新增節日"
        )
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    hw3_format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions}"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=hw3_format_instructions)
    tmp_response = agent.invoke({"input": prompt.format_messages(question=question3)}).get('output')
    # response = get_format_result(llm, tmp_response)
    response = get_hw_format_result(llm, tmp_response)
    return response

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        # Default MIME type if none is found
        mime_type = 'application/octet-stream'
    
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encode_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encode_data}"
    
def generate_hw04(question):
    llm = get_openAI_llm()
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)

    response_schemas = [
        ResponseSchema(
            name="score",
            description="請解析提供的圖片檔案, 並回答圖片中指定隊伍的積分",
            type="integer"
        )
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    hw4_format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","請辨識圖片中的文字表格,{format_instructions}"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ],
        ),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=hw4_format_instructions)
    tmp_response = llm.invoke(prompt.format_messages(question=question)).content
    # response = get_format_result(llm, tmp_response)
    response = get_hw_format_result(llm, tmp_response)
    return response    

    
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

# question = "2025年台灣4月紀念日有哪些?"
# answer = generate_hw01(question)
# answer = generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前紀錄的節日清單中，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單')
question = "請問中華台北的積分是多少"
answer = generate_hw04(question)
print(answer)