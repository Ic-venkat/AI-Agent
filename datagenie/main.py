from dotenv import find_dotenv,load_dotenv
from pprint import pprint
import os
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
import pandas as pd
from langchain.docstore.document import Document
import streamlit as st
from loadcsv import loadCSV
from langchain.schema import AIMessage, HumanMessage
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pandasai.llm import OpenAI
import os
from typing import Annotated, List, Tuple, Union
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from pandasai import SmartDataframe
from typing import Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langgraph.prebuilt.tool_executor import ToolExecutor
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import END, StateGraph
from langchain_core.messages import SystemMessage


os.environ["LANGCHAIN_PROJECT"]="raglanggraph"
load_dotenv(find_dotenv(), override=True)


llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
llmPandasAI = OpenAI(api_token=os.environ["OPENAI_API_KEY"])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

title_container = st.container()
with title_container:
    st.title("Ericsson Data Genie")

uploaded_file = st.sidebar.file_uploader(
    "Upload File For Data Analysis",
    type="csv",
    accept_multiple_files=False,
    key="file-uploader-side-bar",
)

# Creating a document retriver tool
class retrieverInput(BaseModel):
    input: str = Field(description="Input string to search and retrieve documents")
    k: int = Field(description = "This value is to fetch only top K documents")
    
def retrieve(input: str, k:int)->list:
    retriever = vectorstore.as_retriever(search_kwargs={'k':k})
    docs = retriever.get_relevant_documents(input,)
    return docs
    
retrieverDoc = StructuredTool.from_function(
    func=retrieve,
    name="retriever",
    description="Retrieves relevant information about a particular incident, if there is no relevant document then increase k value",
    args_schema=retrieverInput,
    return_direct=False
)


# Creating a pandas data analyst tool
class dataAnalystInput(BaseModel):
    input: str = Field(description="Input string for data analyst tool to answer")
def analyst(input: str)->Any:
    df1 = SmartDataframe(df, config={"llm":llmPandasAI})
    output = df1.chat(input)
    return output

dataAnalyst = StructuredTool.from_function(
    func=analyst,
    name="dataAnalyst",
    description="This tool answers the question for user input using pandas in the background and returns response, if user asks for any question which required full overview, data transformation then use this. Example: Give me total count of data. Answer: Total incident count is 100",
    args_schema=dataAnalystInput,
    return_direct=True,
)

# Binding tools to llm for function invocation
tools = [retrieverDoc, dataAnalyst]
tool_executor = ToolExecutor(tools)
functions = [format_tool_to_openai_function(t) for t in tools]
llm = llm.bind_functions(functions)


# Creating different supervisor node
def supervisor(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Defining continue function this takes descition weather to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    if("function_call" not in last_message.additional_kwargs):
        return "end"
    else:
        return "continue"

# Call tool function, this is another node in langgraph where we call different tools. llm will take a descition to call the tool.
def call_tool(state):
    messages = state['messages']
    last_message = messages[-1]
    print(last_message)
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    print(f"The agent action is {action}")
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    print(f"The tool result is: {response}")
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=";")
        vectorstore = loadCSV(df) 
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.stop()
        
        
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    st.session_state["chat_history"].append(
        SystemMessage(content="You are a helpful assistant, use retriever tool to retrieve information about a incident and use data analyst tool to do basic data analysis based on user input")
        )


# Creating langgraph workflow with 2 nodes and 2 tools, in these 2 tools one of the tool acts as pandas data analyst agent.
workflow = StateGraph(AgentState)
workflow.add_node("supervisor",supervisor)
workflow.add_node("call_tool", call_tool)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    should_continue,
    {
        "continue": "call_tool",
        "end": END
    }
)
workflow.add_edge("call_tool","supervisor")
app = workflow.compile()


def chat_actions():
    st.session_state["chat_history"].append(
        HumanMessage(content=st.session_state["Chat_input-chatbot"]),
    )
    try:
        
        config = {"recursion_limit": 5}
        output_result = app.invoke(
            {
                "messages": st.session_state["chat_history"]
            }, config=config
        )
        st.session_state["chat_history"].append(
            AIMessage(content=output_result["messages"][-1].content)
        )

        for i in st.session_state["chat_history"]:
            with st.chat_message(name=i.dict()["type"]):
                st.write(i.content)
    except Exception as e:
        print(f"Exception occured {e}")
        


prompt = st.chat_input(
    "Say something", key="Chat_input-chatbot", on_submit=chat_actions
)