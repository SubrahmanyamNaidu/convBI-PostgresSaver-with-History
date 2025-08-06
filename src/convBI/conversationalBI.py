from dotenv import load_dotenv
load_dotenv()

import os 

from langgraph.graph import StateGraph,START,END 
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from typing_extensions import TypedDict
from typing import Annotated,Dict,Any


from prompts import intent_prompt,greeting_prompt,table_identification_prompt,prompt_ddl,text_to_sql_prompt
import psycopg 
import json


class WorkflowState(TypedDict):
    history:Annotated[list,add_messages]
    question:str 
    intent:str
    database_ddl:str
    tablename:str 
    rephrased_question:str 
    semantic_info:Dict[str,Any]
    sql_query:str 
    query_result:str 
    query_error_message:str
    needs_clarification:bool 
    visualization_data:Dict[str,Any]

class TextToSQLWorkflow:
    def __init__(self):
        self.llm=AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    def _build_workflow(self)->StateGraph[WorkflowState]:
        graph_builder=StateGraph(WorkflowState)
        graph_builder.add_node("intent_classification",self._intent_classification_agent)
        graph_builder.add_node("greeting",self._greeting_agent)
        graph_builder.add_node("database_ddl",self._database_ddl_agent)
        graph_builder.add_node("table_identification",self._table_identification_agent)
        graph_builder.add_node("table_semantics_info",self._table_semantics_info_agent)
        graph_builder.add_node("text_to_sql",self._text_to_sql_agent)

        graph_builder.add_edge(START,"intent_classification")
        graph_builder.add_conditional_edges(
            "intent_classification",
            lambda state: state["intent"]=="general",
            {True:"greeting",False:"database_ddl"}
            )
        
        graph_builder.add_edge("database_ddl","table_identification")
        graph_builder.add_edge("table_identification","table_semantics_info")
        graph_builder.add_edge("table_semantics_info","text_to_sql")
        graph_builder.add_edge("greeting",END)
        graph_builder.add_edge("text_to_sql",END)

        return graph_builder

    def _intent_classification_agent(self,state:WorkflowState)->WorkflowState:
        prompt=ChatPromptTemplate.from_messages(intent_prompt)

        prev_conv=state["history"][-6:] if state["history"] else []

        chain=prompt|self.llm 
        result=chain.invoke({
            "question":state["question"],
            "history":prev_conv   
            })
        
        state["intent"]=result.content.strip().lower() # need to a validation for the ["general","system_query"]
        with open("intent.json","w") as intent_json:
            json.dump(state,intent_json,indent=2)

        return state

    def _greeting_agent(self,state:WorkflowState)->WorkflowState:
        prompt=ChatPromptTemplate.from_messages(greeting_prompt)
        chain=prompt|self.llm 
        result=chain.invoke({
            "question":state["question"]
        })
        state["final_answer"]=result.content.strip()

        return state
    
    def _database_ddl_agent(self,state:WorkflowState)->WorkflowState:
        state["database_ddl"]=prompt_ddl
        return state
    
    def _table_identification_agent(self,state:WorkflowState)->WorkflowState: 
        prompt=ChatPromptTemplate.from_messages(table_identification_prompt)
        prev_conv=state["history"][-6:] if state["history"] else []
        chain=prompt|self.llm 
        result=chain.invoke({
            "question":state["question"],
            "history":prev_conv, 
            "ddl":state["database_ddl"]
        })
        state["tablename"]=result.content.strip()
        return state
    
    def _table_semantics_info_agent(self,state:WorkflowState)->WorkflowState:
        with open("test.semantics.json","r") as semantics_json:
            semantics=json.load(semantics_json)
        
        required_table_semantics=semantics[state["tablename"]]
        
        state["semantic_info"]=required_table_semantics

        return state
    
    def _text_to_sql_agent(self,state:WorkflowState)->WorkflowState:
        prompt=ChatPromptTemplate.from_messages(text_to_sql_prompt)

        prev_conv=state["history"][-6:] if state["history"] else []
        chain=prompt|self.llm
        result=chain.invoke({
            "semantic_info":state["semantic_info"] ,
            "question":state["question"],
            "history":prev_conv
        })

        state["sql_query"]=result.content.strip()

        state["history"]=[{"role":"user", "content": state["question"]}]
        state["history"]=[{"role":"assistant","content":state["sql_query"]}]
        with open("text_to_sql.json","w") as text_to_sql_json:
            json.dump(state,text_to_sql_json,indent=2)
        return state
    
    def run_workflow(self,question:str,thread_id:str="1"):
        input_state=WorkflowState(
            question=question,
            intent="",
            database_ddl="",
            tablename="",
            rephrased_question="", 
            semantic_info="",
            sql_query="", 
            query_result="", 
            query_error_message="",
            needs_clarification="", 
            visualization_data=""
        )
        print(input_state)


        DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            workflow=self._build_workflow()
            graph=workflow.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": "123"}}
            graph.invoke(input_state, config=config)


if __name__ == "__main__":
    workflow = TextToSQLWorkflow()
    final_state = workflow.run_workflow("what is the heightest unsafe events in a area")
    print(final_state)
