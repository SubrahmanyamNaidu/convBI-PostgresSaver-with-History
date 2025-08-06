from dotenv import load_dotenv
load_dotenv()

import os 

from langgraph.graph import StateGraph,START,END 
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated,Dict,Any


from prompts import intent_prompt,greeting_prompt,table_identification_prompt,prompt_ddl,text_to_sql_prompt,clarification_prompt,summarizer_prompt
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
    final_answer:str

class TextToSQLWorkflow:
    def __init__(self):
        self.llm=AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    
    def _serialize_state_for_json(self, state: WorkflowState) -> Dict[str, Any]:
        """Helper method to serialize state for JSON output, handling non-serializable objects"""
        serializable_state = {}
        for key, value in state.items():
            if key == "history" and value:
                # Convert AIMessage objects to serializable format
                serializable_history = []
                for msg in value:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        serializable_history.append({
                            "type": msg.type,
                            "content": msg.content
                        })
                    else:
                        serializable_history.append(str(msg))
                serializable_state[key] = serializable_history
            else:
                serializable_state[key] = value
        return serializable_state

    def _build_workflow(self)->StateGraph[WorkflowState]:
        graph_builder=StateGraph(WorkflowState)
        graph_builder.add_node("intent_classification",self._intent_classification_agent)
        graph_builder.add_node("greeting",self._greeting_agent)
        graph_builder.add_node("database_ddl",self._database_ddl_agent)
        graph_builder.add_node("table_identification",self._table_identification_agent)
        graph_builder.add_node("table_semantics_info",self._table_semantics_info_agent)
        graph_builder.add_node("text_to_sql",self._text_to_sql_agent)
        graph_builder.add_node("execute_sql_query", self._execute_sql_query)
        graph_builder.add_node("summarizer", self._summarizer_agent)
        graph_builder.add_node("clarification_agent", self._clarification_agent)
        graph_builder.add_node("visualization",self._visualization_agent)
        

        graph_builder.add_edge(START,"intent_classification")
        graph_builder.add_conditional_edges(
            "intent_classification",
            lambda state: state["intent"]=="general",
            {True:"greeting",False:"database_ddl"}
            )
        
        graph_builder.add_edge("database_ddl","table_identification")
        graph_builder.add_edge("table_identification","table_semantics_info")
        graph_builder.add_edge("table_semantics_info","text_to_sql")
        graph_builder.add_edge("text_to_sql","execute_sql_query")
        graph_builder.add_conditional_edges(
            "execute_sql_query",
            lambda state:state["needs_clarification"]==True,
            {True:"clarification_agent",False:"summarizer"}
            )
        graph_builder.add_edge("summarizer", "visualization")
        graph_builder.add_edge("visualization",END)
        graph_builder.add_edge("greeting",END)
        # graph_builder.add_edge("text_to_sql",END)

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
        
        # Use the helper method to serialize state for JSON
        try:
            serializable_state = self._serialize_state_for_json(state)
            with open("intent.json","w") as intent_json:
                json.dump(serializable_state, intent_json, indent=2)
        except Exception as e:
            print(f"Warning: Could not save intent.json: {e}")

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
        try:
            with open("test.semantics.json","r") as semantics_json:
                semantics=json.load(semantics_json)
            
            required_table_semantics=semantics[state["tablename"]]
            state["semantic_info"]=required_table_semantics
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load semantic info: {e}")
            state["semantic_info"] = {}

        return state
    
    def _text_to_sql_agent(self,state:WorkflowState)->WorkflowState:
        prompt=ChatPromptTemplate.from_messages(text_to_sql_prompt)

        prev_conv=state["history"][-6:] if state["history"] else []
        # print("="*8)
        # print(prev_conv)
        # print("="*6)
        chain=prompt|self.llm
        result=chain.invoke({
            "semantic_info":state["semantic_info"] ,
            "question":state["question"],
            "history":prev_conv
        })

        state["sql_query"]=result.content.strip()



        state["history"] = [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["sql_query"])
        ]
        
        # Use the helper method to serialize state for JSON
        # try:
        #     serializable_state = self._serialize_state_for_json(state)
        #     with open("text_to_sql.json","w") as text_to_sql_json:
        #         json.dump(serializable_state, text_to_sql_json, indent=2)
        # except Exception as e:
        #     print(f"Warning: Could not save text_to_sql.json: {e}")
            
        return state
    
    def _execute_sql_query(self, state: WorkflowState) -> WorkflowState:
        try:
            conn = self._get_db_connection()
            if not conn:
                raise Exception("Could not establish database connection")
            
            cursor = conn.cursor()
            cursor.execute(state["sql_query"])
           
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            formatted_results = []
            for row in results:
                row_dict = dict(zip(columns, row))
                formatted_results.append(row_dict)
            
            state["query_result"] = str(formatted_results)
            # Optimize state by storing only essential query info
            # state["history"] = [{"role":"system", "content":f"query_result_count: {len(results)}"}]
            state["needs_clarification"] = False
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            state["error_message"] = str(e)
            state["needs_clarification"] = True
        return state

    def _get_db_connection(self):
        try:
            DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            connection = psycopg.connect(DATABASE_URL)
            return connection
        except psycopg.Error as e:
            return None  
    
    def _summarizer_agent(self, state: WorkflowState) -> WorkflowState:

        
        prompt = ChatPromptTemplate.from_messages(summarizer_prompt)
        # Optimize history to reduce state size
        prez_conv = state["history"][-1:] if state["history"] else []
        chain = prompt | self.llm
        result = chain.invoke({
            "question": state["question"],
            "history": prez_conv,
            "query_result": state["query_result"],
            "tablename": state["tablename"]
        })
        state["final_answer"] = result.content.strip().lower()
        state["history"] = [
            AIMessage(content=state["final_answer"])
        ]
        

        return state
    
    def _clarification_agent(self, state: WorkflowState) -> WorkflowState:
        prompt = ChatPromptTemplate.from_messages(clarification_prompt)
        prez_conv=state["history"]
        if len(state["history"])>2:
            prez_conv=state["history"][-2:]
        chain = prompt | self.llm
        result = chain.invoke({
            "question": state["question"],
            "history": prez_conv,
            "error_message": state["error_message"]
        })
        state["final_answer"] = result.content.strip().lower()
        
        return state
    
    def _visualization_agent(self, state: WorkflowState) -> WorkflowState:
        
        """
        This agent uses GPT to generate a JSON for Apache ECharts based on the summary data.
        It creates a chart configuration in the ECharts JSON format.
        """
        question = state["question"]
        query_result = state["query_result"]
        
        # Assuming the query result is a list of dictionaries, let's pass it along with the question to GPT.
        try:
            # Parse the result (assuming it's a list of dictionaries)
            results = query_result

            # Now prompt GPT to generate the ECharts JSON for visualization
            prompt = ChatPromptTemplate.from_template(
                """
                Based on the following question and the query result data, generate an ECharts JSON  configuration for a chart:
                previous conversation: {history}

                Question: {question}
                Query Result Data (Assuming it's a list of dictionaries with column names and values): {query_result}

                Generate a JSON in the ECharts format suitable for a bar chart, line chart, or pie chart, depending on the question. Include any necessary configuration like xAxis, yAxis, series, tooltip, etc.
                #Instruction
                - Do generate Echarts only if it makes meaningful to generate chart based on the Question and Query Result Data
                - Respon with JSON no extra information/explanation need.
                - Don't add ```json or ``` in the output 
                - if you feel chat makes no meaning for the give Question and Query Result Data just return empty json curly braces
                """
            )

            chain = prompt | self.llm  # Assuming `self.llm` is already initialized as AzureChatOpenAI
            # Optimize history to reduce state size
            prez_conv = state["history"][-1:] if state["history"] else []

            result = chain.invoke({
                "question": question,
                "query_result":results, # Pass the results as JSON string to GPT
                "history": prez_conv
            })
            # Parse the output and save the JSON to state

            state["visualization_data"] = json.loads(result.content.strip())  # Save the generated JSON
            
        except json.JSONDecodeError as e:
            state["error_message"] = f"Error generating visualization data: {e}"
            state["needs_clarification"] = True
            
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
            visualization_data="",
            final_answer=""
        )
        # print(input_state)

        DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            workflow=self._build_workflow()
            graph=workflow.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": "71"}}
            result = graph.invoke(input_state, config=config)
            return result  # Return the result instead of None


if __name__ == "__main__":
    workflow = TextToSQLWorkflow()
    final_state = workflow.run_workflow("in the ni tct")
    serializable_state = workflow._serialize_state_for_json(final_state)
    with open("text_to_sql.json","w") as text_to_sql_json:
        json.dump(serializable_state, text_to_sql_json, indent=2)
