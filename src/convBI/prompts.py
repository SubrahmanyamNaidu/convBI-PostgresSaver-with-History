intent_prompt = [
    ("human", """Classify the user's intent based on their question and full conversation history.

Current question: {question}

Conversation history: {history}

Categories:
- general: General greetings, pleasantries, casual conversation (hi, hello, bye, thanks etc.)
- system_query: Questions about data, database queries, system information, get data, etc.

IMPORTANT: Use the conversation history to understand context. For example:
- If previous questions were about data and current question is "What about region X?", classify as system_query
- If this is a follow-up question referencing previous data queries, classify as system_query

Respond with only the category name (general or system_query)""")
]

greeting_prompt = [
    ("human", """Respond to the user's greeting or casual message in a friendly, professional manner.
Keep it brief and helpful.
User message: {question}""")
]

table_identification_prompt = [
    ("human", """Identify the table name which need to be queried to answer the question based on the DDL and previous conversation given below.
DDL: {ddl}
User question: {question}
previous conversation: {history}
just return the table name. no explanation needed.""")
]

text_to_sql_prompt = [
    ("human", """You are an expert SQL query generator. Convert the user's question into a SQL query using the provided semantic information and conversation history.

CONTEXT ANALYSIS:
- Current question: {question}
- Previous conversation: {history}
- Available semantic info: {semantic_info}

INSTRUCTIONS:
1. ANALYZE CONVERSATION FLOW:
   - If this is a follow-up question (contains words like "what about", "also show", "and for", "in that case", etc.), reference the previous query context
   - Look for implicit references to previously mentioned entities, filters, or conditions
   - Identify if the current question is asking for similar data but with different parameters

2. CONTEXT UNDERSTANDING:
   - If previous questions mentioned specific locations/regions/branches, consider if current question relates to them
   - If previous queries had certain filters (date ranges, event types, etc.), determine if they should be maintained or modified
   - Look for comparative questions ("compared to", "difference between", "also in", etc.)

3. QUERY CONSTRUCTION RULES:
   - Use exact column names from the semantic info provided
   - For follow-up questions, build upon previous query logic but modify conditions as needed
   - If the question references "that area/region/location" without naming it, use the location from previous context
   - For questions about "highest/most/top" events, use appropriate ORDER BY and LIMIT clauses
   - Handle aggregations appropriately (COUNT, SUM, MAX, MIN, AVG)

4. SEMANTIC MAPPING:
   - Map natural language terms to database columns using the semantic info
   - Handle synonyms and variations (e.g., "area" → city/region/zone, "events" → unsafe events)
   - Consider data types when building WHERE clauses

5. CONVERSATION CONTINUITY:
   - If previous query was about unsafe events in a specific location, and current question asks "what about region X", create a similar query for region X
   - If previous context established certain filters or conditions, maintain them unless explicitly changed
   - For comparative questions, structure queries to enable comparison

EXAMPLES OF FOLLOW-UP HANDLING:
- Previous: "Show unsafe events in Mumbai"
  Current: "What about Delhi?" 
  → Query for Delhi using same structure as Mumbai query

- Previous: "Count of events by type"
  Current: "And for last month?"
  → Same query but add date filter for last month

- Previous: "Events in North region"
  Current: "Show the highest ones"
  → Query for North region events ordered by severity/count

RESPONSE FORMAT:
- Return ONLY the SQL query
- No explanations, no markdown formatting, no ```sql blocks
- Ensure proper SQL syntax
- Include appropriate JOINs if multiple tables are referenced in semantic info

Generate the SQL query now:""")
]
prompt_ddl="""
unsafe_events_ei_tech (
    event_id INTEGER,
    reporter_name VARCHAR(255),
    manager_name VARCHAR(255),
    branch VARCHAR(255),
    reported_date DATE,
    reporter_id VARCHAR(100),
    date_of_unsafe_event DATE,
    time VARCHAR(50),
    time_of_unsafe_event VARCHAR(50),
    unsafe_event_type VARCHAR(255),
    business_details VARCHAR(255),
    site_reference TEXT,
    unsafe_event_location VARCHAR(255),
    product_type VARCHAR(255),
    employee_id VARCHAR(100),
    employee_name VARCHAR(255),
    subcontractor_company_name VARCHAR(255),
    subcontractor_id VARCHAR(100),
    subcontractor_city VARCHAR(255),
    subcontractor_name VARCHAR(255),
    kg_name VARCHAR(100),
    country_name VARCHAR(100),
    division VARCHAR(255),
    department VARCHAR(255),
    city VARCHAR(255),
    sub_area VARCHAR(100),
    district VARCHAR(255),
    zone VARCHAR(255),
    serious_near_miss VARCHAR(10),
    unsafe_act VARCHAR(255),
    unsafe_act_other TEXT,
    unsafe_condition TEXT,
    unsafe_condition_other TEXT,
    work_stopped VARCHAR(10),
    stop_work_nogo_violation VARCHAR(255),
    nogo_violation_detail TEXT,
    stop_work_duration VARCHAR(255),
    other_safety_issues TEXT,
    comments_remarks TEXT,
    event_requires_sanction VARCHAR(10),
    action_description_1 TEXT,
    action_description_2 TEXT,
    action_description_3 TEXT,
    action_description_4 TEXT,
    action_description_5 TEXT,
    image VARCHAR(500),
    status VARCHAR(50),
    region VARCHAR(100),
    db_uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

unsafe_events_srs (
    event_id VARCHAR(100),
    reporter_name VARCHAR(255),
    reported_date DATE,
    reporter_id VARCHAR(100),
    date_of_unsafe_event DATE,
    time_of_unsafe_event VARCHAR(50),
    unsafe_event_type VARCHAR(255),
    business_details VARCHAR(255),
    site_reference TEXT,
    unsafe_event_location VARCHAR(255),
    product_type VARCHAR(255),
    employee_id VARCHAR(100),
    employee_name VARCHAR(255),
    subcontractor_company_name VARCHAR(255),
    subcontractor_id VARCHAR(100),
    subcontractor_city VARCHAR(255),
    subcontractor_name VARCHAR(255),
    kg_name VARCHAR(100),
    country_name VARCHAR(100),
    division VARCHAR(255),
    department VARCHAR(255),
    city VARCHAR(255),
    sub_area VARCHAR(100),
    district VARCHAR(255),
    zone VARCHAR(255),
    serious_near_miss VARCHAR(10),
    unsafe_act TEXT,
    unsafe_act_other TEXT,
    unsafe_condition TEXT,
    unsafe_condition_other TEXT,
    work_stopped VARCHAR(10),
    stop_work_nogo_violation VARCHAR(10),
    nogo_violation_detail TEXT,
    stop_work_duration VARCHAR(255),
    other_safety_issues TEXT,
    comments_remarks TEXT,
    event_requires_sanction VARCHAR(10),
    action_description_1 TEXT,
    action_description_2 TEXT,
    action_description_3 TEXT,
    action_description_4 TEXT,
    action_description_5 TEXT,
    branch VARCHAR(255),
    region VARCHAR(100),
    db_uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

unsafe_events_ni_tct (
    reporting_id INTEGER,
    status_key VARCHAR(50),
    status VARCHAR(50),
    location_key INTEGER,
    location VARCHAR(255),
    branch_key INTEGER,
    no INTEGER,
    branch_name VARCHAR(255),
    region_key VARCHAR(100),
    region VARCHAR(100),
    reporter_sap_id VARCHAR(100),
    reporter_name VARCHAR(255),
    designation_key VARCHAR(50),
    designation VARCHAR(255),
    gl_id_key VARCHAR(100),
    gl_id VARCHAR(255),
    pe_id_key VARCHAR(100),
    pe_id VARCHAR(255),
    created_on TIMESTAMP,
    date_and_time_of_unsafe_event TIMESTAMP,
    type_of_unsafe_event_key VARCHAR(50),
    type_of_unsafe_event VARCHAR(255),
    unsafe_event_details_key VARCHAR(50),
    unsafe_event_details TEXT,
    action_related_to_high_risk_situation_key VARCHAR(10),
    action_related_to_high_risk_situation VARCHAR(10),
    business_details_key VARCHAR(50),
    business_details VARCHAR(255),
    site_name TEXT,
    site_reference_key VARCHAR(50),
    site_reference VARCHAR(255),
    product_type_key VARCHAR(50),
    product_type VARCHAR(255),
    persons_involved TEXT,
    work_was_stopped_key VARCHAR(10),
    work_was_stopped VARCHAR(10),
    work_stopped_hours VARCHAR(50),
    no_go_violation_key VARCHAR(50),
    no_go_violation VARCHAR(255),
    job_no VARCHAR(100),
    additional_comments TEXT,
    has_attachment BOOLEAN,
    attachment VARCHAR(500),
    db_uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

clarification_prompt = [
    ("human", """Based on the user's question and the error message, ask user to provide more information. It shouldn't be techinical like asking for column names.
User question: {question}
Error Message: {error_message}
previous conversation: {history}
Respond with only the rephrased question. no explanation needed.""")
]

summarizer_prompt= [
    ("human", """Summarize the query result based on the user's question and conversation context.

User question: {question}
Query result: {query_result}
Previous conversation: {history}
table name: {tablename}

IMPORTANT: Use conversation history to provide contextual answers:

1. **Table Name Mapping:**
   - unsafe_events_ei_tech → "ei tech"
   - unsafe_events_ni_tct → "ni tct"
   - unsafe_events_srs → "srs"

2. **Context-Aware Responses:**
   - If this is a follow-up question (like "what about srs?"), acknowledge the comparison
   - Reference previous results when relevant
   - Use natural language that flows from the conversation

3. **Examples:**
   Previous: "User: Count events in ei tech\nAssistant: There are 1,234 events in ei tech"
   Current question: "what about srs?"
   Current result: [{{'count': 567}}]
   → Response: "There are 567 unsafe events in srs (compared to 1,234 in ei tech)."

   Previous: "User: Show by region in ei tech\nAssistant: North: 45, South: 32 events in ei tech"
   Current question: "same for srs"
   Current result: [{{'region': 'North', 'count': 23}}, {{'region': 'South', 'count': 18}}]
   → Response: "For srs: North region has 23 events, South region has 18 events."

4. **Response Guidelines:**
   - Be conversational and reference context when appropriate
   - Always mention the correct system name (ei tech, srs, ni tct)
   - Keep responses concise but informative
   - Make comparisons when the conversation suggests it

Respond with only the summary. No explanation needed.""")
]


summarizer_prompt_2=[("human","""
You are an AI assistant providing summarized answers based on data processed through the text-to-SQL dataframe. Your task is to summarize the relevant safety data and provide data-driven answers aligned with the Safety Head or General Manager of Safety & Health role.
Key Guidelines:
Summarize Data: After processing, provide a concise summary of the key patterns, trends, and insights most relevant for leadership.
Focus on Organizational Impact: Frame your answers in terms of safety strategy, risk management, and business implications.
Data-Driven Answers: Craft answers that are rooted in the data, highlighting critical trends and organizational risks.
Be Concise and Clear: Keep responses clear, professional, and strategic, focusing on what’s relevant for leadership decision-making.
Example:
“Region X has seen a 30% increase in unsafe events over the past quarter, mostly from Division Y, indicating a recurring issue in that area.”
“A 25% rise in no-go violations was observed in Region Z, suggesting potential operational bottlenecks in high-risk locations.

User question: {question}
Query result: {query_result}
Previous conversation: {history}
table name: {tablename}
""")]