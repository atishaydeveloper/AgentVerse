import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import litellm


# Load environment variables
load_dotenv()
search_tool = SerperDevTool(n_results=10)
# Define agents with required fields

agents = {
    "Researcher": Agent(
        name="Researcher",
        role="Lead Research Scientist",
        goal="Uncover cutting-edge developments in {topic}",
        backstory="You're a seasoned researcher with a knack for uncovering the latest developments in {topic}. "
              "Known for your ability to find the most relevant information and present it in a clear and concise manner. "
              "You have a deep understanding of research methodologies and are skilled at using various research tools and databases. "
              "Your analytical skills allow you to sift through large amounts of data to identify key insights and trends. "
              "You are committed to staying up-to-date with the latest advancements in your field and are always looking for new ways to improve your research techniques.",
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    ),
    "Reporting Analyst": Agent(
        name="Reporting Analyst",
        role="Reporting Analyst",
        goal="Create detailed reports based on {topic} data analysis and research findings",
        backstory="You're a meticulous analyst with a keen eye for detail. You're known for your ability to turn complex data "
                "into clear and concise reports, making it easy for others to understand and act on the information you provide. "
                "You have a strong background in data analysis and are proficient in using various analytical tools and software. "
                "Your reports are well-structured and include comprehensive data visualizations that highlight key findings. "
                "You are skilled at identifying patterns and trends in data and are able to provide actionable insights that drive decision-making.",
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    ),
    "Writer": Agent(
        name="Writer",
        role="Creative Writer",
        goal="Craft compelling and engaging written content about {topic}",
        backstory="You're a passionate writer with a flair for storytelling. You have a knack for transforming ideas into words "
                "that captivate readers and evoke emotions, making {topic} come alive in your writing. "
                "You have a strong command of language and are skilled at writing in various styles and tones. "
                "Your writing is engaging and persuasive, and you have a talent for making complex topics accessible to a wide audience. "
                "You are always looking for new ways to connect with your readers and are committed to producing high-quality content that resonates with them.",
        
        allow_delegation=False,
        verbose=True,        
    ),
    "Editor": Agent(
        name="Editor",
        role="Content Editor",
        goal="Refine and polish written content to ensure clarity, coherence, and quality",
        backstory="You're an expert editor with an eye for detail and a commitment to excellence. "
                "Known for your ability to enhance content without losing its original essence, you ensure that every piece about {topic} is flawless and impactful. "
                "You have a strong background in editing and are proficient in using various editing tools and software. "
                "Your editing process is thorough and meticulous, and you are skilled at identifying and correcting errors in grammar, punctuation, and style. "
                "You are committed to maintaining the highest standards of quality in your work and are always looking for ways to improve the clarity and coherence of the content you edit.",
        
        allow_delegation=False,
        verbose=True,
    ),
    "Event Planner": Agent(
        name="Event Planner",
        role="Event Coordinator",
        goal="Plan and organize successful events centered around {topic}",
        backstory="You're a seasoned event planner with a talent for turning visions into reality. "
                "Known for your creativity and attention to detail, you excel at orchestrating events that leave a lasting impression on attendees. "
                "You have a strong background in event planning and are proficient in using various event management tools and software. "
                "Your events are well-organized and run smoothly, and you are skilled at managing all aspects of event planning, from budgeting and scheduling to logistics and coordination. "
                "You are committed to creating memorable experiences for attendees and are always looking for new ways to make your events more engaging and impactful.",
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    ),
    "Risk Analyst": Agent(
        name="Risk Analyst",
        role="Risk Analyst",
        goal="Identify, assess, and mitigate potential risks associated with {topic}",
        backstory="You're a detail-oriented risk analyst with expertise in identifying potential pitfalls and crafting strategies to minimize impact. "
                "Your insights are critical for ensuring success in initiatives related to {topic}. "
                "You have a strong background in risk analysis and are proficient in using various risk management tools and software. "
                "Your risk assessments are thorough and comprehensive, and you are skilled at identifying and analyzing potential risks and their impact. "
                "You are committed to developing effective risk mitigation strategies and are always looking for new ways to improve your risk management processes.",
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    ),
    "Content Planner": Agent(
        name="Content Planner",
        role="Content Strategist",
        goal="Develop and implement a comprehensive content plan for {topic}",
        backstory="You're a strategic thinker with a deep understanding of audience engagement. "
                "Known for your ability to design content strategies that align with objectives, you ensure that every piece of content about {topic} serves its purpose effectively. "
                "You have a strong background in content planning and are proficient in using various content management tools and software. "
                "Your content plans are well-structured and include detailed content calendars, target audience analysis, and key performance indicators. "
                "You are skilled at identifying content opportunities and are committed to creating content that resonates with your audience and drives engagement.",
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    )
}

# Define tasks
tasks = {
    "Researcher": Task(
        description=("""1. Conduct comprehensive research on {topic} including:
            - Recent developments and news
            - Key industry trends and innovations
            - Expert opinions and analyses
            - Statistical data and market insights
        2. Evaluate source credibility and fact-check all information
        3. Organize findings into a structured research brief
        4. Include all relevant citations and sources
        5. Ensure the research covers multiple perspectives and is unbiased
        6. Highlight any potential future trends or predictions related to the topic
    """),
        expected_output="""A detailed research report containing:
        - Executive summary of key findings
        - Comprehensive analysis of current trends and developments
        - List of verified facts and statistics
        - All citations and links to original sources
        - Clear categorization of main themes and patterns
        - Visual aids such as charts or graphs to support data
        - Potential future trends or predictions
        Please format with clear sections and bullet points for easy reference.""",
        agent=agents["Researcher"]
    ),
    
    "Reporting Analyst" : Task(
        description=("""Review the context you got and expand each topic into a full section for a report.
        Make sure the report is detailed and contains any and all relevant information.
        Ensure the report is well-structured and logically organized.
        Include any relevant case studies or real-world examples to support the information.
        """),
        expected_output="""A fully fledged report with the main topics, each with a full section of information.
        Formatted as markdown without '```'.
        Each section should include:
            - An introduction to the topic
            - Detailed analysis and discussion
            - Relevant case studies or examples
            - Conclusion summarizing the key points
        """,
        agent=agents["Reporting Analyst"]
    ),

    "Event Planner" : Task(
        description=("""Create a detailed plan for an event focused on {topic}.
        Include the agenda, list of activities, resource requirements, and a proposed timeline.
        Consider potential challenges and include contingency plans.
        """),
        expected_output="""A comprehensive event plan document, formatted as markdown without '```',
        with a clear agenda, budget estimation, and a timeline for execution.
        The event plan should include:
            - A detailed agenda with time slots for each activity
            - List of required resources and their estimated costs
            - Proposed timeline for planning and execution
            - Contingency plans for potential challenges
        """,
        agent=agents["Event Planner"]
    ),
    
    "Content Planner" : Task(
        description=("""Create a detailed plan for an event focused on {topic}.
        Include the agenda, list of activities, resource requirements, and a proposed timeline.
        Consider potential challenges and include contingency plans.
        """),
        expected_output="""A comprehensive event plan document, formatted as markdown without '```',
        with a clear agenda, budget estimation, and a timeline for execution.
        The event plan should include:
            - A detailed agenda with time slots for each activity
            - List of required resources and their estimated costs
            - Proposed timeline for planning and execution
            - Contingency plans for potential challenges
        """,
        agent=agents["Content Planner"]
    ),
    "Writer": Task(
        description=("""Write an engaging and creative article about {topic}.
        Ensure the tone aligns with the intended audience and purpose.
        Use storytelling techniques to make the article captivating.
        Include quotes from experts or relevant personalities if possible.
        """),
        expected_output="""A well-written article of approximately 1000 words, formatted as markdown without '```',
        with an introduction, body, and conclusion that captivates readers.
        The article should include:
            - A compelling introduction that hooks the reader
            - A well-structured body with clear subheadings
            - Quotes or insights from experts
            - A conclusion that reinforces the main points and provides a call to action or thought-provoking statement
        """,
        agent=agents["Writer"]
    ),
    
    "Editor" : Task(
        description=("""Edit and refine the draft content related to {topic}.
        Ensure the content is clear, concise, error-free, and adheres to the required style guide.
        Provide constructive feedback and suggestions for further improvements.
        """),
        expected_output="""A polished version of the content with all necessary corrections and enhancements.
        Comments and suggestions should be provided where applicable for further improvements.
        The edited content should:
            - Be free of grammatical and spelling errors
            - Have improved clarity and readability
            - Adhere to the specified style guide
            - Include any additional suggestions for enhancing the content
        """,
        agent=agents["Editor"]
    ),
    
    "Risk_Analyst" : Task(
        description=("""Conduct a thorough risk assessment for {topic}.
        Identify potential risks, analyze their impact, and suggest mitigation strategies.
        Include both short-term and long-term risks.
        """),
        expected_output="""A risk assessment report formatted as markdown without '```',
        listing potential risks, their likelihood and impact, and recommended solutions.
        The report should include:
            - A table or list of identified risks
            - Analysis of the likelihood and potential impact of each risk
            - Suggested mitigation strategies for each risk
            - Consideration of both short-term and long-term risks
        """,
        agent=agents['Risk Analyst']
    )
    
    
    
    # Add other tasks here...
}


# Streamlit page config
st.set_page_config(page_title="AgentVerse", page_icon="🌐", layout="wide")

st.markdown("""
    <style>
    /* Sidebar Styling */
    .stSidebar {
        background-color: #2c3e50; /* Dark blue-gray for a modern look */
        color: white; /* White text for better readability */
    }

    /* Button Styling */
    .stButton {
        background-color: #3498db; /* Bright blue for the button */
        color: white; /* White text for visibility */
        border-radius: 5px; /* Slight rounded corners */
        font-weight: bold; /* Emphasize the button text */
    }

    /* Button Hover Effect */
    .stButton:hover {
        background-color: #2980b9; /* Darker blue for hover effect */
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    
    /* Submit Button Specific Styling */
    .stButton[data-baseweb="button"] {
        background-color: #e74c3c; /* Red background for the submit button */
        color: white; /* White text for visibility */
        font-size: 16px; /* Slightly larger font for emphasis */
        padding: 12px 30px; /* Adjust padding for the submit button */
        border-radius: 8px; /* More rounded corners */
    }

    /* Main Title Styling */
    h1 {
        color: #e74c3c; /* Bold red for the title */
        font-size: 36px; /* Larger font size */
        text-align: center; /* Center the title */
    }

    /* Sidebar Header Styling */
    .stSidebar h2 {
        color: #ecf0f1; /* Light gray for sidebar headers */
    }

    /* General Text Styling */
    .stMarkdown {
        font-size: 18px; /* Slightly larger text for readability */
        color: #ecf0f1; /* Light gray text for readability */
        line-height: 1.6; /* Better spacing for easier reading */
    }

    /* Input fields Styling */
    .stTextInput, .stTextArea, .stMultiSelect {
        background-color: #34495e; /* Darker background for input fields */
        border-radius: 5px; /* Slight rounded corners */
        padding: 10px; /* Add padding for better input field appearance */
        color: white; /* White text for better visibility */
    }

    /* Placeholder Text */
    .stTextInput::placeholder, .stTextArea::placeholder {
        color: #bdc3c7; /* Lighter color for placeholder text */
    }

    /* Select and Multi-Select Dropdown Styling */
    .stMultiSelect {
        background-color: #34495e; /* Darker background for multi-select */
        border-radius: 5px;
        padding: 10px;
        color: white; /* White text */
    }

    /* Sidebar Text Styling */
    .stSidebar .stMarkdown {
        color: #ecf0f1; /* Ensure visibility of text in sidebar */
    }

    /* Title in the Sidebar */
    .stSidebar h1 {
        color: #ecf0f1; /* Light gray for sidebar titles */
    }

    /* Text in the Buttons */
    .stButton p {
        color: white; /* White text inside buttons */
    }
    
    /* Make links more visible */
    a {
        color: #3498db; /* Blue for links */
    }
    a:hover {
        color: #2980b9; /* Darker blue for hover effect */
    }

    /* Ensure visibility of labels and inputs */
    .stTextInput label, .stTextArea label, .stMultiSelect label {
        color: white; /* White text for labels */
    }
    </style>
""", unsafe_allow_html=True)




# Title and description
st.title("**AgentVerse**:")
st.title("Your Multi-Agent Content Creator 🤖")
st.markdown("---")
st.markdown("Explore the power of multi-agent AI to craft content tailored to your needs.")



# Sidebar
with st.sidebar:
    st.header("Content Settings")
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area below
        2. Select the agents you want to use for content creation
        3. Adjust the temperature if needed (higher = more creative)
        4. Click 'Generate Content' to start
        5. Wait for the AI to generate your article
        6. Download the result as a markdown file
        """)
    # Text area for topic input
    
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )
    st.markdown("**Example Topics:** Technology Trends, AI in Healthcare, Future of Education")

    

    
    # Multi-select for agent selection
    selected_agents = st.multiselect(
        "Select agents to use:",
        options=list(agents.keys()),
        help="Hover over an agent to learn about its capabilities.",
    )
    
    
    agentaa = {
    "Researcher": {
        "goal": f"Conduct thorough research to gather relevant information for {topic}."
    },
    "Reporting Analyst": {
        "goal": f"Analyze data and create insightful reports on {topic}."
    },
    "Writer": {
        "goal": f"Write engaging and informative content about {topic}."
    },
    "Editor": {
        "goal": "Edit and refine content to ensure clarity and quality."
    },
    "Event Planner": {
        "goal": f"Organize and plan events related to {topic}."
    },
    "Risk Analyst": {
        "goal": f"Assess and mitigate potential risks related to {topic}."
    },
    "Content Planner": {
        "goal": f"Develop and implement a comprehensive content plan for {topic}."
    }
}
    
    
    if selected_agents:
        st.sidebar.markdown("### Tasks for Selected Agents")
        for agent in selected_agents:
            st.sidebar.markdown(f"- {agentaa.get(agent, {}).get('goal', 'No tasks available')}")



    # Map selected agent names to agent objects
    selected_agent_objects = [agents[agent] for agent in selected_agents]
    
    # Map selected agent names to their respective tasks
    selected_task_objects = [tasks[agent] for agent in selected_agents if agent in tasks]
    
    print("Selected agents are:", selected_agent_objects)
    print("Selected tasks are:", selected_task_objects)
    
    # Add more sidebar controls if needed
    st.markdown("### Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Make the generate button more prominent in the sidebar
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)
    

import os

# Set environment variable for debug logs
os.environ['LITELLM_LOG'] = 'DEBUG'

os.environ['GEMINI_API_KEY'] = 'AIzaSyBwdyO2efl6XzXKeEHv3E8PIdHnpAt5wJU'

# Function to generate content based on selected agents and tasks

def generate_content(topic, selected_agents, selected_tasks):
    
    
    llm = LLM(
        model="gemini/gemini-1.5-pro",
        api_key=os.environ['GEMINI_API_KEY']
        
    )
    

    # Create Crew with selected agents and tasks
    crew = Crew(
        agents=selected_agents,
        tasks=selected_tasks,
        verbose=True
    )
    return crew.kickoff(inputs={"topic": topic})


import time

# Main content area
if generate_button:
    with st.spinner('Generating content...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
        try:
            result = generate_content(topic, selected_agent_objects, selected_task_objects)
            st.markdown("### Generated Content")
            st.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
            
            with st.expander("View Generated Content"):
                st.markdown(result)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
          
    st.markdown("[Share on Twitter](https://twitter.com/intent/tweet?text=Check+out+this+amazing+content+generated+by+AgentVerse!)")
    st.markdown("[Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https://yourwebsite.com&title=Check+out+this+amazing+content+generated+by+AgentVerse!&summary=Generated+content+on+AgentVerse.)")

    # Footer
    rating = st.slider("Rate the generated content:", 1, 5)
    

    
