#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai_tools import BraveSearchTool
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
load_dotenv()

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class OpenRouterAgent:
    """OpenRouter-powered AI Agent using CrewAI framework"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.setup_environment()
        self.llm = self.create_llm()
        self.search_tool = BraveSearchTool()
        self.scrape_tool = ScrapeWebsiteTool()

    def setup_environment(self):
        """Setup OpenRouter environment variables"""
        if not self.api_key:
            print(f"{Colors.FAIL}✗ OPENAI_API_KEY not found in environment variables{Colors.ENDC}")
            print(f"{Colors.WARNING}Please add your OpenRouter API key to .env file:{Colors.ENDC}")
            print(f"{Colors.OKCYAN}OPENAI_API_KEY=your_openrouter_api_key_here{Colors.ENDC}")
            exit(1)

        print(f"{Colors.OKGREEN}✓ OpenRouter environment configured{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Using model: openrouter/qwen/qwen3-coder:free{Colors.ENDC}")

    def create_llm(self):
        """Create LLM connection to OpenRouter"""
        return LLM(
            model="openrouter/qwen/qwen3-coder",  # Add openrouter/ prefix
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

    def test_llm_connection(self):
        """Test if LLM connection is working"""
        try:
            print(f"{Colors.OKCYAN}Testing LLM connection...{Colors.ENDC}")

            # Create a simple test agent
            test_agent = Agent(
                role="Test Agent",
                goal="Test connection",
                backstory="Simple test agent",
                llm=self.llm,
                verbose=False
            )

            test_task = Task(
                description="Say 'Connection successful' in exactly those two words.",
                expected_output="The phrase 'Connection successful'",
                agent=test_agent
            )

            crew = Crew(agents=[test_agent], tasks=[test_task], verbose=False)
            result = crew.kickoff()

            print(f"{Colors.OKGREEN}✓ LLM connection test passed{Colors.ENDC}")
            return True

        except Exception as e:
            print(f"{Colors.FAIL}✗ LLM connection test failed{Colors.ENDC}")
            print(f"{Colors.WARNING}Error details: {str(e)}{Colors.ENDC}")
            return False

    def solve_problem(self, problem: str) -> str:
        """
        Solve a problem using the OpenRouter-powered agent
        """
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🤖 OpenRouter Agent Problem Solving")
        print("=" * 40)
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Problem: {problem}{Colors.ENDC}")
        print()

        # Create the agent
        agent = Agent(
            role="Intelligent Problem Solver",
            goal=f"Analyze and solve the given problem: {problem}",
            backstory="""You are an intelligent AI agent powered by Qwen3-Coder model. 
            You excel at breaking down complex problems, conducting web research when needed, 
            and providing comprehensive solutions. You can search the internet and scrape 
            websites to gather current information when the problem requires it.""",
            llm=self.llm,
            tools=[self.search_tool, self.scrape_tool],
            verbose=True,
            allow_delegation=False
        )

        # Create the task
        task = Task(
            description=f"""
            Analyze the problem: "{problem}"

            Follow these steps:
            1. Determine if the problem requires web search for current information or can be solved with reasoning
            2. If web search is needed, use the search tool to find relevant information
            3. If you find useful URLs, use the scrape tool to get detailed content
            4. Synthesize all information into a comprehensive answer
            5. Provide a clear, actionable solution

            Always provide a complete answer that directly addresses the user's question.
            """,
            expected_output="A comprehensive solution that directly answers the problem with clear explanations and actionable information.",
            agent=agent
        )

        # Create and run the crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        print(f"{Colors.OKCYAN}🚀 Starting problem solving...{Colors.ENDC}")
        result = crew.kickoff()

        print(f"\n{Colors.OKGREEN}")
        print("=" * 50)
        print("✅ PROBLEM SOLVED!")
        print("=" * 50)
        print(f"{Colors.ENDC}")

        return str(result)


def main():
    """Main execution function"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🌟 OpenRouter AI Agent")
    print("Powered by Qwen3-Coder Free Model")
    print("=" * 40)
    print(f"{Colors.ENDC}")

    # Initialize the agent
    agent = OpenRouterAgent()

    # Test LLM connection first
    if not agent.test_llm_connection():
        print(
            f"{Colors.FAIL}❌ Unable to establish LLM connection. Please check your API key and try again.{Colors.ENDC}")
        return

    print(f"{Colors.OKGREEN}🎉 Agent ready for problem solving!{Colors.ENDC}")

    # Interactive problem solving
    while True:
        print(f"\n{Colors.OKCYAN}Enter your problem or question (type 'exit' to quit):{Colors.ENDC}")
        problem = input(f"{Colors.WARNING}> {Colors.ENDC}")

        if problem.lower() in ['exit', 'quit', 'q']:
            print(f"{Colors.OKGREEN}Goodbye! 👋{Colors.ENDC}")
            break

        if problem.strip():
            result = agent.solve_problem(problem)
            print(f"\n{Colors.OKGREEN}🎯 Solution:{Colors.ENDC}")
            print(result)
        else:
            print(f"{Colors.WARNING}Please enter a valid problem or question.{Colors.ENDC}")


if __name__ == "__main__":
    main()