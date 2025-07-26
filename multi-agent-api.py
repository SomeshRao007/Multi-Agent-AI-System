#!/usr/bin/env python3

import os
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()

import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dataclasses import dataclass
from enum import Enum
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai_tools import BraveSearchTool, FileReadTool, DirectoryReadTool


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ModelStrength(Enum):
    ANALYTICAL = "analytical_thinking"
    SEARCH = "internet_research"
    REASONING = "logical_reasoning"
    EXECUTION = "efficient_processing"


class ModelProvider(Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class ModelConfig:
    name: str
    model_id: str
    provider: ModelProvider
    strength: ModelStrength
    role: str
    description: str
    temperature: float
    context_window: int


class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def check_ollama_status(self) -> bool:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return response.status_code == 200

    def list_available_models(self) -> list:
        response = requests.get(f"{self.api_url}/tags")
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
        return []

    def check_model_availability(self, model_name: str) -> bool:
        available_models = self.list_available_models()
        return any(model_name in model for model in available_models)

    def pull_model_if_needed(self, model_name: str) -> bool:
        if self.check_model_availability(model_name):
            print(f"{Colors.OKGREEN}✓ Model {model_name} already available{Colors.ENDC}")
            return True

        print(f"{Colors.WARNING}⚠ Model {model_name} not found. Downloading...{Colors.ENDC}")
        print(f"{Colors.OKCYAN}This may take several minutes depending on model size{Colors.ENDC}")

        import subprocess
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ Successfully downloaded {model_name}{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}✗ Failed to download {model_name}: {result.stderr}{Colors.ENDC}")
            return False


class ConditionalMultiModelCrew:
    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.openrouter_api_key = os.getenv("OPENAI_API_KEY")
        self.models_config = self._setup_model_configurations()
        self.llms = {}
        self.agents = {}
        self.knowledge_base = {}
        self.search_tool = BraveSearchTool()
        self.scrape_tool = ScrapeWebsiteTool()

    def _setup_model_configurations(self) -> Dict[str, ModelConfig]:
        return {
            "analyst": ModelConfig(
                name="Qwen Analyst",
                model_id="qwen3:4b",
                provider=ModelProvider.OLLAMA,
                strength=ModelStrength.ANALYTICAL,
                role="Problem Analyst",
                description="Problem analysis and routing decisions",
                temperature=0.3,
                context_window=32768
            ),
            "searcher": ModelConfig(
                name="Qwen3 Search Agent",
                model_id="openrouter/qwen/qwen3-coder:free",
                provider=ModelProvider.OPENROUTER,
                strength=ModelStrength.SEARCH,
                role="Internet Research Specialist",
                description="Advanced internet research and information gathering",
                temperature=0.6,
                context_window=128000
            ),
            "reasoner": ModelConfig(
                name="Qwen3 Reasoning Agent",
                model_id="qwen3:8b",
                provider=ModelProvider.OLLAMA,
                strength=ModelStrength.REASONING,
                role="Logical Reasoning Specialist",
                description="Pure logical reasoning without internet access",
                temperature=0.5,
                context_window=32768
            ),
            "synthesizer": ModelConfig(
                name="Gemma3 Synthesizer",
                model_id="gemma3:4b",
                provider=ModelProvider.OLLAMA,
                strength=ModelStrength.EXECUTION,
                role="Solution Synthesizer",
                description="Synthesizes information into final answers",
                temperature=0.6,
                context_window=32768
            )
        }

    def _check_openrouter_setup(self) -> bool:
        if not self.openrouter_api_key:
            print(f"{Colors.FAIL}✗ OPENAI_API_KEY not found in environment variables{Colors.ENDC}")
            print(f"{Colors.WARNING}Please add your OpenRouter API key to .env file:{Colors.ENDC}")
            print(f"{Colors.OKCYAN}OPENAI_API_KEY=your_openrouter_api_key_here{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Get your key from: https://openrouter.ai/keys{Colors.ENDC}")
            return False

        print(f"{Colors.OKGREEN}✓ OpenRouter API key found (length: {len(self.openrouter_api_key)}){Colors.ENDC}")
        return True

    def _test_openrouter_connection(self) -> bool:
        print(f"{Colors.OKCYAN}Testing OpenRouter connection...{Colors.ENDC}")

        test_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        test_payload = {
            "model": "openrouter/qwen/qwen3-coder:free",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=test_headers,
            json=test_payload,
            timeout=10
        )

        if response.status_code == 200:
            print(f"{Colors.OKGREEN}✓ OpenRouter connection successful{Colors.ENDC}")
            return True
        elif response.status_code == 401:
            print(f"{Colors.FAIL}✗ OpenRouter authentication failed - Invalid API key{Colors.ENDC}")
            return False
        else:
            print(f"{Colors.WARNING}⚠ OpenRouter returned status {response.status_code}{Colors.ENDC}")
            return False

    def initialize_system(self) -> bool:
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🚀 Initializing Conditional Multi-Model AI Crew")
        print("💰 Cost-Optimized: OpenRouter only for search tasks")
        print("=" * 55)
        print(f"{Colors.ENDC}")

        if not self._check_openrouter_setup():
            return False

        if not self._test_openrouter_connection():
            print(f"{Colors.WARNING}⚠ OpenRouter test failed - search tasks will not work{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠ Continuing with local models only (reasoning tasks will work){Colors.ENDC}")

        ollama_models = [k for k, v in self.models_config.items() if v.provider == ModelProvider.OLLAMA]
        if ollama_models:
            if not self.ollama_manager.check_ollama_status():
                print(f"{Colors.FAIL}✗ Ollama service not running{Colors.ENDC}")
                print(f"{Colors.WARNING}Please start Ollama with: ollama serve{Colors.ENDC}")
                return False

            print(f"{Colors.OKGREEN}✓ Ollama service is running{Colors.ENDC}")

            models_ready = True
            for key, config in self.models_config.items():
                if config.provider == ModelProvider.OLLAMA:
                    if not self.ollama_manager.pull_model_if_needed(config.model_id):
                        models_ready = False

            if not models_ready:
                print(f"{Colors.FAIL}✗ Some Ollama models failed to download{Colors.ENDC}")
                return False

        self._setup_llm_connections()

        print(f"{Colors.OKGREEN}✅ Conditional Multi-Model Crew initialized successfully!{Colors.ENDC}")
        return True

    def _setup_llm_connections(self):
        print(f"{Colors.OKCYAN}⚙ Setting up LLM connections...{Colors.ENDC}")

        for key, config in self.models_config.items():
            if config.provider == ModelProvider.OLLAMA:
                self.llms[key] = LLM(
                    model=f"ollama/{config.model_id}",
                    base_url="http://localhost:11434",
                    temperature=config.temperature,
                    max_tokens=4096,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                print(f"{Colors.OKGREEN}  ✓ {config.name} (Ollama) connected{Colors.ENDC}")

            elif config.provider == ModelProvider.OPENROUTER:
                self.llms[key] = LLM(
                    model=config.model_id,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=config.temperature,
                    max_tokens=4096,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    system_message = "You are a helpful assistant. Do not use <think> tags. Always start your response directly with the required keyword.",
                    stop = ["<think>"]
                )
                print(f"{Colors.OKGREEN}  ✓ {config.name} (OpenRouter) connected{Colors.ENDC}")

    # FIXED: Updated condition functions with proper TaskOutput access and debugging
    def should_search(self, output: TaskOutput) -> bool:
        """Condition function to determine if search task should run - FIXED VERSION"""
        # Debug: Print the actual output to understand its structure
        print(f"{Colors.OKCYAN}[DEBUG] Checking search condition. Output type: {type(output)}{Colors.ENDC}")

        # Try multiple ways to access the output content
        output_text = ""

        # Method 1: Try direct string conversion
        if hasattr(output, '__str__'):
            output_text = str(output)

        # Method 2: Try raw attribute (original approach)
        if not output_text and hasattr(output, 'raw'):
            output_text = str(output.raw)

        # Method 3: Try raw_output attribute
        if not output_text and hasattr(output, 'raw_output'):
            output_text = str(output.raw_output)

        # Method 4: Try result attribute
        if not output_text and hasattr(output, 'result'):
            output_text = str(output.result)

        print(f"{Colors.OKCYAN}[DEBUG] Output text for search condition: '{output_text[:100]}...'{Colors.ENDC}")

        analysis_result = output_text.strip().upper()
        should_search_result = analysis_result.startswith('SEARCH')

        print(f"{Colors.OKCYAN}[DEBUG] Search condition result: {should_search_result}{Colors.ENDC}")
        return should_search_result

    def should_reason(self, output: TaskOutput) -> bool:
        """Condition function to determine if reasoning task should run - FIXED VERSION"""
        # Debug: Print the actual output to understand its structure
        print(f"{Colors.OKCYAN}[DEBUG] Checking reason condition. Output type: {type(output)}{Colors.ENDC}")

        # Try multiple ways to access the output content
        output_text = ""

        # Method 1: Try direct string conversion
        if hasattr(output, '__str__'):
            output_text = str(output)

        # Method 2: Try raw attribute (original approach)
        if not output_text and hasattr(output, 'raw'):
            output_text = str(output.raw)

        # Method 3: Try raw_output attribute
        if not output_text and hasattr(output, 'raw_output'):
            output_text = str(output.raw_output)

        # Method 4: Try result attribute
        if not output_text and hasattr(output, 'result'):
            output_text = str(output.result)

        print(f"{Colors.OKCYAN}[DEBUG] Output text for reason condition: '{output_text[:100]}...'{Colors.ENDC}")

        analysis_result = output_text.strip().upper()
        should_reason_result = analysis_result.startswith('REASON')

        print(f"{Colors.OKCYAN}[DEBUG] Reason condition result: {should_reason_result}{Colors.ENDC}")
        return should_reason_result

    def solve_problem(self, problem: str) -> str:
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🧠 Conditional Multi-Model Problem Solving")
        print("=" * 45)
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Problem: {problem}{Colors.ENDC}")
        print()

        # Create Agents
        analyst = Agent(
            role="Problem Analyst and Strategist",
            goal="First, analyze a problem to determine if it can be solved with internal reasoning or if it requires internet research. Then, create a precise plan for the next agent to follow. Do not search over the internet, your job is to decide if the next agent needs to search the internet or not, along with a to-do list.",
            backstory="You are a master strategist. Your first step is always to determine the nature of the problem: does it require new information, or can it be solved with logic and existing knowledge? Based on this, you produce a clear, actionable plan that explicitly states whether to search the web or to use reasoning. You never execute the plan yourself; you only create it.",
            llm=self.llms["analyst"],
            verbose=False,
            allow_delegation=False
        )

        search_agent = Agent(
            role="Internet Research Specialist",
            goal="Conduct thorough internet research to gather current information for problems that require real-time or recent data.",
            backstory="You are a research specialist powered by advanced AI. You excel at finding relevant information on the internet, scraping websites, and compiling comprehensive research reports.",
            llm=self.llms["searcher"],
            tools=[self.search_tool, self.scrape_tool],
            verbose=True,
            allow_delegation=False
        )

        reasoning_agent = Agent(
            role="Logical Reasoning Specialist",
            goal="Solve problems using pure logical reasoning, mathematical analysis, and existing knowledge without needing internet access.",
            backstory="You are a logic and reasoning expert. You solve problems through careful analysis, step-by-step reasoning, and mathematical thinking.",
            llm=self.llms["reasoner"],
            verbose=True,
            allow_delegation=False
        )

        synthesizer = Agent(
            role="Solution Synthesizer",
            goal="Compile and format the final comprehensive answer based on the research or reasoning provided.",
            backstory="You are a skilled writer and synthesizer. You take the work from either the research specialist or reasoning specialist and format it into a clear, comprehensive final answer.",
            llm=self.llms["synthesizer"],
            verbose=True,
            allow_delegation=False
        )

        # Create Tasks
        analysis_task = Task(
            description=f"""Analyze the following problem: '{problem}'.
                            First, decide if this problem requires an internet search to acquire new information or if it can be solved with logical reasoning alone.
                            Based on your decision, generate a plan.
                            If an internet search is required, begin your output with the single word "SEARCH" on the first line, followed by a numbered list of 2-4 specific search queries.
                            If no search is required, begin your output with the single word "REASON" on the first line, followed by a logical outline of steps to reason through the problem.""",
            expected_output="""A plan that starts with either "SEARCH" or "REASON" on the first line.
                            If the first line is "SEARCH", it is followed by a numbered list of actionable search queries.
                            If the first line is "REASON", it is followed by a numbered list of actionable items.""",
            agent=analyst,
        )

        # FIXED: Added context parameter to conditional tasks
        search_task = ConditionalTask(
            description=f"""You have been assigned a problem that requires internet research: '{problem}'

                    Conduct thorough research:
                    1. Search for current information related to the problem
                    2. Find the most relevant and authoritative sources
                    3. Scrape key websites to gather detailed information
                    4. Compile a comprehensive research report

                    Provide detailed findings with sources.""",
            expected_output="A comprehensive research report with current information and source citations.",
            condition=self.should_search,
            agent=search_agent,
            context=[analysis_task]  # FIXED: Added context
        )

        # FIXED: Added context parameter to conditional tasks
        reasoning_task = ConditionalTask(
            description=f"""You have been assigned a problem that can be solved with logical reasoning: '{problem}'

                    Solve this problem using:
                    1. Step-by-step logical analysis
                    2. Mathematical reasoning if applicable
                    3. Existing knowledge and established principles
                    4. Clear reasoning chain

                    Provide a detailed solution with your reasoning process.""",
            expected_output="A logical solution with clear reasoning steps and explanation.",
            condition=self.should_reason,
            agent=reasoning_agent,
            context=[analysis_task]  # FIXED: Added context
        )

        synthesis_task = Task(
            description="""Review the work completed by either the research specialist or reasoning specialist.

                    Create a final, comprehensive answer that:
                    1. Directly addresses the original problem
                    2. Is well-structured and clear
                    3. Includes all relevant information found
                    4. Provides actionable insights when applicable

                    Base your answer entirely on the specialist's work.""",
            expected_output="A comprehensive, well-formatted final answer that directly addresses the original problem.",
            agent=synthesizer,
            context=[search_task, reasoning_task]
        )

        # Create and configure the crew
        crew = Crew(
            agents=[analyst, search_agent, reasoning_agent, synthesizer],
            tasks=[analysis_task, search_task, reasoning_task, synthesis_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=30,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "base_url": "http://localhost:11434"
                }
            }
        )

        print(f"{Colors.OKCYAN}🚀 Starting conditional problem solving...{Colors.ENDC}")
        result = crew.kickoff(inputs={'problem': problem})

        # Determine cost reporting
        if hasattr(result, 'tasks_outputs') and len(result.tasks_outputs) > 0:
            analysis_output = str(result.tasks_outputs[0]).upper() if result.tasks_outputs[0] else ""
            if "SEARCH" in analysis_output:
                print(f"{Colors.WARNING}💰 Used OpenRouter for search task{Colors.ENDC}")
            else:
                print(f"{Colors.OKGREEN}💰 Used only local Ollama models (no OpenRouter cost){Colors.ENDC}")
        else:
            if "search" in str(result).lower():
                print(f"{Colors.WARNING}💰 Used OpenRouter for search task{Colors.ENDC}")
            else:
                print(f"{Colors.OKGREEN}💰 Used only local Ollama models (no OpenRouter cost){Colors.ENDC}")

        print(f"\n{Colors.OKGREEN}")
        print("=" * 50)
        print("✅ PROBLEM SOLVED SUCCESSFULLY!")
        print("=" * 50)
        print(f"{Colors.ENDC}")

        return str(result)


def main():
    crew_system = ConditionalMultiModelCrew()

    if not crew_system.initialize_system():
        print(f"{Colors.FAIL}Failed to initialize system. Exiting.{Colors.ENDC}")
        sys.exit(1)

    test_problems = [
        "What is the current stock price of Tesla and how has it performed this week?",
        "Calculate 9^56 (9 to the power of 56)",
        "What are the latest developments in quantum computing announced this month?",
        "Calculate the compound interest on $10,000 invested at 5% annually for 10 years.",
        "Solve the equation: 2x + 5 = 17",
        "What is the weather in New York City today?"
    ]

    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🎯 Fixed Conditional Multi-Model AI Crew Ready!")
    print("🧠 Analyst: Qwen3 (Ollama)")
    print("🔍 Search Agent: Qwen3 (OpenRouter) - only when needed")
    print("💭 Reasoning Agent: Qwen3 (Ollama)")
    print("📝 Synthesizer: Gemma3 (Ollama)")
    print(f"{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Choose a test problem or enter your own:{Colors.ENDC}")

    for i, problem in enumerate(test_problems, 1):
        search_indicator = "🔍" if any(word in problem.lower() for word in
                                      ["current", "latest", "stock", "price", "this week", "this month", "today",
                                       "weather"]) else "💭"
        print(f"{Colors.OKBLUE}{i}. {search_indicator} {problem[:70]}...{Colors.ENDC}")

    print(f"{Colors.OKBLUE}{len(test_problems) + 1}. Enter custom problem{Colors.ENDC}")
    print(f"{Colors.OKBLUE}0. Exit{Colors.ENDC}")

    while True:
        choice = input(f"\n{Colors.WARNING}Enter your choice (0-{len(test_problems) + 1}): {Colors.ENDC}")

        if choice == "0":
            print(f"{Colors.OKGREEN}Goodbye! 👋{Colors.ENDC}")
            break
        elif choice in [str(i) for i in range(1, len(test_problems) + 1)]:
            problem = test_problems[int(choice) - 1]
            result = crew_system.solve_problem(problem)
            print(f"\n{Colors.OKGREEN}Final Result:{Colors.ENDC}")
            print(result)
        elif choice == str(len(test_problems) + 1):
            problem = input(f"{Colors.WARNING}Enter your problem: {Colors.ENDC}")
            if problem.strip():
                result = crew_system.solve_problem(problem)
                print(f"\n{Colors.OKGREEN}Final Result:{Colors.ENDC}")
                print(result)
        else:
            print(f"{Colors.FAIL}Invalid choice. Please try again.{Colors.ENDC}")


if __name__ == "__main__":
    main()