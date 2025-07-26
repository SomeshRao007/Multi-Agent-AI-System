#!/usr/bin/env python3

import os

from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv()

import sys
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dataclasses import dataclass
from enum import Enum
# Core CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai_tools import BraveSearchTool, FileReadTool, DirectoryReadTool


# Color codes for enhanced terminal output
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
    """Define each model's primary strength"""
    ANALYTICAL = "analytical_thinking"
    EXECUTION = "efficient_processing"
    REASONING = "advanced_reasoning"


@dataclass
class ModelConfig:
    """Configuration for each AI model"""
    name: str
    ollama_model: str
    strength: ModelStrength
    role: str
    description: str
    temperature: float
    context_window: int


class OllamaManager:
    """Manages Ollama model operations and health checks"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def check_ollama_status(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_available_models(self) -> list:
        """Get list of downloaded models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                return [model['name'] for model in response.json().get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        available_models = self.list_available_models()
        return any(model_name in model for model in available_models)

    def pull_model_if_needed(self, model_name: str) -> bool:
        """Download model if not available"""
        if self.check_model_availability(model_name):
            print(f"{Colors.OKGREEN}✓ Model {model_name} already available{Colors.ENDC}")
            return True

        print(f"{Colors.WARNING}⚠ Model {model_name} not found. Downloading...{Colors.ENDC}")
        print(f"{Colors.OKCYAN}This may take several minutes depending on model size{Colors.ENDC}")

        try:
            # Use subprocess for better control of the download process
            import subprocess
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                print(f"{Colors.OKGREEN}✓ Successfully downloaded {model_name}{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL}✗ Failed to download {model_name}: {result.stderr}{Colors.ENDC}")
                return False

        except subprocess.TimeoutExpired:
            print(f"{Colors.FAIL}✗ Download timeout for {model_name}{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}✗ Error downloading {model_name}: {str(e)}{Colors.ENDC}")
            return False


class MultiModelCrew:
    """Manages the multi-model AI crew with conditional agent execution."""

    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.models_config = self._setup_model_configurations()
        self.llms = {}
        self.search_tool = BraveSearchTool()
        self.scrape_tool = ScrapeWebsiteTool()

    def _setup_model_configurations(self) -> Dict[str, ModelConfig]:
        """Configure each model with its optimal settings and role"""
        # We define a single execution model that will be used for two different agents
        execution_model_name = "deepseek-r1:8b" # You can change this to deepseek-r1:8b or another model

        return {
            "analyst": ModelConfig(
                name="Qwen Analyst",
                ollama_model="qwen3:4b",
                strength=ModelStrength.ANALYTICAL,
                role="Problem Analyst",
                description="Analyzes problems to create a plan.",
                temperature=0.3,
                context_window=32768
            ),
            "reasoner": ModelConfig( # New agent for reasoning
                name="DeepSeek Reasoner",
                ollama_model=execution_model_name,
                strength=ModelStrength.REASONING,
                role="Logical Reasoner",
                description="Solves problems using internal knowledge and logic.",
                temperature=0.4,
                context_window=128000
            ),
            "searcher": ModelConfig( # New agent for searching
                name="DeepSeek Searcher",
                ollama_model=execution_model_name,
                strength=ModelStrength.EXECUTION,
                role="Information Searcher",
                description="Searches the internet to find information.",
                temperature=0.6,
                context_window=128000
            ),
            "synthesizer": ModelConfig(
                name="Gemma Synthesizer",
                ollama_model="gemma3:1b",
                strength=ModelStrength.EXECUTION,
                role="Solution Synthesizer",
                description="Synthesizes information into a final answer.",
                temperature=0.6,
                context_window=32768
            )
        }

    def initialize_system(self) -> bool:
        """Initialize the entire multi-model system"""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🚀 Initializing Multi-Model AI Crew with Ollama")
        print("=" * 50)
        print(f"{Colors.ENDC}")

        if not self.ollama_manager.check_ollama_status():
            print(f"{Colors.FAIL}✗ Ollama service not running{Colors.ENDC}")
            print(f"{Colors.WARNING}Please start Ollama with: ollama serve{Colors.ENDC}")
            return False
        print(f"{Colors.OKGREEN}✓ Ollama service is running{Colors.ENDC}")

        models_ready = True
        for config in self.models_config.values():
            if not self.ollama_manager.pull_model_if_needed(config.ollama_model):
                models_ready = False

        if not models_ready:
            print(f"{Colors.FAIL}✗ Some models failed to download{Colors.ENDC}")
            return False

        self._setup_llm_connections()
        print(f"{Colors.OKGREEN}✅ Multi-Model Crew initialized successfully!{Colors.ENDC}")
        return True

    def _setup_llm_connections(self):
        """Setup LLM connections to Ollama models"""
        print(f"{Colors.OKCYAN}⚙ Setting up LLM connections...{Colors.ENDC}")
        for key, config in self.models_config.items():
            try:
                self.llms[key] = LLM(
                    model=f"ollama/{config.ollama_model}",
                    base_url="http://localhost:11434",
                    temperature=config.temperature,
                    max_tokens=4096,
                    top_p=0.9
                )
                print(f"{Colors.OKGREEN}  ✓ {config.name} connected{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.FAIL}  ✗ Failed to connect {config.name}: {str(e)}{Colors.ENDC}")
                raise

    def solve_problem(self, problem: str) -> str:
        """
        Solves a problem by first analyzing it, then dynamically assigning the
        task to a specialist crew (reasoning or searching).
        """
        print(f"{Colors.HEADER}{Colors.BOLD}Phase 1: Analysis{Colors.ENDC}")
        # --- STAGE 1: ANALYSIS ---
        analyst = Agent(
            role="Problem Analyst and Strategist",
            goal="Analyze a problem and create a plan. Decide if it requires an internet search ('SEARCH') or internal logic ('REASON'). Your output must be only the plan.",
            backstory="You are a master strategist. You produce a clear, actionable plan that starts with either 'SEARCH' or 'REASON'. You never execute the plan yourself.",
            llm=self.llms["analyst"],
            verbose=True
        )

        analysis_task = Task(
            description=f"Analyze the problem: '{problem}'. If an internet search is needed, start your plan with 'SEARCH' and list 2-3 well structured search queries, Do not search by yourself. If not, start with 'REASON' and list of todo list for next agent to respond. Do not solve it.",
            expected_output="A plan starting with 'SEARCH' or 'REASON', followed by a numbered list.",
            agent=analyst,
        )

        # Create and run the analysis crew
        analysis_crew = Crew(agents=[analyst], tasks=[analysis_task], process=Process.sequential, verbose=False)
        analysis_output = analysis_crew.kickoff(inputs={'problem': problem})

        # --- FIX 1: Convert CrewOutput object to a simple string ---
        plan = str(analysis_output)

        print(f"\n{Colors.OKCYAN}Analysis Complete. Plan:{Colors.ENDC}\n{plan}\n")
        print(f"{Colors.HEADER}{Colors.BOLD}Phase 2: Execution{Colors.ENDC}")

        # --- STAGE 2: DYNAMIC EXECUTION ---
        execution_agents = []
        execution_tasks = []

        # Define the Synthesizer agent, which is used in both cases
        synthesizer = Agent(
            role="Solution Synthesizer",
            goal="Review the provided information and compile a final, comprehensive answer.",
            backstory="You are a focused writer. You receive a report and your only job is to format it into a final, comprehensive answer.",
            llm=self.llms["synthesizer"],
            verbose=True,
        )

        # --- FIX 2: Check for 'SEARCH' more robustly ---
        if "SEARCH" in plan.upper():
            print(f"{Colors.OKBLUE}Decision: Searching the internet.{Colors.ENDC}")
            # 1. Define Searcher Agent
            searcher = Agent(
                role="Information Searcher",
                goal="Check the plan provided by analyst and Execute the search queries from the plan, scrape the best result, and provide a detailed report. If no search queries are found in the plan then devise them based on analyst plan.",
                backstory="You are an expert at using search tools to find relevant information on the internet.",
                llm=self.llms["searcher"],
                tools=[self.search_tool, self.scrape_tool],
                verbose=True,
            )
            execution_agents.extend([searcher, synthesizer])

            # 2. Define Search and Synthesis Tasks
            search_task = Task(
                description="Execute the 'SEARCH' plan provided. Use your tools to search the web and scrape the most relevant webpage. Consolidate your findings into a detailed report.",
                expected_output="A detailed report based on the scraped web content.",
                agent=searcher,
            )
            synthesis_task = Task(
                description="Review the research report and synthesize it into a final, well-structured answer to the original problem.",
                expected_output="A comprehensive final answer based on the research report.",
                agent=synthesizer,
                context=[search_task],
            )
            execution_tasks.extend([search_task, synthesis_task])

        else: # Default to REASON
            print(f"{Colors.OKBLUE}Decision: Using internal reasoning.{Colors.ENDC}")
            # 1. Define Reasoner Agent
            reasoner = Agent(
                role="Logical Reasoner",
                goal="Check the plan provided by analyst and Execute the 'REASON' plan provided using your internal knowledge and problem-solving skills.",
                backstory="You are a master of logic and reasoning. You solve problems step-by-step without accessing external tools.",
                llm=self.llms["reasoner"],
                tools=[], # IMPORTANT: The reasoner has no tools
                verbose=True,
            )
            execution_agents.extend([reasoner, synthesizer])

            # 2. Define Reasoning and Synthesis Tasks
            reasoning_task = Task(
                description="Execute the 'REASON' plan step-by-step to solve the problem. Provide a clear, well-reasoned final answer.",
                expected_output="A final, detailed answer derived from logical deduction.",
                agent=reasoner,
            )
            synthesis_task = Task(
                description="Review the reasoned answer, refine its formatting, and ensure it directly answers the original problem.",
                expected_output="A clean, well-formatted final answer.",
                agent=synthesizer,
                context=[reasoning_task],
            )
            execution_tasks.extend([reasoning_task, synthesis_task])

        # 3. Create and run the appropriate execution crew
        execution_crew = Crew(
            agents=execution_agents,
            tasks=execution_tasks,
            process=Process.sequential,
            verbose=True
        )

        try:
            print(f"{Colors.OKCYAN}🚀 Starting collaborative execution...{Colors.ENDC}")
            # Pass the string version of the plan to the next crew
            result = execution_crew.kickoff(inputs={'plan': plan})
            print(f"\n{Colors.OKGREEN}{'='*50}\n✅ PROBLEM SOLVED SUCCESSFULLY!\n{'='*50}{Colors.ENDC}")
            return str(result)
        except Exception as e:
            print(f"{Colors.FAIL}\n✗ Error during execution: {str(e)}{Colors.ENDC}")
            return f"Error: {str(e)}"

def main():
    """Main execution function"""

    # Initialize the multi-model crew
    crew_system = MultiModelCrew()

    if not crew_system.initialize_system():
        print(f"{Colors.FAIL}Failed to initialize system. Exiting.{Colors.ENDC}")
        sys.exit(1)

    # Example problems to test the system
    test_problems = [
        "What are the key differences in architecture and performance between RISC-V and ARM processors, and what are the implications for the future of mobile computing?",
        "Design a sustainable energy solution for a small rural community with limited grid access.",
        "Analyze the potential impact of quantum computing on current encryption methods and propose transition strategies."
    ]

    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🎯 Multi-Model AI Crew Ready!")
    print(f"{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Choose a test problem or enter your own:{Colors.ENDC}")

    for i, problem in enumerate(test_problems, 1):
        print(f"{Colors.OKBLUE}{i}. {problem[:80]}...{Colors.ENDC}")

    print(f"{Colors.OKBLUE}{len(test_problems) + 1}. Enter custom problem{Colors.ENDC}")
    print(f"{Colors.OKBLUE}0. Exit{Colors.ENDC}")

    while True:
        try:
            choice = input(f"\n{Colors.WARNING}Enter your choice (0-{len(test_problems) + 1}): {Colors.ENDC}")
            choice = int(choice)

            if choice == 0:
                print(f"{Colors.OKGREEN}Goodbye! 👋{Colors.ENDC}")
                break
            elif 1 <= choice <= len(test_problems):
                problem = test_problems[choice - 1]
                result = crew_system.solve_problem(problem)
                print(f"\n{Colors.OKGREEN}Final Result:{Colors.ENDC}")
                print(result)
            elif choice == len(test_problems) + 1:
                problem = input(f"{Colors.WARNING}Enter your problem: {Colors.ENDC}")
                if problem.strip():
                    result = crew_system.solve_problem(problem)
                    print(f"\n{Colors.OKGREEN}Final Result:{Colors.ENDC}")
                    print(result)
            else:
                print(f"{Colors.FAIL}Invalid choice. Please try again.{Colors.ENDC}")

        except ValueError:
            print(f"{Colors.FAIL}Please enter a valid number.{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.OKGREEN}Goodbye! 👋{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}")


if __name__ == "__main__":
    main()