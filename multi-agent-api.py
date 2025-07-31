#!/usr/bin/env python3

import os
import re
import sys
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai_tools import BraveSearchTool
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from dataclasses import dataclass
from dotenv import load_dotenv
from enum import Enum
from service_manager import ServiceManager, Colors
from typing import Dict

load_dotenv()


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


class ConditionalMultiModelCrew:
    def __init__(self):
        self.service_manager = ServiceManager(openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.models_config = self._setup_model_configurations()
        self.llms = {}
        self.agents = {}
        self.knowledge_base = {}
        self._cached_analysis_output = None
        self.search_tool = BraveSearchTool()
        self.scrape_tool = ScrapeWebsiteTool()

    def _setup_model_configurations(self) -> Dict[str, ModelConfig]:
        return {
            "analyst": ModelConfig(
                name="GLM Analyst",
                model_id="openrouter/z-ai/glm-4.5",
                provider=ModelProvider.OPENROUTER,
                strength=ModelStrength.ANALYTICAL,
                role="Problem Analyst",
                description="Problem analysis and routing decisions",
                temperature=0.3,
                context_window=128000
            ),
            "searcher": ModelConfig(
                name="Qwen3 Search Agent",
                model_id="openrouter/qwen/qwen3-coder",
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

    def initialize_system(self) -> bool:
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🚀 Initializing Fixed Conditional Multi-Model AI Crew")
        print("🔧 With Proper TaskOutput Handling")
        print("=" * 55)
        print(f"{Colors.ENDC}")

        if not self.service_manager.check_openrouter_setup():
            return False

        if not self.service_manager.test_openrouter_connection():
            print(f"{Colors.WARNING}⚠ OpenRouter test failed - search tasks will not work{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠ Continuing with local models only (reasoning tasks will work){Colors.ENDC}")

        ollama_models = [k for k, v in self.models_config.items() if v.provider == ModelProvider.OLLAMA]
        if ollama_models:
            if not self.service_manager.check_ollama_status():
                print(f"{Colors.FAIL}✗ Ollama service not running{Colors.ENDC}")
                print(f"{Colors.WARNING}Please start Ollama with: ollama serve{Colors.ENDC}")
                return False

            print(f"{Colors.OKGREEN}✓ Ollama service is running{Colors.ENDC}")

            models_ready = True
            for key, config in self.models_config.items():
                if config.provider == ModelProvider.OLLAMA:
                    if not self.service_manager.pull_model_if_needed(config.model_id):
                        models_ready = False

            if not models_ready:
                print(f"{Colors.FAIL}✗ Some Ollama models failed to download{Colors.ENDC}")
                return False

        self._setup_llm_connections()

        print(f"{Colors.OKGREEN}✅ Multi-Model agents initialized successfully!{Colors.ENDC}")
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
                    api_key=self.service_manager.openrouter_api_key,
                    temperature=config.temperature,
                    max_tokens=4096,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                print(f"{Colors.OKGREEN}  ✓ {config.name} (OpenRouter) connected{Colors.ENDC}")

    def _extract_and_cache_output(self, output: TaskOutput) -> str:
        """Extract output once and cache it"""
        if self._cached_analysis_output is not None:
            return self._cached_analysis_output

        extraction_methods = [
            ("output.raw", lambda: getattr(output, 'raw', None)),
            ("str(output)", lambda: str(output)),
            ("output.raw_output", lambda: getattr(output, 'raw_output', None)),
        ]

        for method_name, method in extraction_methods:
            content = method()
            if content and str(content).strip():
                think_pattern = r'<think>.*?</think>'
                cleaned = re.sub(think_pattern, '', str(content), flags=re.DOTALL | re.IGNORECASE)
                cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
                cleaned = cleaned.strip()
                self._cached_analysis_output = cleaned
                return cleaned

        return ""

    def should_search(self, output: TaskOutput) -> bool:
        clean_output = self._extract_and_cache_output(output)
        return clean_output.upper().startswith('SEARCH')

    def should_reason(self, output: TaskOutput) -> bool:
        clean_output = self._extract_and_cache_output(output)
        return clean_output.upper().startswith('REASON')

    def solve_problem(self, problem: str) -> str:
        self._cached_analysis_output = None
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🧠 Fixed Conditional Multi-Model Problem Solving")
        print("=" * 50)
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Problem: {problem}{Colors.ENDC}")
        print()

        analyst = Agent(
            role="Problem Analyst and Strategist",
            goal="ONLY route problems, never solve them. Determine if a problem needs SEARCH or REASON.",
            backstory="""You are a routing agent. Your job is NEVER to solve problems, only to decide the approach.
            You must respond with either 'SEARCH' or 'REASON' as the first word, followed by action steps.
            You do NOT provide solutions, answers, or solve anything. You only route.""",
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

        analysis_task = Task(
            description=f"""CRITICAL: You must analyze this problem and decide routing ONLY. Do NOT solve it. Problem: '{problem}' 
            Your ONLY job is to decide routing. You must respond in this EXACT format: For current/live data (stocks, weather, news): Start with "SEARCH" and For math/logic/general knowledge: Start with "REASON"
            Format: First word must be exactly "SEARCH" or "REASON" followed by numbered steps.""",
            expected_output="Must start with exactly 'SEARCH' or 'REASON' followed by numbered action items. NO SOLUTIONS.",
            agent=analyst,
        )

        search_task = ConditionalTask(
            description=f"""You have been assigned a problem that requires internet research: '{problem}'

                    Conduct thorough research:
                    1. Search for current information related to the problem
                    2. Find the most relevant and authoritative sources
                    3. Scrape key websites to gather detailed information
                    4. Compile a comprehensive research report

            Focus on accuracy and current data.""",
            expected_output="Comprehensive research report with current information and source citations.",
            condition=self.should_search,
            agent=search_agent,
            context=[analysis_task]
        )

        reasoning_task = ConditionalTask(
            description=f"""You have been assigned a problem that can be solved with logical reasoning: '{problem}'

                    Solve this problem using:
                    1. Step-by-step logical analysis
                    2. Mathematical reasoning if applicable
                    3. Existing knowledge and established principles
                    4. Clear reasoning chain

            Provide detailed solution with reasoning process.""",
            expected_output="Complete logical solution with clear reasoning steps.",
            condition=self.should_reason,
            agent=reasoning_agent,
            context=[analysis_task]
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

        print(f"{Colors.OKCYAN}🚀 Starting fixed conditional problem solving...{Colors.ENDC}")
        result = crew.kickoff(inputs={'problem': problem})

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
    print("🎯 Multi-Model AI Crew Ready!")
    print("🧠 Analyst: Qwen3-4B (Ollama)")
    print("🔍 Search Agent: Qwen3-480B (OpenRouter) - only when needed")
    print("💭 Reasoning Agent: Qwen3-8B (Ollama)")
    print("📝 Synthesizer: Gemma3-4B (Ollama)")
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