#!/usr/bin/env python3
"""
Multi-Model AI Crew with CrewAI and Ollama Integration
=====================================================

This script creates a collaborative AI crew using three specialized models:
- Qwen3-1.7B: Fast analytical thinking and problem breakdown into ToDo list
- DeepSeek-R1-8B: Advanced reasoning, processing ToDo list and then synthesising a final solution.

Hardware Requirements:
- RTX 3060 12GB VRAM (✓ Your setup is perfect)
- 32GB RAM (✓)
- Ubuntu 24 LTS (✓)

Author: AI Engineering Assistant
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Core CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Utility imports
import requests
from dataclasses import dataclass
from enum import Enum


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
    """Main class for managing the multi-model AI crew"""

    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.models_config = self._setup_model_configurations()
        self.llms = {}
        self.agents = {}
        self.knowledge_base = {}

    def _setup_model_configurations(self) -> Dict[str, ModelConfig]:
        """Configure each model with its optimal settings and role"""
        return {
            "analyst": ModelConfig(
                name="Qwen3 Analyst",
                ollama_model="qwen3:1.7b",
                strength=ModelStrength.ANALYTICAL,
                role="Problem Analyst",
                description="analytical thinking, problem decomposition, creating a ToDo list",
                temperature=0.3,  # Lower for more focused analysis
                context_window=4096
            ),
            # "executor": ModelConfig(
            #     name="Llama Executor",
            #     ollama_model="llama3.2:1b",
            #     strength=ModelStrength.EXECUTION,
            #     role="Task Executor & Processor",
            #     description="Efficient task execution, information processing, and detail handling",
            #     temperature=0.5,  # Balanced for reliable execution
            #     context_window=4096
            # ),
            "synthesizer": ModelConfig(
                name="DeepSeek Synthesizer",
                ollama_model="deepseek-r1:8b",
                strength=ModelStrength.REASONING,
                role="Advanced Reasoner problem solving",
                description="Complex reasoning, for solving problems and synthesise solution",
                temperature=0.6,  # Higher for creative reasoning
                context_window=8192
            )
        }

    def initialize_system(self) -> bool:
        """Initialize the entire multi-model system"""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🚀 Initializing Multi-Model AI Crew with Ollama")
        print("=" * 50)
        print(f"{Colors.ENDC}")

        # Check Ollama status
        if not self.ollama_manager.check_ollama_status():
            print(f"{Colors.FAIL}✗ Ollama service not running{Colors.ENDC}")
            print(f"{Colors.WARNING}Please start Ollama with: ollama serve{Colors.ENDC}")
            return False

        print(f"{Colors.OKGREEN}✓ Ollama service is running{Colors.ENDC}")

        # Download required models
        models_ready = True
        for config in self.models_config.values():
            if not self.ollama_manager.pull_model_if_needed(config.ollama_model):
                models_ready = False

        if not models_ready:
            print(f"{Colors.FAIL}✗ Some models failed to download{Colors.ENDC}")
            return False

        # Initialize LLM connections
        self._setup_llm_connections()

        # Create specialized agents
        # self._create_agents()

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
                    # Additional parameters for optimal performance
                    max_tokens=2048,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                print(f"{Colors.OKGREEN}  ✓ {config.name} connected{Colors.ENDC}")

            except Exception as e:
                print(f"{Colors.FAIL}  ✗ Failed to connect {config.name}: {str(e)}{Colors.ENDC}")
                raise

    def solve_problem(self, problem: str) -> str:
        """
        Main method to solve a problem using the multi-model crew.
        Creates fresh agents and tasks for each run to prevent state leakage.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("🔧 Multi-Model Problem Solving Session")
        print("=" * 40)
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Problem: {problem}{Colors.ENDC}")
        print()



        # 1. Analyst Agent (Qwen3)
        analyst = Agent(
            role="Problem Analyst",
            goal="Create a numbered list of steps to solve a problem. Your ONLY output should be this list.",
            backstory="You are a machine-like planner. You receive a problem and output a numbered list of steps. You do not add commentary, explanations, or any text other than the numbered list itself.",
            llm=self.llms["analyst"],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=2,
            # memory=True
        )



        # 3. Synthesizer Agent (DeepSeek-R1)
        synthesizer = Agent(
            role="Solution Synthesizer",  # Renamed for clarity
            goal="Methodically execute every step in a given to-do list to build a final answer.",
            backstory="You are a focused executor. You DO NOT create plans. You receive a numbered to-do list from a Problem Analyst and your only job is to perform each task on that list, using the results to construct a final, comprehensive answer.",
            llm=self.llms["synthesizer"],
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            max_retry_limit=2,
            # reasoning=True,  # Default: False
            # max_reasoning_attempts=5,
            # respect_context_window=True,
            # memory=True
        )

        # --- Create Fresh Tasks for this specific run ---

        # Task 1: Analysis and Planning
        analysis_task = Task(
            description=f"Analyze the following problem and break it down into a numbered to-do list: '{problem}'",
            expected_output="""A numbered list of 4-6 actionable steps required to solve the problem.
                    IMPORTANT: Your output MUST ONLY be the numbered list. Do not provide any introduction, explanation, or conclusion. Just the list itself.""",
            agent=analyst,
            # Human input is not needed for this automated task
            # human_input=False
        )

        # Task 2: Synthesis and Final Solution
        synthesis_task = Task(
            description="""You have been given a to-do list by the 'Problem Analyst'. Your task is to execute EVERY step on that list.
                    Do not add new steps or deviate from the plan.
                    Base your entire final answer on the results of executing these steps.
                    The to-do list you must follow is in the context.""",
            expected_output="A comprehensive, well-structured final answer that is the direct result of completing all steps in the provided to-do list. The answer must directly address the original problem.",
            agent=synthesizer,
            context=[analysis_task]
        )
        # --- Create and configure the crew with the fresh components ---
        crew = Crew(
            agents=[analyst,  synthesizer],#executor,
            tasks=[analysis_task,  synthesis_task],#execution_task,
            process=Process.sequential,
            verbose=True,
            memory=False,  # Crew's memory is fresh for each new instance
            max_rpm=30,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "base_url": "http://localhost:11434"
                }
            }
        )

        # Execute the crew workflow
        try:
            print(f"{Colors.OKCYAN}🚀 Starting collaborative problem solving...{Colors.ENDC}")
            result = crew.kickoff(inputs={'problem': problem})

            print(f"\n{Colors.OKGREEN}")
            print("=" * 50)
            print("✅ PROBLEM SOLVED SUCCESSFULLY!")
            print("=" * 50)
            print(f"{Colors.ENDC}")

            return str(result)

        except Exception as e:
            print(f"{Colors.FAIL}\n✗ Error during problem solving: {str(e)}{Colors.ENDC}")
            return f"Error: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "ollama_running": self.ollama_manager.check_ollama_status(),
            "models_available": {},
            "agents_ready": len(self.agents) == 3,
            "llms_connected": len(self.llms) == 3
        }

        for key, config in self.models_config.items():
            status["models_available"][config.ollama_model] = self.ollama_manager.check_model_availability(
                config.ollama_model)

        return status


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