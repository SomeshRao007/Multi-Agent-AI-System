#!/usr/bin/env python3

import os
import requests
import subprocess


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


class ServiceManager:
    """Manages Ollama and OpenRouter service operations and health checks"""

    def __init__(self, base_url: str = "http://localhost:11434", openrouter_api_key: str = None):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    def check_ollama_status(self) -> bool:
        """Check if Ollama service is running"""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return response.status_code == 200

    def list_available_models(self) -> list:
        """Get list of downloaded models"""
        response = requests.get(f"{self.api_url}/tags")
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
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

    def check_openrouter_setup(self) -> bool:
        """Check if OpenRouter API key is configured"""
        if not self.openrouter_api_key:
            print(f"{Colors.FAIL}✗ OpenRouter API key not found{Colors.ENDC}")
            print(f"{Colors.WARNING}Please add your OpenRouter API key to .env file:{Colors.ENDC}")
            print(f"{Colors.OKCYAN}OPENROUTER_API_KEY=your_openrouter_api_key_here{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Or use: OPENAI_API_KEY=your_openrouter_api_key_here{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Get your key from: https://openrouter.ai/keys{Colors.ENDC}")
            return False

        print(f"{Colors.OKGREEN}✓ OpenRouter API key found (length: {len(self.openrouter_api_key)}){Colors.ENDC}")
        return True

    def test_openrouter_connection(self) -> bool:
        """Test OpenRouter API connection"""
        print(f"{Colors.OKCYAN}Testing OpenRouter connection...{Colors.ENDC}")

        test_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "CrewAI Multi-Agent System"
        }

        test_payload = {
            "model": "qwen/qwen3-coder",
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
            print(f"{Colors.WARNING}Make sure your API key starts with 'sk-or-v1-'{Colors.ENDC}")
            return False
        else:
            print(f"{Colors.WARNING}⚠ OpenRouter returned status {response.status_code}{Colors.ENDC}")
            print(f"{Colors.WARNING}Response: {response.text[:200]}{Colors.ENDC}")
            return False