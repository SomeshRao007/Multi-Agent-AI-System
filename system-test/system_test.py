#!/usr/bin/env python3
"""
Multi-Model CrewAI System Test Script
====================================

This script tests your Ollama + CrewAI setup to ensure everything is working correctly
before running the full multi-model tool.

Run this first to verify your system is ready!
"""

import requests
import subprocess
import sys
import json
from typing import Dict, List, Tuple


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: bool):
    """Print status with color coding"""
    if status:
        print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")
    return status


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def test_ollama_service() -> bool:
    """Test if Ollama service is running"""
    print_info("Testing Ollama service...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return print_status("Ollama service is running", response.status_code == 200)
    except requests.exceptions.RequestException:
        print_status("Ollama service is running", False)
        print_warning("Please start Ollama with: ollama serve")
        return False


def test_required_models() -> Tuple[bool, List[str]]:
    """Test if required models are available"""
    print_info("Checking required models...")

    required_models = [
        "qwen3:1.7b",
        "deepseek-r1:8b"
    ]

    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print_status("Could not retrieve model list", False)
            return False, []

        available_models = [model['name'] for model in response.json().get('models', [])]
        missing_models = []

        for model in required_models:
            found = any(model in available for available in available_models)
            if print_status(f"Model {model} available", found):
                continue
            else:
                missing_models.append(model)

        return len(missing_models) == 0, missing_models

    except Exception as e:
        print_status(f"Error checking models: {str(e)}", False)
        return False, required_models


def _test_model_inference(model_name: str) -> bool:
    """Test if a model can perform inference"""
    print_info(f"Testing inference for {model_name}...")

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return print_status(f"{model_name} inference working", True)
        else:
            return print_status(f"{model_name} inference working", False)

    except Exception as e:
        print_status(f"{model_name} inference working - Error: {str(e)}", False)
        return False


def test_python_packages() -> bool:
    """Test if required Python packages are installed"""
    print_info("Checking Python packages...")

    required_packages = [
        "crewai",
        "langchain",
        "ollama",
        "requests"
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"Package {package} installed", True)
        except ImportError:
            print_status(f"Package {package} installed", False)
            all_installed = False

    return all_installed


def test_crewai_ollama_integration() -> bool:
    """Test CrewAI + Ollama integration"""
    print_info("Testing CrewAI + Ollama integration...")

    try:
        from crewai.llm import LLM

        # Try to create an LLM connection
        llm = LLM(
            model="ollama/qwen3:1.7b",
            base_url="http://localhost:11434",
            temperature=0.3
        )

        return print_status("CrewAI + Ollama integration working", True)

    except Exception as e:
        print_status(f"CrewAI + Ollama integration working - Error: {str(e)}", False)
        return False


def suggest_fixes(missing_models: List[str], failed_tests: List[str]):
    """Suggest fixes for failed tests"""
    if not missing_models and not failed_tests:
        return

    print(f"\n{Colors.WARNING}🔧 Suggested Fixes:{Colors.ENDC}")
    print(f"{Colors.WARNING}{'=' * 20}{Colors.ENDC}")

    if "Ollama service" in failed_tests:
        print(f"{Colors.OKCYAN}1. Start Ollama service:{Colors.ENDC}")
        print("   ollama serve")
        print()

    if missing_models:
        print(f"{Colors.OKCYAN}2. Download missing models:{Colors.ENDC}")
        for model in missing_models:
            print(f"   ollama pull {model}")
        print()

    if "Python packages" in failed_tests:
        print(f"{Colors.OKCYAN}3. Install missing Python packages:{Colors.ENDC}")
        print("   pip install -r requirements.txt")
        print()

    if "CrewAI integration" in failed_tests:
        print(f"{Colors.OKCYAN}4. Fix CrewAI integration:{Colors.ENDC}")
        print("   pip install --upgrade crewai")
        print("   pip install --upgrade langchain")
        print()


def run_comprehensive_test():
    """Run all tests and provide comprehensive report"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🧪 Multi-Model CrewAI System Test")
    print("=" * 40)
    print(f"{Colors.ENDC}")

    failed_tests = []
    missing_models = []

    # Test 1: Ollama Service
    if not test_ollama_service():
        failed_tests.append("Ollama service")
        print_warning("Skipping model tests - Ollama not running")
        return False, [], failed_tests

    # Test 2: Required Models
    models_ok, missing = test_required_models()
    if not models_ok:
        missing_models = missing
        failed_tests.append("Required models")

    # Test 3: Model Inference (only if models are available)
    if models_ok:
        for model in ["qwen3:1.7b", "llama3.2:1b", "deepseek-r1:7b"]:
            if not _test_model_inference(model):
                failed_tests.append(f"{model} inference")

    # Test 4: Python Packages
    if not test_python_packages():
        failed_tests.append("Python packages")

    # Test 5: CrewAI Integration
    if not test_crewai_ollama_integration():
        failed_tests.append("CrewAI integration")

    print(f"\n{Colors.HEADER}📊 Test Results:{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 20}{Colors.ENDC}")

    total_tests = 5 + (3 if models_ok else 0)  # 3 inference tests if models available
    passed_tests = total_tests - len(failed_tests)

    if not failed_tests:
        print(f"{Colors.OKGREEN}🎉 All tests passed! Your system is ready! ({passed_tests}/{total_tests}){Colors.ENDC}")
        print(f"{Colors.OKGREEN}You can now run: python multimodal_crewai_ollama.py{Colors.ENDC}")
        return True, missing_models, failed_tests
    else:
        print(f"{Colors.WARNING}⚠ {passed_tests}/{total_tests} tests passed{Colors.ENDC}")
        suggest_fixes(missing_models, failed_tests)
        return False, missing_models, failed_tests


def main():
    """Main test execution"""
    success, missing_models, failed_tests = run_comprehensive_test()

    if success:
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL}❌ System not ready. Please fix the issues above and run the test again.{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()