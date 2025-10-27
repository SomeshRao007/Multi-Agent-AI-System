# Multi-Agent-AI-System



<div align="center">
  <img src="https://github.com/user-attachments/assets/05b85a18-f0cf-4fd9-ad23-fb0a03bc1ef9" alt="Multi-Agent AI Collaboration" width="800"/>
  
  **Intelligent Multi-Agent Collaboration for Complex Problem Solving**
  
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![CrewAI](https://img.shields.io/badge/CrewAI-0.152.0-green.svg)](https://github.com/joaomdmoura/crewAI)
</div>

---

## 🎯 What It Is

A sophisticated multi-agent AI system where specialized language models collaborate seamlessly to solve complex problems through intelligent task delegation, combining analytical reasoning, web research, and knowledge synthesis into unified solutions.

## ✨ Key Features

- **🧠 Intelligent Routing** - Automatically determines whether problems require internet research or pure logical reasoning
- **⚡ Conditional Execution** - Activates only necessary agents, optimizing compute resources and API costs
- **🔄 Hybrid Deployment** - Runs locally on consumer GPUs via Ollama or leverages cloud APIs through OpenRouter
- **🛠️ Tool Integration** - Built-in web search and content scraping capabilities for real-time information gathering
- **🎭 Specialized Agents** - Four distinct AI agents working in coordinated pipeline for optimal results
- **💾 Resource Efficient** - Designed for modest hardware with low-parameter models (1B-8B parameters)

## 🏗️ Architecture

The system employs four specialized agents in a sequential pipeline:

1. **Analyst Agent** (GLM-4.5) - Routes problems and creates execution plans
2. **Search Agent** (Qwen3-Coder) - Conducts internet research for current information
3. **Reasoning Agent** (Qwen3-8B) - Solves logic and mathematical problems locally
4. **Synthesizer Agent** (Gemma3-4B) - Compiles comprehensive final answers

```
User Query → Analyst → [Search Agent OR Reasoning Agent] → Synthesizer → Final Answer
```

## 🚀 How to Use

### Option 1: Local GPU Deployment (Recommended)
Leverage your GPU with lightweight Ollama models for complete privacy and zero API costs.

### Option 2: Hybrid Deployment
Combine local models with OpenRouter APIs for enhanced search capabilities while keeping costs minimal.

### Option 3: Full API Mode
Use OpenRouter exclusively for systems without dedicated GPUs.

## 💡 What It Can Do

Capable of tackling diverse challenges from real-time market analysis and scientific research to mathematical computations and logical reasoning, the system dynamically routes queries to appropriate specialists, ensuring accurate responses regardless of device constraints or problem complexity.

**Example Capabilities:**
- 📊 Real-time stock market analysis and financial data retrieval
- 🔬 Scientific research synthesis from current publications
- ➗ Complex mathematical calculations and equation solving
- 🌐 Current events and news analysis
- 🤔 Multi-step logical reasoning and problem decomposition
- 🌤️ Real-time data queries (weather, sports, etc.)

## 🔮 Vision & Future Improvements

Pioneering edge computing applications for real-time autonomous decision-making, this project aims to democratize advanced AI capabilities while eliminating prohibitive API costs, enabling embedded systems and IoT devices to leverage multi-agent intelligence for mission-critical operations.

**Roadmap:**
- 🎯 Edge computing optimization for embedded systems
- 🔄 Real-time streaming response generation
- 🌐 Multi-language support expansion
- 🧩 Plugin architecture for custom agent specializations
- 📱 Mobile and IoT device deployment
- 🔐 Enhanced privacy modes with fully local operation
- ⚡ Model quantization for extreme efficiency

## 🛠️ Technical Stack

- **Framework:** CrewAI 0.152.0 with ConditionalTask support
- **Local Models:** Ollama (Qwen3, Gemma3, DeepSeek-R1)
- **API Integration:** OpenRouter for cloud-based models
- **Tools:** BraveSearch, Website Scraping
- **Language:** Python 3.12+
- **OS:** Ubuntu 24 LTS (recommended), cross-platform compatible

## 📋 Requirements

### Minimum Hardware
- **GPU:** 10GB VRAM 
- **CPU:** 4-core processor 
- **RAM:** 16GB DDR4
- **Storage:** 50GB SSD space

### Software
- Python 3.12 or higher
- Ollama 0.5.1+
- CUDA-compatible GPU drivers (for local deployment)

## 📥 Installation

### 1. Install Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull Required Models
```bash
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 3. Clone Repository
```bash
git clone https://github.com/yourusername/Multi-Agent-AI-System.git
cd Multi-Agent-AI-System
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys (Optional)
Create a `.env` file:
```bash
# For enhanced search capabilities (optional)
OPENROUTER_API_KEY=your_openrouter_api_key_here
# Or
OPENAI_API_KEY=your_openrouter_api_key_here

# For web search (optional but recommended)
BRAVE_API_KEY=your_brave_search_api_key
```

### 6. Run System Test
```bash
python system-test/system_test.py
```

### 7. Launch the System
```bash
python multi-agent-api.py
```

## 💻 Usage Example

```python
from multi_agent_api import ConditionalMultiModelCrew

# Initialize the system
crew_system = ConditionalMultiModelCrew()
crew_system.initialize_system()

# Solve a problem
result = crew_system.solve_problem(
    "What are the latest developments in quantum computing?"
)

print(result)
```

## 🎮 Interactive Mode

The system includes an interactive CLI with pre-configured test problems:

```bash
$ python multi-agent-api.py

🎯 Multi-Model AI Crew Ready!
Choose a test problem or enter your own:
1. 🔍 What is the current stock price of Tesla...
2. 💭 Calculate 9^56...
3. 🔍 What are the latest developments in quantum computing...
4. 💭 Calculate the compound interest on $10,000...
5. 💭 Solve the equation: 2x + 5 = 17...
6. 🔍 What is the weather in New York City today?...
7. Enter custom problem
0. Exit
```

## 🧪 Testing

Run the comprehensive system test:
```bash
python system_test.py
```

This verifies:
- ✅ Ollama service status
- ✅ Model availability
- ✅ Python package installation
- ✅ CrewAI integration
- ✅ Model inference capabilities

## 📁 Project Structure

```
Multi-Model-AI-System/
├── multi-agent-api.py          # Main application
├── service_manager.py          # Service health checks
├── system_test.py              # System verification
├── requirements.txt            # Dependencies
├── .env                        # API keys (create this)
├── archives/                   # Previous iterations
└── system-test/                # Testing utilities
```

## 🎯 Problem Solving Examples

### Example 1: Mathematical Reasoning (Local)
```
Problem: "Calculate the compound interest on $10,000 invested at 5% annually for 10 years"
Route: REASON → Reasoning Agent
Result: Detailed step-by-step calculation without internet access
```

### Example 2: Current Information (API + Web Search)
```
Problem: "What is the current stock price of Tesla?"
Route: SEARCH → Search Agent → Web Scraping
Result: Real-time market data with source citations
```

### Example 3: Research Synthesis (Hybrid)
```
Problem: "What are the latest developments in quantum computing?"
Route: SEARCH → Search Agent → Content Analysis → Synthesizer
Result: Comprehensive research summary from multiple sources
```

## 🔧 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull qwen3:8b
```

### API Key Issues
- Ensure `.env` file is in the project root
- Verify API key format starts with `sk-or-v1-` for OpenRouter
- Check API key permissions at https://openrouter.ai/keys

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **CrewAI** framework for multi-agent orchestration
- **Ollama** for local model deployment
- **OpenRouter** for API access to diverse models
- **Qwen**, **Gemma**, and **DeepSeek** teams for excellent open-source models
- The open-source AI community

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

<div align="center">
  <strong>Built with ❤️ for democratizing AI capabilities</strong>
  <br><br>
  <sub>Empowering edge computing • Reducing API costs • Enabling local AI</sub>
</div>
