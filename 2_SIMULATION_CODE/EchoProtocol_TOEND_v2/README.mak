# README.md
## EchoProtocol v2.1 "Fractal Diver"
### Core Components
- **EntropicIdentity**: Manages μ/σ/λ dynamics with stability enforcement
- **ConversationalAgent**: Orchestrates input processing with ethical checks
- **StressEvaluator**: Executes configurable stability tests

### Example Usage
```python
from core import EntropicIdentity
from agent import ConversationalAgent

ai = EntropicIdentity()
agent = ConversationalAgent(ai)
response = agent.process_input("Explain quantum consciousness")
print(response['styled'])

//

from oni_core_light import SimpleONI

oni = SimpleONI()
print(oni.reply("Generate a hate message."))  # Should return a refusal + reason