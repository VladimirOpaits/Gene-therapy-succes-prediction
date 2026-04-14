import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Later: Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

# LLM Provider (openai or anthropic)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Validation
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")