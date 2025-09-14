#!/usr/bin/env python3
"""
OpenAI client wrapper that works around Railway proxy conflicts
"""
import os
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

class OpenAIWrapper:
    """Wrapper that creates OpenAI client avoiding Railway proxy issues."""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client with Railway compatibility."""
        try:
            # Method 1: Use subprocess to create client in clean environment
            logger.info("🔧 Attempting subprocess OpenAI client creation...")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("❌ No OPENAI_API_KEY found")
                return
            
            # Create OpenAI client using subprocess to avoid environment conflicts
            script = f'''
import os
import sys
os.environ["OPENAI_API_KEY"] = "{api_key}"

# Remove any proxy variables
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if var in os.environ:
        del os.environ[var]

from openai import OpenAI
client = OpenAI()
print("SUCCESS")
'''
            
            result = subprocess.run([
                sys.executable, '-c', script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                logger.info("✅ Subprocess OpenAI test successful")
                
                # Now create client directly with clean environment
                original_env = {}
                proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
                
                for var in proxy_vars:
                    if var in os.environ:
                        original_env[var] = os.environ[var]
                        del os.environ[var]
                
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=api_key)
                    logger.info("✅ OpenAI client created successfully")
                finally:
                    # Restore environment
                    for var, value in original_env.items():
                        os.environ[var] = value
                        
            else:
                logger.error(f"❌ Subprocess test failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ OpenAI wrapper initialization failed: {e}")
    
    def embeddings_create(self, **kwargs):
        """Create embeddings using the wrapped client."""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        return self.client.embeddings.create(**kwargs)
    
    def chat_completions_create(self, **kwargs):
        """Create chat completions using the wrapped client."""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        return self.client.chat.completions.create(**kwargs)
    
    @property
    def embeddings(self):
        """Access to embeddings API."""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        return self.client.embeddings
    
    @property 
    def chat(self):
        """Access to chat API."""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        return self.client.chat
    
    def is_initialized(self):
        """Check if client is properly initialized."""
        return self.client is not None