#!/usr/bin/env python3
"""
Test script to verify OpenAI client initialization in Railway environment
"""
import os
import sys

def test_openai_import():
    """Test if OpenAI can be imported properly."""
    try:
        from openai import OpenAI
        print("✅ OpenAI import successful")
        return True
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_openai_client():
    """Test OpenAI client initialization."""
    try:
        from openai import OpenAI

        # Check environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        print(f"API key present: {bool(api_key)}")
        if api_key:
            print(f"API key starts with: {api_key[:20]}...")

        if not api_key:
            print("❌ No API key found")
            return False

        # Initialize client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")

        # Test methods exist
        if hasattr(client, 'embeddings'):
            print("✅ embeddings method exists")
        else:
            print("❌ embeddings method missing")

        if hasattr(client, 'chat'):
            print("✅ chat method exists")
        else:
            print("❌ chat method missing")

        return True

    except Exception as e:
        print(f"❌ OpenAI client creation failed: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_openai_api_call():
    """Test a simple API call to verify connectivity."""
    try:
        from openai import OpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key for API test")
            return False

        client = OpenAI(api_key=api_key)

        # Test embeddings call with text-embedding-3-small
        print("Testing embeddings API call...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["test query"]
        )

        embedding = response.data[0].embedding
        print(f"✅ Embeddings API call successful, dimension: {len(embedding)}")
        return True

    except Exception as e:
        print(f"❌ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== OpenAI Integration Test ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")

    print("\n1. Testing OpenAI import...")
    import_ok = test_openai_import()

    print("\n2. Testing OpenAI client initialization...")
    client_ok = test_openai_client()

    if import_ok and client_ok:
        print("\n3. Testing OpenAI API call...")
        api_ok = test_openai_api_call()

        if api_ok:
            print("\n🎉 All tests passed! OpenAI integration should work.")
        else:
            print("\n⚠️  Client works but API call failed.")
    else:
        print("\n❌ Basic setup failed, skipping API test.")