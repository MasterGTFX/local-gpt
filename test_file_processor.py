#!/usr/bin/env python3
"""
Test script for file processing functionality
"""

import os
from file_processor import FileProcessor

def test_file_processor():
    """Test the FileProcessor functionality"""
    print("Testing FileProcessor...")

    # Initialize processor without LLM client for basic testing
    processor = FileProcessor()

    # Test file processing
    test_file = "test_file.txt"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False

    print(f"Processing file: {test_file}")
    result = processor.process_file(test_file)

    if result['success']:
        print("✅ File processed successfully!")
        print(f"   Filename: {result['file_info']['filename']}")
        print(f"   Size: {result['file_info']['size']} bytes")
        print(f"   Extension: {result['file_info']['extension']}")
        print(f"   Content length: {result.get('content_length', 0)} chars")
        print("\n--- Markdown Content Preview (first 200 chars) ---")
        content = result['markdown_content']
        print(content[:200] + "..." if len(content) > 200 else content)
        print("--- End Preview ---\n")

        # Test multiple files processing
        print("Testing multiple files processing...")
        results = processor.process_multiple_files([test_file])
        summary = processor.get_processing_summary(results)
        print(f"✅ Summary: {summary['successful']}/{summary['total_files']} files processed")

        # Test LLM context formatting
        print("Testing LLM context formatting...")
        context = processor.format_files_for_llm_context(results)
        print(f"✅ Context generated: {len(context)} characters")
        print("\n--- Context Preview (first 300 chars) ---")
        print(context[:300] + "..." if len(context) > 300 else context)
        print("--- End Context Preview ---\n")

        return True
    else:
        print(f"❌ File processing failed: {result['error']}")
        return False

if __name__ == "__main__":
    success = test_file_processor()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")