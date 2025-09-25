#!/usr/bin/env python3
"""
Test script for multiple file types
"""

from file_processor import FileProcessor

def test_multiple_file_types():
    """Test processing multiple file types"""
    print("Testing multiple file types...")

    processor = FileProcessor()

    # Test files
    test_files = ["test_file.txt", "test_data.json"]

    # Check which files exist
    import os
    existing_files = [f for f in test_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} test files: {existing_files}")

    if not existing_files:
        print("❌ No test files found")
        return False

    # Process multiple files
    results = processor.process_multiple_files(existing_files)
    summary = processor.get_processing_summary(results)

    print(f"\n📊 Processing Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Successful: {summary['successful']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   Total content: {summary['total_content_length']} chars")

    # Show successful files
    if summary['successful_files']:
        print(f"\n✅ Successfully processed:")
        for filename in summary['successful_files']:
            print(f"   • {filename}")

    # Show failed files
    if summary['failed_files']:
        print(f"\n❌ Failed to process:")
        for filename, error in summary['failed_files']:
            print(f"   • {filename}: {error}")

    # Test LLM context formatting
    context = processor.format_files_for_llm_context(results)
    print(f"\n🤖 LLM Context:")
    print(f"   Length: {len(context)} characters")
    print(f"   Preview (first 400 chars):")
    print("   " + "─" * 50)
    print("   " + context[:400].replace('\n', '\n   ') + "...")
    print("   " + "─" * 50)

    return summary['successful'] > 0

if __name__ == "__main__":
    success = test_multiple_file_types()
    if success:
        print("\n🎉 Multiple file type tests passed!")
    else:
        print("\n💥 Tests failed!")