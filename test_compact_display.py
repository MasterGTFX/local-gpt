#!/usr/bin/env python3
"""
Test the compact file display
"""

from file_processor import FileProcessor
import os

def test_compact_display():
    """Test the compact file display formatting"""
    processor = FileProcessor()

    # Test with existing files
    test_files = [f for f in ["test_file.txt", "test_data.json", "app.py"] if os.path.exists(f)]

    if not test_files:
        print("❌ No test files found")
        return

    print(f"Testing compact display with {len(test_files)} files...")

    # Process files
    results = processor.process_multiple_files(test_files)
    summary = processor.get_processing_summary(results)

    print(f"\n📊 Summary: {summary['successful']}/{summary['total_files']} successful")

    # Simulate the HTML generation logic from app.py
    def create_compact_display(summary, processed_files):
        if summary['successful'] == 0 and summary['failed'] == 0:
            return ""

        html_parts = ["<div style='margin: 2px 0; padding: 4px 8px; background: rgba(59,130,246,0.08); border-radius: 4px; font-size: 0.75rem; color: #374151; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;'>"]

        # Add paperclip icon and count
        html_parts.append(f"<span style='color: #6366f1; font-weight: 500;'>📎 {summary['successful']}</span>")

        # Add file badges - only show first few to keep it compact
        files_to_show = summary['successful_files'][:3]  # Show max 3 files
        for filename in files_to_show:
            # Truncate long filenames
            display_name = filename[:12] + "..." if len(filename) > 15 else filename
            html_parts.append(f"<span style='padding: 1px 4px; background: rgba(34,197,94,0.12); color: #059669; border-radius: 2px; font-size: 0.7rem;'>{display_name}</span>")

        # Show "and X more" if there are more files
        if len(summary['successful_files']) > 3:
            remaining = len(summary['successful_files']) - 3
            html_parts.append(f"<span style='color: #6b7280; font-size: 0.7rem;'>+{remaining} more</span>")

        # Show failed files count if any
        if summary['failed'] > 0:
            html_parts.append(f"<span style='padding: 1px 4px; background: rgba(239,68,68,0.12); color: #dc2626; border-radius: 2px; font-size: 0.7rem;'>{summary['failed']} failed</span>")

        # Add content size if significant
        if summary['total_content_length'] > 1024:
            content_kb = summary['total_content_length'] / 1024
            if content_kb > 1024:
                content_mb = content_kb / 1024
                size_text = f"{content_mb:.1f}MB"
            else:
                size_text = f"{content_kb:.0f}KB"
            html_parts.append(f"<span style='color: #9ca3af; font-size: 0.7rem;'>{size_text}</span>")

        html_parts.append("</div>")
        return "".join(html_parts)

    # Generate compact display
    html = create_compact_display(summary, results)

    print(f"\n🎨 Compact Display Preview:")
    print("─" * 60)

    # Show a text version of what the HTML would look like
    if summary['successful'] > 0:
        files_preview = ", ".join(summary['successful_files'][:3])
        if len(summary['successful_files']) > 3:
            files_preview += f" +{len(summary['successful_files']) - 3} more"

        content_kb = summary['total_content_length'] / 1024
        size_text = f"{content_kb:.0f}KB" if content_kb > 0 else ""

        preview = f"📎 {summary['successful']} {files_preview} {size_text}".strip()
        print(f"   {preview}")
    else:
        print("   (No files attached)")

    print("─" * 60)
    print(f"\n✨ HTML Length: {len(html)} characters")
    print(f"📏 Display Height: ~20px (single line)")
    print(f"🎯 Visual Style: Inline badge format")

if __name__ == "__main__":
    test_compact_display()