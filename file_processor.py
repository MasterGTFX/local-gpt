import os
import hashlib
import mimetypes
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from markitdown import MarkItDown
import requests


class FileProcessor:
    """Handle file processing and conversion to markdown using MarkItDown"""

    # Maximum file size in bytes (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Documents
        '.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
        # Audio
        '.mp3', '.wav', '.m4a', '.ogg', '.flac',
        # Text formats
        '.txt', '.md', '.html', '.htm', '.csv', '.json', '.xml', '.yaml', '.yml',
        # Code files
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.sql'
    }

    def __init__(self, llm_api_key: str = None, llm_base_url: str = None):
        """Initialize file processor with optional LLM client for image descriptions"""
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url or "https://openrouter.ai/api/v1"

        # Initialize MarkItDown
        self.md_processor = None
        self._init_markitdown()

    def _init_markitdown(self):
        """Initialize MarkItDown with optional LLM client"""
        try:
            if self.llm_api_key:
                # Create a simple LLM client for MarkItDown
                class SimpleOpenAIClient:
                    def __init__(self, api_key: str, base_url: str):
                        self.api_key = api_key
                        self.base_url = base_url

                    def chat_completions_create(self, model: str, messages: List[Dict], **kwargs):
                        """Simple OpenAI-compatible chat completion for MarkItDown"""
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }

                        data = {
                            "model": model,
                            "messages": messages,
                            **kwargs
                        }

                        response = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=data
                        )
                        response.raise_for_status()
                        return response.json()

                client = SimpleOpenAIClient(self.llm_api_key, self.llm_base_url)
                self.md_processor = MarkItDown(llm_client=client, llm_model="gpt-4o-mini")
            else:
                self.md_processor = MarkItDown()
        except Exception as e:
            print(f"Warning: Could not initialize MarkItDown with LLM client: {e}")
            self.md_processor = MarkItDown()

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def is_file_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        extension = Path(file_path).suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def get_file_info(self, file_path: str) -> Dict:
        """Get basic file information"""
        path = Path(file_path)
        stat = path.stat()

        return {
            'filename': path.name,
            'size': stat.st_size,
            'extension': path.suffix.lower(),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'hash': self.calculate_file_hash(file_path)
        }

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate file for processing"""
        if not os.path.exists(file_path):
            return False, "File does not exist"

        path = Path(file_path)

        # Check file size
        if path.stat().st_size > self.MAX_FILE_SIZE:
            return False, f"File too large (max {self.MAX_FILE_SIZE // (1024*1024)}MB)"

        # Check if supported
        if not self.is_file_supported(file_path):
            return False, f"Unsupported file type: {path.suffix}"

        return True, None

    def process_file(self, file_path: str) -> Dict:
        """Process a file and convert to markdown"""
        # Validate file
        is_valid, error = self.validate_file(file_path)
        if not is_valid:
            return {
                'success': False,
                'error': error,
                'file_info': None,
                'markdown_content': None
            }

        try:
            # Get file info
            file_info = self.get_file_info(file_path)

            # Convert to markdown
            result = self.md_processor.convert(file_path)

            # Extract content and metadata
            markdown_content = result.text_content

            # Add metadata header to markdown
            metadata_header = self._create_metadata_header(file_info, len(markdown_content))
            full_markdown = f"{metadata_header}\n\n{markdown_content}"

            return {
                'success': True,
                'error': None,
                'file_info': file_info,
                'markdown_content': full_markdown,
                'content_length': len(markdown_content)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'file_info': self.get_file_info(file_path) if os.path.exists(file_path) else None,
                'markdown_content': None
            }

    def _create_metadata_header(self, file_info: Dict, content_length: int) -> str:
        """Create a metadata header for the markdown content"""
        size_mb = file_info['size'] / (1024 * 1024)

        return f"""---
File: {file_info['filename']}
Size: {size_mb:.2f} MB ({file_info['size']:,} bytes)
Type: {file_info['mime_type'] or 'Unknown'}
Extension: {file_info['extension']}
Content Length: {content_length:,} characters
Hash: {file_info['hash'][:16]}...
---"""

    def process_multiple_files(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple files"""
        results = []
        for file_path in file_paths:
            result = self.process_file(file_path)
            results.append(result)
        return results

    def format_files_for_llm_context(self, processed_files: List[Dict]) -> str:
        """Format processed files for LLM context"""
        if not processed_files:
            return ""

        successful_files = [f for f in processed_files if f['success']]
        if not successful_files:
            return ""

        # Create header
        file_names = [f['file_info']['filename'] for f in successful_files]
        header = f"[ATTACHED FILES: {', '.join(file_names)}]\n\n"

        # Combine all markdown content
        content_parts = []
        for file_data in successful_files:
            content_parts.append(file_data['markdown_content'])

        full_content = header + "\n\n---\n\n".join(content_parts)

        return full_content

    def get_processing_summary(self, results: List[Dict]) -> Dict:
        """Get summary of processing results"""
        total_files = len(results)
        successful = len([r for r in results if r['success']])
        failed = total_files - successful

        successful_files = [r['file_info']['filename'] for r in results if r['success']]
        failed_files = [(r['file_info']['filename'] if r['file_info'] else 'Unknown', r['error'])
                       for r in results if not r['success']]

        total_content_length = sum(r.get('content_length', 0) for r in results if r['success'])

        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_content_length': total_content_length
        }