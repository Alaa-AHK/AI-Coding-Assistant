import os
import tempfile
import subprocess


class CodeExecutor:
    """Executes code in a sandboxed subprocess."""

    def execute(self, code: str, timeout: int = 30) -> dict:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'error': None if result.returncode == 0 else "Process exited with non-zero status"
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': 'Execution timed out',
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e)
            }
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
