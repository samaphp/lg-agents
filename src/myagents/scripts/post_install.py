import subprocess
import sys

# Install playwright browsers
# Needed for Langgraph studio to work
def main():
    print("Installing Playwright browsers...")
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
        print("Playwright browsers installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Playwright browsers: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 