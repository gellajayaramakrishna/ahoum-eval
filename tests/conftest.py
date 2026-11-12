import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SAMPLE = """Person A: I'm really scared about the job interview tomorrow.
Person B: Don't worry, you've prepared well. Just be confident!
Person A: What if I forget everything?
Person B: Take a deep breath. You know this material better than anyone."""
