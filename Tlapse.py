#!/usr/bin/env python3
# Tlapse.py
"""
Entry script — запускає інструмент у інтерактивному режимі.
Запуск: python Tlapse.py
Питає користувача шлях до джерела, FPS, вибір режиму, hw-енкодер.
"""
import os
import sys
from engine.core import run_interactive

if __name__ == "__main__":
    try:
        run_interactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
