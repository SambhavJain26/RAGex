import os
import time
from playwright.sync_api import sync_playwright

output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(output_dir, exist_ok=True)

def run():
    print("Launching Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:5000...")
        page.goto("http://localhost:5000", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector(".welcome-message", timeout=15000)
        time.sleep(1)

        # 1. Capture Dashboard Overview
        p1 = os.path.join(output_dir, "01-dashboard-overview.png")
        page.screenshot(path=p1, full_page=False)
        print(f"Captured: {p1}")

        # 2. Query RAG Pipeline
        print("Querying RAG Pipeline...")
        page.fill("#message-input", "What are the pricing tiers and 2025 roadmap for Carllm?")
        page.click("#send-btn")
        page.wait_for_selector(".bot-message", timeout=30000)
        time.sleep(2)
        p2 = os.path.join(output_dir, "02-rag-query-response.png")
        page.screenshot(path=p2, full_page=False)
        print(f"Captured: {p2}")

        # 3. Query Calculator Tool
        print("Querying Calculator Tool...")
        page.fill("#message-input", "Calculate sqrt(144) + 25 * 4")
        page.click("#send-btn")
        # Wait until there are at least 2 bot messages
        page.wait_for_function("document.querySelectorAll('.bot-message').length >= 2", timeout=15000)
        time.sleep(2)
        p3 = os.path.join(output_dir, "03-calculator-tool.png")
        page.screenshot(path=p3, full_page=False)
        print(f"Captured: {p3}")

        # 4. Query Dictionary Tool
        print("Querying Dictionary Tool...")
        page.fill("#message-input", "Define serendipity")
        page.click("#send-btn")
        page.wait_for_function("document.querySelectorAll('.bot-message').length >= 3", timeout=15000)
        time.sleep(2)
        p4 = os.path.join(output_dir, "04-dictionary-tool.png")
        page.screenshot(path=p4, full_page=False)
        print(f"Captured: {p4}")

        browser.close()
        print("Screenshots captured successfully!")

if __name__ == "__main__":
    run()
