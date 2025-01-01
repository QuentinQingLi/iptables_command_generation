import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_page_content(url):
    """Fetches the HTML content of a webpage."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_examples_from_page(html, example_selector, command_selector):
    """Extracts iptables examples and their explanations from a webpage."""
    soup = BeautifulSoup(html, "html.parser")

    # Find all spans containing the description
    descriptions = [
        span for span in soup.select(example_selector) 
        if "This command" in span.get_text()
    ]

    data = []

    for description in descriptions:
        # Get the text of the description
        desc_text = description.get_text(strip=True)

        # Find the next <pre> tag (command block)
        command = description.find_next("pre")
        if command:
            cmd_text = command.get_text(strip=True)
            data.append({
                "description": desc_text,
                "command": cmd_text
            })

    return data

def scrape_website(url, example_selector, command_selector):
    """Scrapes iptables examples from a website."""
    html = fetch_page_content(url)
    if not html:
        return []
    return extract_examples_from_page(html, example_selector, command_selector)

def save_to_csv(data, filename="iptables_commands.csv"):
    """Saves the extracted data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved data to {filename}")

# Configuration for scraping multiple sites
sites = [
    {
        "url": "https://www.geeksforgeeks.org/iptables-command-in-linux-with-examples/",
        "example_selector": "span",  # This will fetch all <span> tags
        "command_selector": "pre"  # Replace with the actual selector for commands
    },
    # Add more sites with their respective selectors here...
]

# Main scraping loop
# Main script
if __name__ == "__main__":
    all_data = []
    for site in sites:
        print(f"Scraping {site['url']}...")
        data = scrape_website(site["url"], site["example_selector"], site["command_selector"])
        all_data.extend(data)
        time.sleep(2)  # Add a delay between requests
    
    # Save the scraped data
    save_to_csv(all_data)
