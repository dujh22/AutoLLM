import arxiv as arx
import requests
import csv
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os

# Function to get the code URL from Papers with Code
def get_paper_code_url(paper_id):
    base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
    code_url = base_url + paper_id
    try:
        code_response = requests.get(code_url, verify=False).json()
        if "official" in code_response and code_response["official"]:
            github_code_url = code_response["official"]["url"]
            return github_code_url
    except:
        return None

# Function to get the star count from the GitHub repository
def get_stars(github_code_url):
    try:
        code_html = requests.get(github_code_url, verify=False)
        soup = BeautifulSoup(code_html.text, "html.parser")
        a_stars = soup.find_all("a", href=github_code_url.split("https://github.com")[-1] + "/stargazers")
        stars = a_stars[0].text.strip().split("\n")[0] if a_stars else "0"
        return stars
    except:
        return "0"

# Function to load already fetched paper IDs
def load_existing_paper_ids(filename):
    if not os.path.exists(filename):
        return set()
    
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        existing_ids = {row[0] for row in reader}
    return existing_ids

# Perform the arXiv search and save results to CSV
def fetch_and_save_arxiv_data():
    # File where results are saved
    filename = "arxiv_papers.csv"
    
    # Load existing paper IDs to support resuming
    existing_ids = load_existing_paper_ids(filename)
    
    # Set up the arXiv search with descending date order
    arxiv_search = arx.Search(
        query="Large Language Models", 
        max_results=1000, 
        sort_by=arx.SortCriterion.SubmittedDate, 
        sort_order=arx.SortOrder.Descending
    )
    
    # Open the file in append mode
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # If file is new, write header
        if not existing_ids:
            writer.writerow([
                "Paper ID", "Title", "URL", "Summary", "First Author", 
                "Publish Date", "Update Date", "Code URL", "Stars"
            ])
        
        # Use tqdm for progress tracking
        for result in tqdm(arxiv_search.results(), total=1000, desc="Fetching Papers"):
            paper_id = result.get_short_id()
            
            # Skip if paper has already been processed
            if paper_id in existing_ids:
                continue
            
            paper_title = result.title
            paper_url = result.entry_id
            paper_summary = result.summary.replace("\n", "")
            paper_first_author = result.authors[0]
            publish_time = result.published.date()
            update_time = result.updated.date()

            # Get the code URL and stars if available
            code_url = get_paper_code_url(paper_id)
            stars = get_stars(code_url) if code_url else "N/A"

            # Append paper data to CSV
            writer.writerow([
                paper_id, paper_title, paper_url, paper_summary, 
                paper_first_author, publish_time, update_time, code_url, stars
            ])
            
            # Flush the file buffer to ensure data is written
            file.flush()

            # Delay to avoid rate-limiting
            time.sleep(1)
    
    print(f"Data saved to {filename}")

# Execute the function
fetch_and_save_arxiv_data()