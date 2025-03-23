import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jugantor_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_current_articles():
    """Get list of article URLs already scraped"""
    existing_urls = set()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if history file exists
    if os.path.exists('data/article_history.txt'):
        with open('data/article_history.txt', 'r', encoding='utf-8') as f:
            for line in f:
                existing_urls.add(line.strip())
    
    return existing_urls

def save_article_url(url):
    """Save article URL to history file"""
    with open('data/article_history.txt', 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def get_sentence_embeddings(sentences, model, tokenizer):
    """Get embeddings for a list of sentences using BERT"""
    embeddings = []
    
    for sentence in sentences:
        # Tokenize and convert to tensor
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token as sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(embedding[0])
    
    return np.array(embeddings)

def split_into_sentences(text):
    """Split Bengali text into sentences"""
    # Split on Bengali sentence terminators (danda), exclamation marks, and question marks
    sentences = re.split(r'[।!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def summarize_bengali_with_rag(text, model, tokenizer, num_sentences=3):
    """
    Summarize Bengali text using a RAG-inspired approach:
    1. Split text into sentences
    2. Get sentence embeddings
    3. Calculate sentence importance based on centrality
    4. Select top sentences maintaining original order
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # Clean text
    text = clean_text(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # If text is already short, return as is
    if len(sentences) <= num_sentences:
        return text
    
    # Get sentence embeddings
    embeddings = get_sentence_embeddings(sentences, model, tokenizer)
    
    # Calculate sentence centrality (similarity to other sentences)
    similarity_matrix = cosine_similarity(embeddings)
    centrality_scores = np.sum(similarity_matrix, axis=1)
    
    # Get indices of top sentences by centrality
    top_indices = np.argsort(centrality_scores)[-num_sentences:]
    
    # Sort indices to maintain original order
    top_indices = sorted(top_indices)
    
    # Construct summary from selected sentences
    selected_sentences = [sentences[i] for i in top_indices]
    summary = '। '.join(selected_sentences) + '।'
    
    return summary

def scrape_and_summarize_jugantor(model, tokenizer):
    """Scrape latest news articles from Jugantor and summarize them"""
    logger.info(f"Starting scrape cycle...")
    
    # Get already scraped article URLs
    existing_urls = get_current_articles()
    
    # Headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    }
    
    # Base URL and category URLs
    base_url = "https://www.kalerkantho.com"
    category_urls = {
        "latest": "https://www.kalerkantho.com/online/latest",
        "national": "https://www.kalerkantho.com/online/national",
        "economics": "https://www.kalerkantho.com/online/economics",
        "international": "https://www.kalerkantho.com/online/international",
        "sports": "https://www.kalerkantho.com/online/sports",
        "entertainment": "https://www.kalerkantho.com/online/entertainment",
        "job-seek": "https://www.kalerkantho.com/online/job-seek"
    }
    
    # Create or append to CSV file
    csv_filename = f'data/jugantor_news_with_summaries_{datetime.now().strftime("%Y%m%d")}.csv'
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow([
                'title', 'full_content', 'rag_summary', 'image_url', 
                'article_url', 'published_at', 'scraped_at', 'category'
            ])
        
        try:
            # Find article links from all category pages
            article_links = []
            
            # First get links from the homepage
            logger.info("Fetching homepage to extract article links...")
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            homepage_links = get_article_links_from_page(soup, base_url, "homepage")
            article_links.extend(homepage_links)
            logger.info(f"Found {len(homepage_links)} article links on homepage")
            
            # Then get links from each category page
            for category, url in category_urls.items():
                try:
                    logger.info(f"Fetching category page: {category} ({url})")
                    time.sleep(random.uniform(1, 2))  # Small delay between requests
                    
                    category_response = requests.get(url, headers=headers)
                    category_response.raise_for_status()
                    
                    category_soup = BeautifulSoup(category_response.text, 'html.parser')
                    category_links = get_article_links_from_page(category_soup, base_url, category)
                    article_links.extend(category_links)
                    
                    logger.info(f"Found {len(category_links)} article links in {category}")
                    
                except Exception as e:
                    logger.error(f"Error fetching category {category}: {str(e)}")
            
            # Deduplicate links (same URL might appear in multiple categories)
            unique_links = {}
            for url, category in article_links:
                if url not in unique_links:
                    unique_links[url] = category
            
            # Convert back to list of tuples
            article_links = [(url, category) for url, category in unique_links.items()]
            
            logger.info(f"Found a total of {len(article_links)} unique article links")
            
            # Filter out already scraped articles
            new_articles = [(url, cat) for url, cat in article_links if url not in existing_urls]
            
            if not new_articles:
                logger.info("No new articles found in this cycle.")
                return 0
            
            logger.info(f"Found {len(new_articles)} new article links. Processing...")
            
            # Process each article
            articles_scraped = 0
            for i, (article_url, category) in enumerate(new_articles):
                try:
                    # Add a random delay to avoid being blocked
                    time.sleep(random.uniform(2, 3))
                    
                    logger.info(f"Fetching article {i+1}/{len(new_articles)}: {article_url}")
                    article_response = requests.get(article_url, headers=headers)
                    article_response.raise_for_status()
                    
                    # Parse article HTML
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    
                    # Log the HTML structure for debugging (only the first article)
                    if i == 0:
                        with open('debug_article_html.html', 'w', encoding='utf-8') as f:
                            f.write(article_soup.prettify())
                        logger.info("Saved first article HTML for debugging")
                    
                    # Extract title
                    title = extract_title(article_soup)
                    
                    # Extract publication date
                    published_at = extract_published_date(article_soup)
                    
                    # Extract main image
                    image_url = extract_image_url(article_soup)
                    
                    # Extract content
                    article_content = extract_article_content(article_soup)
                    
                    # Current timestamp
                    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Generate RAG summary for the article
                    logger.info(f"Generating RAG summary for article {i+1}...")
                    rag_summary = ""
                    if article_content and article_content != "Content extraction failed":
                        rag_summary = summarize_bengali_with_rag(article_content, model, tokenizer)
                        logger.info(f"Summary generated ({len(rag_summary)} characters)")
                    else:
                        rag_summary = "Unable to generate summary - no content available"
                    
                    # Write to CSV
                    writer.writerow([
                        title, article_content, rag_summary, image_url, 
                        article_url, published_at, scraped_at, category
                    ])
                    
                    # Add to history
                    save_article_url(article_url)
                    
                    logger.info(f"Saved article with summary successfully")
                    articles_scraped += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article {article_url}: {str(e)}")
                    
            return articles_scraped
                    
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return 0

def get_article_links_from_page(soup, base_url, category):
    """Extract article links from a page"""
    article_links = []
    
    # Try different selectors for article links
    link_selectors = [
        'a.link_overlay',       # Common for featured articles
        '.headline a',           # Headlines
        'h1 a',                  # Main headlines
        'h2 a',                  # Secondary headlines
        '.news_title a',         # News titles
        '.image_box a',          # Image boxes with links
        '.list_item a',          # List items
        '.lead-news a',          # Lead news
        '.news-card a',          # News cards
        'article a',             # Article links
        '.news-feed a'           # News feed items
    ]
    
    for selector in link_selectors:
        links = soup.select(selector)
        
        for link in links:
            href = link.get('href')
            if href and '/video/' not in href and '/gallery/' not in href:
                # Make sure it's a full URL
                if not href.startswith('http'):
                    href = base_url.rstrip('/') + href if href.startswith('/') else base_url + href
                
                # Add to article links if it seems like an article (has a number in the URL)
                if re.search(r'/\d+/?$', href):  # Most article URLs end with a number
                    article_links.append((href, category))
    
    return article_links

def extract_title(soup):
    """Extract the article title from soup"""
    title = "No title found"
    for title_selector in ['h1.headline', 'h1', '.article-headline', '.news-headline', '.news_title']:
        title_element = soup.select_one(title_selector)
        if title_element:
            title = clean_text(title_element.text)
            break
    return title

def extract_published_date(soup):
    """Extract the publication date from soup"""
    published_at = ""
    for date_selector in ['.report-time time', '.published-time', '.time-stamp', 'time', '.reporter_info', '.date']:
        date_element = soup.select_one(date_selector)
        if date_element:
            published_at = date_element.get('datetime', '')
            if not published_at:
                published_at = clean_text(date_element.text)
            break
    return published_at

def extract_image_url(soup):
    """Extract the main image URL from soup"""
    image_url = ""
    for img_selector in ['.report-header-image img', '.featured-image img', '.article-image img', 'figure img', '.news-image img', '.image_box img']:
        image_element = soup.select_one(img_selector)
        if image_element:
            image_url = image_element.get('src', '')
            if not image_url:
                image_url = image_element.get('data-src', '')
            if image_url:
                break
    return image_url

def extract_article_content(soup):
    """Extract the article content from soup"""
    article_content = ""
    
    # Try different selectors for content
    content_selectors = [
        '.report-text',                # Main content area
        '.news-content',               # News content area
        '.article-content',            # Article content
        '.report-content',             # Report content
        '.news_content',               # Alternative news content
        '.description',                # Description
        '.news-element-text',          # News elements
        '.news-details'                # News details
    ]
    
    for selector in content_selectors:
        content_elements = soup.select(selector)
        if content_elements:
            for element in content_elements:
                element_text = element.get_text(strip=True)
                if element_text:
                    article_content += element_text + "\n\n"
            if article_content:
                break
    
    # If no content found with selectors, try to get all paragraphs in the main area
    if not article_content:
        # Look for article or main content container
        main_content = None
        for container in ['.news_content', 'article', '.report-content', 'main', '.content-area']:
            main_content = soup.select_one(container)
            if main_content:
                break
        
        # If we found a main content container, get all paragraphs within it
        if main_content:
            paragraphs = main_content.select('p')
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if p_text and len(p_text) > 20:  # Only add non-trivial paragraphs
                    article_content += p_text + "\n\n"
        else:
            # Last resort: get all paragraphs, filtering out navigation/headers/footers
            paragraphs = soup.select('p')
            main_paragraphs = [p for p in paragraphs if 
                             not p.find_parent('header') and 
                             not p.find_parent('footer') and 
                             not p.find_parent('nav')]
            for p in main_paragraphs:
                p_text = p.get_text(strip=True)
                if p_text and len(p_text) > 20:  # Only add non-trivial paragraphs
                    article_content += p_text + "\n\n"
    
    # Clean up the content
    article_content = article_content.strip()
    
    if not article_content:
        article_content = "Content extraction failed"
    
    return article_content

def main():
    logger.info("Starting Jugantor News Crawler and Summarizer")
    logger.info("Press Ctrl+C to stop the program")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load model and tokenizer for summarization - using multilingual BERT
    logger.info("Loading BERT model for summarization...")
    model_name = "bert-base-multilingual-cased"  # Supports Bengali
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    logger.info("Model loaded successfully")
    
    # Keep track of statistics
    total_articles = 0
    cycles = 0
    
    try:
        while True:
            # Run the scraper and summarizer
            new_articles = scrape_and_summarize_jugantor(model, tokenizer)
            total_articles += new_articles
            cycles += 1
            
            # Print statistics
            logger.info(f"Cycle {cycles} completed.")
            logger.info(f"Total articles scraped and summarized so far: {total_articles}")
            
            # Save a summary report
            with open('data/scrape_report.txt', 'w', encoding='utf-8') as f:
                f.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total cycles: {cycles}\n")
                f.write(f"Total articles: {total_articles}\n")
            
            # Wait before next cycle (30 minutes)
            next_scrape_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Next scrape scheduled at: {next_scrape_time}")
            
            wait_minutes = 30
            logger.info(f"Waiting for {wait_minutes} minutes before next scrape cycle...")
            for i in range(wait_minutes):
                time.sleep(60)  # Wait 1 minute
                minutes_left = wait_minutes - i - 1
                if minutes_left > 0 and minutes_left % 5 == 0:  # Log only every 5 minutes
                    logger.info(f"{minutes_left} minutes remaining until next scrape...")
    
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")
        logger.info(f"Total articles scraped and summarized: {total_articles}")
        logger.info(f"Total scrape cycles completed: {cycles}")
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.info("Program terminated due to error.")

if __name__ == "__main__":
    main()
