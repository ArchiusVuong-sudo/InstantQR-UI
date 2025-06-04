import os
import base64
import requests
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse
from playwright.async_api import Page
from requests.exceptions import RequestException


def is_image_accessible(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except RequestException as e:
        print(f"Request failed: {str(e)}")
        return False

async def get_text_elements(page: Page):
    """Use to get all text elements on the page that are visible in the viewport
    """
    return await page.evaluate("""
        () => {
            const nonVisualTags = new Set([
                'HTML', 'HEAD', 'SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT', 'BODY'
            ]);

            function isVisible(el) {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.visibility !== 'hidden' &&
                    style.display !== 'none' &&
                    parseFloat(style.opacity) !== 0
                );
            }

            function isInViewport(el) {
                const rect = el.getBoundingClientRect();
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= window.innerHeight &&
                    rect.right <= window.innerWidth
                );
            }

            function hasVisibleChildren(el) {
                return Array.from(el.children).some(child =>
                    !nonVisualTags.has(child.tagName.toUpperCase()) && isVisible(child)
                );
            }

            function getUniqueCssPath(el) {
                if (!(el instanceof Element)) return '';
                const path = [];

                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    if (el.id) {
                        path.unshift(`#${CSS.escape(el.id)}`);
                        break;
                    }

                    const tag = el.nodeName.toLowerCase();
                    let index = 1;
                    let sibling = el;
                    while ((sibling = sibling.previousElementSibling)) {
                        if (sibling.nodeName.toLowerCase() === tag) index++;
                    }

                    path.unshift(`${tag}:nth-of-type(${index})`);
                    el = el.parentElement;
                }

                return path.join(' > ');
            }

            return Array.from(document.querySelectorAll('*')).filter(el => {
                const tag = el.tagName.toUpperCase();
                return (
                    !nonVisualTags.has(tag) &&
                    tag !== 'IMG' &&
                    isVisible(el) &&
                    isInViewport(el) &&
                    !hasVisibleChildren(el)
                );
            }).map(el => {
                const text = (el.innerText || '').trim();
                if (!text) return null;
                
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return {
                    text: text,
                    selector: getUniqueCssPath(el),
                    fontSize: style.fontSize,
                    width: rect.width,
                    height: rect.height
                };
            }).filter(item => item !== null);
        }
    """)

async def take_viewport_screenshot(page: Page):
    viewport_height = page.viewport_size['height']
    await page.set_viewport_size({
        'width': page.viewport_size['width'],
        'height': int(viewport_height * 1.1)
    })
    
    screenshot_bytes = await page.screenshot(full_page=False)
    return base64.b64encode(screenshot_bytes).decode("utf-8")

def get_past_changes(site_folder: str = None):
    """Get the past changes from the log file in the site folder
    
    Args:
        site_folder: The folder path for the site. If None, returns empty string.
        
    Returns:
        The contents of the log file, or an empty string if the file doesn't exist
    """
    if not site_folder:
        return ""
        
    log_file = f"{site_folder}/log.txt"
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_site_folder(url: str) -> str:
    """Get the folder path for a site based on its URL
    
    Args:
        url: The URL of the site
        
    Returns:
        The folder path for the site
    """
    # Extract domain name from URL
    domain = urlparse(url).netloc
    # Remove www. if present
    domain = domain.replace('www.', '')
    # Create folder name
    folder_name = f"sites/{domain}"
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def upload_to_github(file_path: str, repo_name: str, branch: str = "main") -> Optional[str]:
    """
    Upload a file to GitHub repository
    
    Args:
        file_path: Path to the file to upload
        repo_name: Name of the repository (format: 'owner/repo')
        branch: Branch name (default: 'main')
    
    Returns:
        URL of the uploaded file if successful, None otherwise
    """
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("GITHUB_TOKEN environment variable not set")
        return None

    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Encode content to base64
    content_b64 = base64.b64encode(content).decode('utf-8')
    
    # Prepare the API request
    api_url = f"https://api.github.com/repos/{repo_name}/contents/{os.path.basename(file_path)}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get the current file SHA if it exists
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            sha = response.json()['sha']
        else:
            sha = None
    except Exception as e:
        print(f"Error checking existing file: {e}")
        sha = None

    # Prepare the data for upload
    data = {
        "message": f"Upload image variant {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": content_b64,
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    try:
        response = requests.put(api_url, headers=headers, json=data)
        if response.status_code in [201, 200]:
            # Construct the raw GitHub URL
            owner, repo = repo_name.split('/')
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{os.path.basename(file_path)}"
            print(f"Successfully uploaded to GitHub: {raw_url}")
            return raw_url
        else:
            print(f"Failed to upload to GitHub. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error uploading to GitHub: {e}")
        return None
