import os
import base64
import requests
import tempfile
import streamlit as st
import asyncio
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from openai import OpenAI

from agents import Agent, ModelSettings, Runner, TResponseInputItem
from core.model import JSOutput, Evaluation, ImageChange
from core.util import get_text_elements, take_viewport_screenshot, get_past_changes, get_site_folder, is_image_accessible, upload_to_github

os.system('playwright install')
os.system('playwright install-deps')

# Load environment variables
load_dotenv()

# Default instructions for agents
DEFAULT_IMAGE_SPECIALIST_INSTRUCTIONS = """
<role>
    You are an image specialist. You are given a list of images and you need to find the best image to change.
</role>

<task>
    Your task is to analyze all images on a webpage and identify the one that would benefit most from optimization for conversion rate improvement.
    You should consider:
    - Image placement and context
    - Current emotional impact
    - Potential for improvement
    - Relevance to conversion goals
</task>

<chain_of_thought>
    1. Review all provided images and their context
    2. Analyze each image's current impact and purpose
    3. Identify images that could be optimized for better conversion
    4. Select the single most impactful image to change
    5. Provide clear instructions for the desired change
</chain_of_thought>

<constraints>
    <select_single_image>true</select_single_image>
    <focus_on_conversion>true</focus_on_conversion>
    <preserve_brand>true</preserve_brand>
    <output_format>Clear, actionable instructions for image modification</output_format>
</constraints>
"""

DEFAULT_CODER_INSTRUCTIONS = """
<role>
    You are a top-tier conversion copywriter and CRO strategist. You specialize in writing high-impact website copy that is clear, emotionally resonant, and optimized for user behavior.
</role>

<task>
    Your task is to generate a single change for one specific header or button of a webpage, with the goal of increasing conversion rate while preserving the original tone, meaning, and visual layout.
    Your code is always clean, readable, and precise. You follow requirements and visual details meticulously.
    You create scripts that run directly in the browser console, so your scripts do not use addEventListener for DOM loaded.
    You always check the output of the script and fix the code immediately if there is any error until the script works flawlessly.
    You always output the final change in the format of a script that can be run directly in the browser console.
</task>

<chain_of_thought>
    1. Understand the website: Review the business goal, brand tone, key messaging, and target audience based on the screenshot and selectors.
    2. Identify the test area: Focus on high-impact zones like the hero section, headers, or call-to-actions — especially elements above the fold or visually dominant.
    3. Generate copy variants: Brainstorm alternate versions that preserve the original intent and tone. Keep the new copy within ±5 characters of the original to avoid breaking layout or formatting.
    4. Choose the strongest variant: Pick the one most likely to increase conversion rate based on clarity, emotion, and persuasiveness.
    5. Create a script that can be run directly in the browser console to apply the change.
</chain_of_thought>

<constraints>
    <preserve_tone>true</preserve_tone>
    <preserve_meaning>true</preserve_meaning>
    <length_variation>±5 characters</length_variation>
    <output_format>Plain text only — single variant</output_format>
</constraints>
"""

DEFAULT_EVALUATOR_INSTRUCTIONS = """
<role>
    You are an elite UI/UX validator with absolute precision. You specialize in catching visual flaws caused by copy changes on live webpages. You are trusted to review high-stakes conversion tests where even a 1px shift or hidden character can damage the brand's credibility or performance.
</role>

<task>
    Your task is to **inspect a screenshot** of a webpage where **a single line of text has been modified** (typically a header or call-to-action). You must determine with total confidence whether the change has introduced **any visual defect**.

    You must check:
    - Text clipping or cut-off (even a single pixel)
    - Text wrapping or breaking into multiple lines when it shouldn't - usually the container size should be the same as the original
    - Overflowing text that escapes its container
    - Misalignment within buttons, sections, or containers
    - Padding/margin violations that cause tightness or crowding
    - Imbalance in layout symmetry caused by new copy
    - Font size or line-height disruption

    If you spot even **minor defects**, you must clearly state what is wrong and what must be fixed. You are extremely strict — you never approve a flawed change.

    If everything looks perfect, you confirm that the layout is preserved, the copy fits naturally, and nothing breaks visually or emotionally.
</task>

<chain_of_thought>
    1. Identify the text element that was changed.
    2. Examine it pixel-by-pixel within its container and surroundings.
    3. Check for cut-off letters, broken alignment, wrapping, or overflow.
    4. Compare how the new text visually integrates into the layout.
    5. Make a binary decision: PASS (no issues) or FAIL (with detailed, actionable feedback).
</chain_of_thought>

<constraints>
    <visual_input>screenshot_only</visual_input>
    <no_wrapping_allowed>true</no_wrapping_allowed>
    <no_clipping>true</no_clipping>
    <no_overflow>true</no_overflow>
    <must_maintain_spacing_and_padding>true</must_maintain_spacing_and_padding>
    <feedback_format>Plain text only — must be direct, assertive, and specific</feedback_format>
</constraints>
"""

DEFAULT_IMAGE_GENERATION_PROMPT = """
<role>
    You are a world-class visual designer and creative director. You specialize in generating emotionally compelling, conversion-optimized visuals for websites.
</role>

<task>
    Your task is to generate a visual variant for a specific image on a website. The goal is to improve user engagement or conversion rate while preserving the original intent, emotion, and layout balance.
</task>

<requirements>
    {image_specialist_result.final_output.change_description}
</requirements>

<chain_of_thought>
    1. Understand the website and image context: Use the screenshot to assess the brand, target audience, and conversion goal.
    2. Identify high-impact visual elements: Focus on imagery in the hero section or near CTAs—especially images that convey emotion, value, or identity.
    3. Brainstorm possible visual changes: Consider new compositions, subject focus, lighting, color palette, or story elements. Preserve core meaning and emotional tone.
    4. Choose the most impactful direction: Select the idea most likely to increase conversions based on visual clarity, emotional resonance, and alignment with the product or message.
    5. Output the final visual variant: Do not include process steps or alternatives—just return the best image variant idea.
</chain_of_thought>

<constraints>
    <preserve_emotion>true</preserve_emotion>
    <preserve_layout>true</preserve_layout>
    <style>Modern, clean, conversion-focused</style>
    <background>Transparent</background>
    <composition>Centered, well-balanced</composition>
    <quality>High-resolution, web-optimized, realistic</quality>
</constraints>

<examples>
    <example>
        <original>A clean dashboard on a laptop screen</original>
        <variant>A laptop on a minimalist desk showing the dashboard, with soft shadows and ambient lighting</variant>
    </example>
    <example>
        <original>Cloud icon with arrows</original>
        <variant>A stylized cloud syncing files across multiple devices</variant>
    </example>
    <example>
        <original>A woman using a mobile app</original>
        <variant>A close-up of a finger interacting with the app on a smartphone, focus on interface details</variant>
    </example>
    <example>
        <original>Shipping boxes on a shelf</original>
        <variant>Stacked branded shipping boxes in a warehouse, lit from above with soft shadows</variant>
    </example>
    <example>
        <original>Server racks in a data center</original>
        <variant>A glowing server rack in a darkened room, with blue ambient lighting and visual depth</variant>
    </example>
    <example>
        <original>A user profile interface on screen</original>
        <variant>A zoomed-in mockup of the profile UI on a tablet, floating in a clean 3D space</variant>
    </example>
    <example>
        <original>A person doing yoga in a studio</original>
        <variant>A yoga mat laid out on a balcony at sunrise, with a water bottle and plants in frame</variant>
    </example>
    <example>
        <original>Product mockup on a white background</original>
        <variant>The product angled on a marble surface with soft shadows and reflections</variant>
    </example>
    <example>
        <original>Checklist icon in a circle</original>
        <variant>A 3D-rendered clipboard with animated checkmarks floating above a pastel backdrop</variant>
    </example>
</examples>
"""

DEFAULT_IMAGE_EVALUATOR_INSTRUCTIONS = """
<role>
    You are an elite visual design validator with absolute precision. You specialize in catching visual flaws and ensuring optimal image quality and placement on webpages. You are trusted to review high-stakes image changes where even minor visual issues can damage the brand's credibility or performance.
</role>

<task>
    Your task is to **inspect a screenshot** of a webpage where **a single image has been modified**. You must determine with total confidence whether the change has introduced **any visual defect**.

    You must check:
    - Image quality and resolution
    - Proper scaling and aspect ratio
    - Visual consistency with surrounding elements
    - Proper alignment within containers
    - Color harmony with the page
    - Loading and display issues
    - Responsive behavior
    - Visual hierarchy and balance

    If you spot even **minor defects**, you must clearly state what is wrong and what must be fixed. You are extremely strict — you never approve a flawed change.

    If everything looks perfect, you confirm that the image fits naturally, maintains visual harmony, and nothing breaks visually or emotionally.
</task>

<chain_of_thought>
    1. Identify the image that was changed
    2. Examine it pixel-by-pixel within its container and surroundings
    3. Check for quality issues, scaling problems, or visual inconsistencies
    4. Compare how the new image visually integrates into the layout
    5. Make a binary decision: PASS (no issues) or FAIL (with detailed, actionable feedback)
</chain_of_thought>

<constraints>
    <visual_input>screenshot_only</visual_input>
    <no_quality_issues>true</no_quality_issues>
    <no_scaling_issues>true</no_scaling_issues>
    <no_alignment_issues>true</no_alignment_issues>
    <must_maintain_visual_harmony>true</must_maintain_visual_harmony>
    <feedback_format>Plain text only — must be direct, assertive, and specific</feedback_format>
</constraints>
"""

async def process_image_optimization(url, image_specialist_instructions, image_generation_prompt, image_evaluator_instructions=DEFAULT_IMAGE_EVALUATOR_INSTRUCTIONS):
    image_specialist = Agent(
        name="image_specialist",
        instructions=image_specialist_instructions,
        model="o3",
        output_type=ImageChange,
        model_settings=ModelSettings(
            reasoning={"effort": "high"}
        )
    )

    image_evaluator = Agent(
        name="image_evaluator",
        instructions=image_evaluator_instructions,
        model="o3",
        output_type=Evaluation,
        model_settings=ModelSettings(
            reasoning={"effort": "high"}
        )
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            java_script_enabled=True,
            bypass_csp=True,
            viewport={'width': 1920, 'height': 800}
        )
        
        page = await context.new_page()
        await page.goto(url, wait_until='domcontentloaded')

        # Get all image elements
        images = await page.query_selector_all('img')

        # Collect image information
        image_list = []
        for img in images:
            src = await img.get_attribute('src')
            alt = await img.get_attribute('alt')
            
            # Skip SVG and small images
            if src and not src.lower().endswith('.svg'):
                # Convert relative URL to absolute URL
                absolute_src = urljoin(url, src)
                
                # Check if image is accessible
                if not is_image_accessible(absolute_src):
                    continue
                
                # Get image dimensions
                width = await img.evaluate('img => img.naturalWidth')
                height = await img.evaluate('img => img.naturalHeight')
                       
                # Only include images larger than 300x300 pixels
                if width > 300 and height > 300:
                    image_list.append({
                        'src': absolute_src,
                        'alt': alt,
                        'width': width,
                        'height': height,
                        'size': width * height,  # total pixels
                    })

        # Sort images by size in descending order
        image_list.sort(key=lambda x: x['size'], reverse=True)

        # Only keep the first 10 images
        image_list = image_list[:10]

        # Print out images for debugging
        if len(image_list) == 0:
            print("No images found matching criteria")
            await browser.close()
            return None

        # Take the entire page screenshot
        screenshot = await page.screenshot(full_page=True)

        # Convert the screenshot to base64
        screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
        
        # Run the image specialist agent
        image_specialist_result = await Runner.run(
            image_specialist,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"""
Among all of the images within this site, which image should we change and how should we change it for increasing the conversion rate of the entire website ? First, this is the entire site screenshot:
                    """
                        },
                        {
                            "type": "input_image",
                            "detail": "high",
                            "image_url": f"data:image/jpeg;base64,{screenshot_base64}",
                        },
                        *[item for image in image_list for item in [
                            {
                                "type": "input_text",
                                "text": f"""
                                This image url is: {image['src']}
                                """
                            },
                            {
                                "type": "input_image",
                                "detail": "high",
                                "image_url": image['src']
                            }
                        ]]
                    ],
                }
            ],
        )

        if not hasattr(image_specialist_result, 'final_output'):
            print("Error: Image specialist did not return expected output format")
            await browser.close()
            return None

        # Create sites directory structure
        site_name = urlparse(url).netloc.replace('www.', '')
        sites_dir = os.path.join('sites', site_name, 'img')
        os.makedirs(sites_dir, exist_ok=True)

        # Initialize input items for iterative evaluation
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
The image has been changed according to the following description:
{image_specialist_result.final_output.change_description}

Please evaluate if the new image is visually correct and properly integrated into the page.
"""
                    }
                ]
            }
        ]

        while True:
            try:
                # Get the image URL from the specialist's result
                image_url = image_specialist_result.final_output.image_url

                # Download the image from the URL
                response = requests.get(image_url)
                if response.status_code != 200:
                    print(f"Error: Failed to download image from {image_url}. Status code: {response.status_code}")
                    break
                
                # Convert the image to PNG format using PIL
                img = Image.open(BytesIO(response.content))
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    img.save(temp_file.name, format='PNG')
                    temp_file_path = temp_file.name

                # Initialize OpenAI client
                client = OpenAI()

                # Open the temporary file and pass it to the API
                with open(temp_file_path, 'rb') as image_file:
                    edit_result = client.images.edit(
                        model="gpt-image-1",
                        image=image_file,
                        prompt=image_generation_prompt.format(image_specialist_result=image_specialist_result),
                        quality='high',
                        background="transparent"
                    )

                # Clean up the temporary file
                os.unlink(temp_file_path)

                image_base64 = edit_result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)

                # Save the image to the sites directory
                import uuid
                filename = os.path.join(sites_dir, f"{uuid.uuid4()}.png")
                with open(filename, "wb") as f:
                    f.write(image_bytes)

                # Upload the generated image to GitHub
                repo_name = os.getenv('GITHUB_REPO')
                new_image_url = upload_to_github(filename, repo_name)

                # Replace the image in the page
                js_code = f"""
const targetUrl = new URL('{image_specialist_result.final_output.image_url}', window.location.href).href;
const imgs = document.getElementsByTagName('img');
for (const img of imgs) {{
    const imgUrl = new URL(img.src, window.location.href).href;
    // Normalize URLs by removing 'www.' for comparison
    const normalizedTargetUrl = targetUrl.replace('www.', '');
    const normalizedImgUrl = imgUrl.replace('www.', '');
    if (normalizedImgUrl === normalizedTargetUrl) {{
        img.src = '{new_image_url}';
        break;
    }}
}}
                """
                await page.evaluate(js_code)

                try:
                    # Wait for the image to load and be visible
                    await page.wait_for_load_state('networkidle')
                    
                    # Wait for any image that contains the GitHub URL to be visible
                    await page.wait_for_selector(f'img[src*="raw.githubusercontent.com"]', state='visible', timeout=15000)
                    
                    # Additional checks to ensure image is fully loaded
                    await page.evaluate("""
                        () => {
                            const img = document.querySelector('img[src*="raw.githubusercontent.com"]');
                            return new Promise((resolve) => {
                                if (img.complete) {
                                    resolve();
                                } else {
                                    img.onload = () => resolve();
                                }
                            });
                        }
                    """)
                    
                    # Additional delay to ensure rendering is complete
                    await page.wait_for_timeout(2000)

                    # Take a screenshot of the page
                    screenshot = await page.screenshot(full_page=True)
                    screenshot_path = os.path.join(sites_dir, "after_replacement.png")
                    with open(screenshot_path, "wb") as f:
                        f.write(screenshot)
                    
                    # Convert screenshot to base64 for display
                    screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")

                    # Add screenshot to input items
                    input_items.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "detail": "high",
                                "image_url": f"data:image/jpeg;base64,{screenshot_base64}"
                            }
                        ]
                    })

                    # Run the image evaluator agent
                    print("Running image evaluator agent...")
                    evaluator_result = await Runner.run(image_evaluator, input_items)
                    result: Evaluation = evaluator_result.final_output
                    
                    print(result)
                    print('-' * 80)

                    if result.score == "pass":
                        # Save the final screenshot
                        with open(f"{sites_dir}/log.txt", "a") as f:
                            f.write(f"- {image_specialist_result.final_output.change_description}\n")
                        
                        print(f"Results saved to folder: {sites_dir}")
                        break

                    # Add feedback to input items for next iteration
                    input_items.append({
                        "role": "user",
                        "content": f"Feedback: {result.feedback}"
                    })

                except Exception as e:
                    print(f"Warning: Could not capture screenshot with new image: {e}")
                    break

            except Exception as e:
                print(f"Error processing image: {e}")
                break

        await browser.close()
        return {
            'image_specialist_response': image_specialist_result.final_output.change_description,
            'original_image_url': image_specialist_result.final_output.image_url,
            'new_image_url': new_image_url if 'new_image_url' in locals() else None,
            'screenshot': screenshot_base64 if 'screenshot_base64' in locals() else None,
            'js_code': js_code if 'js_code' in locals() else None,
            'evaluator_response': result.feedback if 'result' in locals() else None
        }

async def process_text_optimization(url, coder_instructions, evaluator_instructions):
    coder = Agent(
        name="coder",
        instructions=coder_instructions,
        model="o3",
        output_type=JSOutput,
        model_settings=ModelSettings(
            reasoning={"effort": "high"}
        )
    )

    evaluator = Agent(
        name="evaluator",
        instructions=evaluator_instructions,
        model="o3",
        output_type=Evaluation,
        model_settings=ModelSettings(
            reasoning={"effort": "high"}
        )
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            java_script_enabled=True,
            bypass_csp=True,
            viewport={'width': 1920, 'height': 800}
        )
        
        page = await context.new_page()
        await page.goto(url, wait_until='domcontentloaded')

        # Create site folder
        site_folder = get_site_folder(url)

        try:
            await page.wait_for_load_state('networkidle', timeout=60000)  # 60 second timeout
        except Exception as e:
            print(f"Warning: Network idle timeout - {str(e)}")
        
        # Wait for all images to be loaded and dynamic content to be loaded
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=30000)
        except Exception as e:
            print(f"Warning: DOM content loaded timeout - {str(e)}")
        
        # Set viewport to 110% height
        viewport_height = page.viewport_size['height']
        await page.set_viewport_size({
            'width': page.viewport_size['width'],
            'height': int(viewport_height * 1.1)
        })

        # Get text elements
        text_elements = await get_text_elements(page)

        # Take viewport screenshot
        base64_image = await take_viewport_screenshot(page)

        # Get past changes
        past_changes = get_past_changes(site_folder)

        input_items: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": f"""
# Overview
Based on the screenshot, please generate a single change for one specific header or button of a webpage, with the goal of increasing conversion rate while preserving the original tone, meaning, and visual layout.

# Past changes - do not repeat the same change, can be same selector but different text:
{past_changes}

# Text elements selectors:
{text_elements}
""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image", 
                        "detail": "high", 
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]

        while True:
            print("Running coder agent...")
            coder_result = await Runner.run(
                coder,
                input_items
            )

            # Save the JavaScript to a file
            with open(f"{site_folder}/script.js", "w") as f:
                f.write(coder_result.final_output.script)

            input_items = coder_result.to_input_list()

            # Run the script
            await page.evaluate(coder_result.final_output.script)

            base64_image = await take_viewport_screenshot(page)
            input_items.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "The result screenshot is below. Please analyze the change and give suggestions on how to improve it visually. Also check if the change is still working as expected."
                    },
                    {
                        "type": "input_image", 
                        "detail": "high", 
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            })

            print("Running evaluator agent...")
            evaluator_result = await Runner.run(evaluator, input_items)
            result: Evaluation = evaluator_result.final_output
            
            print(result)
            print('-' * 80)

            if result.score == "pass":
                # Run the script one final time and take screenshot
                await page.evaluate(coder_result.final_output.script)
                final_screenshot = await take_viewport_screenshot(page)
                
                # Save the final screenshot
                screenshot_data = base64.b64decode(final_screenshot)
                
                with open(f"{site_folder}/log.txt", "a") as f:
                    f.write(f"- {coder_result.final_output.change_description}\n")
                
                print(f"Results saved to folder: {site_folder}")
                break

            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})
        
        await browser.close()
        return {
            'coder_response': coder_result.final_output.change_description,
            'evaluator_response': result.feedback,
            'screenshot': final_screenshot,
            'script': coder_result.final_output.script
        }

def main():
    # Set page config
    st.set_page_config(
        page_title="Website Optimization Tool",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    with st.sidebar:
        st.title("Website Optimization Tool")
        st.markdown("---")
        
        # Add OpenAI API key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable image generation and optimization features"
        )
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API key set successfully!")
            api_key_set = True
        else:
            st.warning("Please enter your OpenAI API key to use all features")
            api_key_set = False
            
        st.markdown("---")
        st.markdown("""
        ### How to use
        1. Enter a website URL
        2. Choose optimization type
        3. Customize instructions (optional)
        4. Run optimization
        5. Review results
        """)
        st.markdown("---")
        st.markdown("Made with ❤️ by InstantQR")

    # Main content
    st.title("Website Optimization Tool")
    st.markdown("Optimize your website for better conversion rates through image and text optimization.")

    # URL input with validation
    url = st.text_input(
        "Enter website URL:",
        placeholder="https://www.example.com",
        help="Enter the full URL of the website you want to optimize"
    )

    if not url:
        st.warning("Please enter a website URL to begin optimization.")
        return

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Create tabs
    tab1, tab2 = st.tabs(["🖼️ Image Optimization", "📝 Text Optimization"])
    
    with tab1:
        st.subheader("Image Optimization")
        st.markdown("Analyze and optimize images for better conversion rates.")
        
        # Instructions section in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Instructions")
            with st.expander("📋 Quick Overview", expanded=True):
                st.markdown("""
                The Image Specialist analyzes your website's images and suggests improvements for better conversion rates.
                
                **Key Features:**
                - Analyzes image placement and context
                - Evaluates emotional impact
                - Identifies conversion potential
                - Provides actionable recommendations
                """)
            
            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.markdown("#### Role & Task")
                st.markdown("""
                - **Role**: Image specialist analyzing website images
                - **Task**: Identify and optimize images for conversion rate improvement
                - **Focus**: Image placement, emotional impact, and conversion potential
                """)
                
                st.markdown("#### Constraints")
                st.markdown("""
                - Select single most impactful image
                - Preserve brand identity
                - Focus on conversion goals
                - Provide clear, actionable instructions
                """)
        
        with col2:
            st.markdown("### Customize")
            with st.expander("✏️ Edit Image Specialist Instructions", expanded=False):
                image_specialist_instructions = st.text_area(
                    "Image Specialist Instructions:",
                    value=DEFAULT_IMAGE_SPECIALIST_INSTRUCTIONS,
                    height=200,
                    help="Modify these instructions to customize how the image specialist analyzes and suggests improvements."
                )
            
            with st.expander("✏️ Edit Image Generation Prompt", expanded=False):
                image_generation_prompt = st.text_area(
                    "Image Generation Prompt:",
                    value=DEFAULT_IMAGE_GENERATION_PROMPT,
                    height=400,
                    help="Modify the prompt used to generate the new image. Use {image_specialist_result.final_output.change_description} as a placeholder for the image specialist's change description."
                )
            
            with st.expander("✏️ Edit Image Evaluator Instructions", expanded=False):
                image_evaluator_instructions = st.text_area(
                    "Image Evaluator Instructions:",
                    value=DEFAULT_IMAGE_EVALUATOR_INSTRUCTIONS,
                    height=400,
                    help="Modify these instructions to customize how the image evaluator assesses image changes."
                )
        
        # Run button
        if st.button("Run Image Optimization", type="primary", disabled=not api_key_set):
            if not api_key_set:
                st.error("Please enter your OpenAI API key in the sidebar to use this feature.")
            else:
                with st.spinner("Processing image optimization..."):
                    try:
                        result = asyncio.run(process_image_optimization(url, image_specialist_instructions, image_generation_prompt, image_evaluator_instructions))
                        if result:
                            # Create two columns for the results
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("### Image Specialist Response")
                                st.info(result['image_specialist_response'])
                                
                                st.markdown("### Generated JavaScript Code")
                                st.code(result['js_code'], language="javascript")
                                
                                st.markdown("### Original Image")
                                st.image(result['original_image_url'], use_container_width=True)
                                
                                st.markdown("### New Image")
                                st.image(result['new_image_url'], use_container_width=True)
                            
                            with col2:
                                st.markdown("### Screenshot After Changes")
                                st.image(f"data:image/jpeg;base64,{result['screenshot']}", use_container_width=True)
                        else:
                            st.error("Failed to process image optimization. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.subheader("Text Optimization")
        st.markdown("Optimize text content while maintaining visual layout and brand consistency.")
        
        # Instructions section in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Instructions")
            with st.expander("📋 Quick Overview", expanded=True):
                st.markdown("""
                The Text Optimization process uses two specialized agents:
                
                **1. Coder (Text Generator)**
                - Generates high-impact text changes
                - Focuses on headers and CTAs
                - Preserves brand tone and meaning
                
                **2. Evaluator (Visual Checker)**
                - Ensures visual consistency
                - Checks layout and spacing
                - Prevents visual defects
                """)
            
            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.markdown("#### Coder Settings")
                st.markdown("""
                - **Role**: Conversion copywriter and CRO strategist
                - **Task**: Generate high-impact text changes
                - **Focus**: Headers, CTAs, and key messaging
                - **Constraints**: Preserve tone and meaning
                """)
                
                st.markdown("#### Evaluator Settings")
                st.markdown("""
                - **Role**: UI/UX validator
                - **Task**: Ensure visual consistency
                - **Focus**: Layout, spacing, and alignment
                - **Constraints**: No visual defects allowed
                """)
        
        with col2:
            st.markdown("### Customize")
            with st.expander("✏️ Edit Coder Instructions", expanded=False):
                coder_instructions = st.text_area(
                    "Coder Instructions:",
                    value=DEFAULT_CODER_INSTRUCTIONS,
                    height=200,
                    help="Modify these instructions to customize how the coder generates text improvements."
                )
            
            with st.expander("✏️ Edit Evaluator Instructions", expanded=False):
                evaluator_instructions = st.text_area(
                    "Evaluator Instructions:",
                    value=DEFAULT_EVALUATOR_INSTRUCTIONS,
                    height=200,
                    help="Modify these instructions to customize how the evaluator assesses text changes."
                )
        
        # Run button
        if st.button("Run Text Optimization", type="primary", disabled=not api_key_set):
            if not api_key_set:
                st.error("Please enter your OpenAI API key in the sidebar to use this feature.")
            else:
                with st.spinner("Processing text optimization..."):
                    try:
                        result = asyncio.run(process_text_optimization(url, coder_instructions, evaluator_instructions))
                        
                        # Create two columns for the results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # JavaScript code section first
                            st.markdown("### Generated JavaScript Code")
                            st.code(result['script'], language="javascript")
                            
                            st.markdown("### Coder Response")
                            st.info(result['coder_response'])
                            
                            st.markdown("### Evaluator Response")
                            st.info(result['evaluator_response'])
                        
                        with col2:
                            st.markdown("### Screenshot After Changes")
                            st.image(f"data:image/jpeg;base64,{result['screenshot']}", use_container_width=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
