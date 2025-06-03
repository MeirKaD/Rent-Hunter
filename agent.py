import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from mcp_use import MCPAgent, MCPClient
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for structured listing data
class ListingImage(BaseModel):
    url: str
    alt_text: Optional[str] = None


class ListingContact(BaseModel):
    phone: Optional[str] = None
    contact_name: Optional[str] = None
    whatsapp_available: bool = False


class PropertyDetails(BaseModel):
    rooms: Optional[float] = None
    square_meters: Optional[int] = None
    floor: Optional[str] = None
    total_floors: Optional[int] = None
    parking: bool = False
    elevator: bool = False
    balcony: bool = False
    air_conditioning: bool = False
    furnished: bool = False


class ListingLocation(BaseModel):
    neighborhood: Optional[str] = None
    street: Optional[str] = None
    city: str = "Tel Aviv"
    full_address: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None


class RentalListing(BaseModel):
    listing_id: str
    title: str
    price: int  # Monthly rent in NIS
    currency: str = "NIS"
    description: str
    location: ListingLocation
    property_details: PropertyDetails
    contact: ListingContact
    images: List[ListingImage] = Field(default_factory=list)
    source_platform: str
    source_url: str
    posted_date: Optional[str] = None
    discovered_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    raw_data: Optional[Dict] = None


class DiscoveryState(TypedDict):
    search_query: str
    raw_search_results: str
    listing_urls: List[str]
    raw_listing_pages: List[str]
    structured_listings: List[RentalListing]
    discovery_summary: str


class DiscoveryAgent:
    def __init__(self):
        # Initialize LLMs
        self.extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Bright Data MCP configuration
        self.bright_data_config = {
            "mcpServers": {
                "Bright Data": {
                    "command": "npx",
                    "args": ["@brightdata/mcp"],
                    "env": {
                        "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                    }
                }
            }
        }

    def _extract_listing_id_from_url(self, url: str) -> str:
        """Extract listing ID from Yad2 URL"""
        # Yad2 URLs typically contain the listing ID
        match = re.search(r'/item/(\d+)', url)
        if match:
            return match.group(1)
        # Fallback: use URL hash
        return str(hash(url))

    async def search_yad2_node(self, state: DiscoveryState):
        """Search for new rental listings on Yad2"""
        dispatch_custom_event("search_status", "Searching Yad2 for new listings...")
        
        # Create MCP client
        client = MCPClient.from_dict(self.bright_data_config)
        agent = MCPAgent(llm=self.fast_llm, client=client, max_steps=20)
        
        search_query = state["search_query"]
        
        # Search Yad2 for rental listings
        search_prompt = f"""
        Search for rental apartments in Tel Aviv on Yad2 website.
        
        Search Query: {search_query}
        
        Use the search_engine tool to find recent rental listings on yad2.co.il
        Focus on apartments for rent in Tel Aviv area.
        
        Look for listings with the following criteria:
        - Property type: Apartment/דירה
        - City: Tel Aviv/תל אביב
        - Listing type: For rent/להשכרה
        
        Return the search results with URLs to individual listings.
        """
        
        try:
            search_results = await agent.run(search_prompt)
            return {"raw_search_results": search_results}
        except Exception as e:
            logger.error(f"Error in Yad2 search: {e}")
            return {"raw_search_results": f"Error occurred: {str(e)}"}

    async def extract_listing_urls_node(self, state: DiscoveryState):
        """Extract Yad2 URLs from search results for direct scraping"""
        dispatch_custom_event("url_extraction_status", "Extracting Yad2 URLs...")
        
        search_results = state["raw_search_results"]
        print(search_results)
        
        # Define structured output model
        from pydantic import BaseModel
        from typing import List
        
        class URLExtraction(BaseModel):
            yad2_urls: List[str] = Field(description="List of Yad2 URLs to scrape for apartment listings")
        
        # Use structured output for URL extraction
        url_extractor = self.fast_llm.with_structured_output(URLExtraction)
        
        url_extraction_prompt = f"""
        Extract all Yad2 URLs from the following search results.
        
        Search Results:
        {search_results}
        
        Find any URLs that contain "yad2.co.il" - these could be individual listings, search pages, or category pages.
        We will scrape these directly to find apartment rental information.
        
        Return all valid Yad2 URLs found.
        """
        
        try:
            extracted_data = url_extractor.invoke(url_extraction_prompt)
            print(f"DEBUG: LLM extracted_data: {extracted_data}") 
            urls = extracted_data.yad2_urls
            print(f"DEBUG: Extracted URLs: {urls}")
            
            unique_urls = list(set(urls))
            print(f"DEBUG: Unique URLs: {unique_urls}") 
            dispatch_custom_event("urls_found", f"Found {len(unique_urls)} Yad2 URLs to scrape")
            return {"listing_urls": unique_urls}
            
        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")
            # Fallback regex extraction

    async def scrape_listings_node(self, state: DiscoveryState):
        """Scrape individual listing pages for detailed information"""
        dispatch_custom_event("scraping_status", "Scraping individual listings...")
        
        listing_urls = state.get("listing_urls", [])
        print(f"DEBUG: listing_urls received in scrape_listings_node: {listing_urls}")
        if not listing_urls:
            return {"raw_listing_pages": []}
        
        # Create MCP client
        client = MCPClient.from_dict(self.bright_data_config)
        agent = MCPAgent(llm=self.fast_llm, client=client, max_steps=30)
        
        raw_pages = []
        
        for i, url in enumerate(listing_urls):  # Limit to 5 for initial testing
            dispatch_custom_event("scraping_progress", f"Scraping listing len(listing_urls))")
            
            scrape_prompt = f"""Use scrape_as_markdown tool for: {url}

            OUTPUT ONLY THE RAW SCRAPED CONTENT. No explanations, no summaries, no modifications."""
            
            try:
                page_content = await agent.run(scrape_prompt)
                raw_pages.append({
                    "url": url,
                    "content": page_content,
                    "scraped_at": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                raw_pages.append({
                    "url": url,
                    "content": f"Error: {str(e)}",
                    "scraped_at": datetime.now().isoformat()
                })
        
        return {"raw_listing_pages": raw_pages}

    def structure_listings_node(self, state: DiscoveryState):
        """Convert raw scraped data into structured listing objects"""
        dispatch_custom_event("structuring_status", "Converting to structured data...")
        
        raw_pages = state["raw_listing_pages"]
        print(raw_pages)
        structured_listings = []
        
        from pydantic import BaseModel
        from typing import List
    
        class LocationData(BaseModel):
            neighborhood: Optional[str] = None
            street: Optional[str] = None
            city: str = "Tel Aviv"
            full_address: Optional[str] = None

        class PropertyDetailsData(BaseModel):
            rooms: Optional[float] = None
            square_meters: Optional[int] = None
            floor: Optional[str] = None
            total_floors: Optional[int] = None
            parking: bool = False
            elevator: bool = False
            balcony: bool = False
            air_conditioning: bool = False
            furnished: bool = False

        class ContactData(BaseModel):
            phone: Optional[str] = None
            contact_name: Optional[str] = None
            whatsapp_available: bool = False

        class ImageData(BaseModel):
            url: str
            alt_text: Optional[str] = None

        class ListingData(BaseModel):
            title: str
            price: int
            description: str
            location: LocationData
            property_details: PropertyDetailsData
            contact: ContactData
            images: List[ImageData] = Field(default_factory=list)
            posted_date: Optional[str] = None

        class MultipleListingsExtraction(BaseModel):
            listings: List[ListingData] = Field(description="List of all apartment listings found on this page")

        for page_data in raw_pages:
            url = page_data["url"]
            content = page_data["content"]
            
            # Create structured extraction prompt
            extraction_prompt = f"""
            Extract structured rental listing information from the following scraped Yad2 page content.
            
            URL: {url}
            Content: {content}
            
            Extract the following information and format as JSON:
            
            {{
                "title": "Property title",
                "price": 0,  // Monthly rent in NIS (numbers only)
                "description": "Full property description",
                "location": {{
                    "neighborhood": "Neighborhood name",
                    "street": "Street name",
                    "city": "Tel Aviv",
                    "full_address": "Complete address"
                }},
                "property_details": {{
                    "rooms": 0.0,  // Number of rooms (e.g., 3.5)
                    "square_meters": 0,  // Size in sqm
                    "floor": "Floor number",
                    "total_floors": 0,
                    "parking": false,
                    "elevator": false,
                    "balcony": false,
                    "air_conditioning": false,
                    "furnished": false
                }},
                "contact": {{
                    "phone": "Phone number",
                    "contact_name": "Contact person name",
                    "whatsapp_available": false
                }},
                "images": [
                    {{"url": "image_url", "alt_text": "description"}}
                ],
                "posted_date": "Posting date if available"
            }}
            
            Important:
            - Extract numeric values only for price and square_meters
            - Use boolean values for amenities
            - If information is not available, use null or appropriate default values
            - Ensure all required fields are present
            """
            
            try:
                # Use structured output
                listing_extractor = self.extraction_llm.with_structured_output(MultipleListingsExtraction)
                
                extraction_prompt = f"""
                Extract all apartment rental listings from the following scraped Yad2 page content.
                
                URL: {url}
                Content: {content}
                
                This page may contain multiple apartment listings. Extract information for each listing you find.
                
                For each listing, extract:
                - title: Property title or address
                - price: Monthly rent in NIS (numbers only) 
                - description: Property description
                - location: neighborhood, street, city, full_address
                - property_details: rooms, square_meters, floor, total_floors, parking, elevator, balcony, air_conditioning, furnished
                - contact: phone, contact_name, whatsapp_available
                - images: list of image URLs with alt_text
                - posted_date: if available
                
                Return all listings found on this page.
                """
                
                extracted_data = listing_extractor.invoke(extraction_prompt)
                
                # Process each listing found on this page
                for listing_data in extracted_data.listings:
                    listing = RentalListing(
                        listing_id=f"{self._extract_listing_id_from_url(url)}_{len(structured_listings)}",
                        title=listing_data.title,
                        price=listing_data.price,
                        description=listing_data.description,
                        location=ListingLocation(
                            neighborhood=listing_data.location.neighborhood,
                            street=listing_data.location.street,
                            city=listing_data.location.city,
                            full_address=listing_data.location.full_address
                        ),
                        property_details=PropertyDetails(
                            rooms=listing_data.property_details.rooms,
                            square_meters=listing_data.property_details.square_meters,
                            floor=listing_data.property_details.floor,
                            total_floors=listing_data.property_details.total_floors,
                            parking=listing_data.property_details.parking,
                            elevator=listing_data.property_details.elevator,
                            balcony=listing_data.property_details.balcony,
                            air_conditioning=listing_data.property_details.air_conditioning,
                            furnished=listing_data.property_details.furnished
                        ),
                        contact=ListingContact(
                            phone=listing_data.contact.phone,
                            contact_name=listing_data.contact.contact_name,
                            whatsapp_available=listing_data.contact.whatsapp_available
                        ),
                        images=[ListingImage(url=img.url, alt_text=img.alt_text) for img in listing_data.images],
                        source_platform="Yad2",
                        source_url=url,
                        posted_date=listing_data.posted_date,
                        raw_data=listing_data.model_dump()
                    )
                    
                    structured_listings.append(listing)
                    dispatch_custom_event("listing_structured", f"Processed: {listing.title[:50]}...")
                
            except Exception as e:
                logger.error(f"Error structuring listing from {url}: {e}")
                # Create minimal listing object for failed extractions
                fallback_listing = RentalListing(
                    listing_id=self._extract_listing_id_from_url(url),
                    title="Failed to extract title",
                    price=0,
                    description=f"Extraction failed: {str(e)}",
                    location=ListingLocation(),
                    property_details=PropertyDetails(),
                    contact=ListingContact(),
                    source_platform="Yad2",
                    source_url=url,
                    raw_data={"error": str(e), "raw_content": content[:500]}
                )
                structured_listings.append(fallback_listing)
        
        return {"structured_listings": structured_listings}

    def summary_node(self, state: DiscoveryState):
        """Generate discovery summary"""
        listings = state["structured_listings"]
        
        total_listings = len(listings)
        successful_extractions = len([l for l in listings if l.price > 0])
        avg_price = sum(l.price for l in listings if l.price > 0) / max(successful_extractions, 1)
        
        neighborhoods = list(set([
            l.location.neighborhood for l in listings 
            if l.location and l.location.neighborhood
        ]))
        
        summary = f"""
        ## Discovery Summary
        
        **Total Listings Found:** {total_listings}
        **Successfully Processed:** {successful_extractions}
        **Average Price:** ₪{avg_price:,.0f}/month
        **Neighborhoods:** {', '.join(neighborhoods[:5])}
        
        **Sample Listings:**
        """
        
        for listing in listings[:3]:
            if listing.price > 0:
                summary += f"\n- **{listing.title}** - ₪{listing.price:,}/month ({listing.location.neighborhood or 'Unknown area'})"
        
        dispatch_custom_event("discovery_complete", f"Found {total_listings} listings")
        
        return {"discovery_summary": summary}

    def build_graph(self):
        """Build and compile the discovery graph"""
        graph_builder = StateGraph(DiscoveryState)
        
        # Add nodes
        graph_builder.add_node("Search Yad2", self.search_yad2_node)
        graph_builder.add_node("Extract URLs", self.extract_listing_urls_node)
        graph_builder.add_node("Scrape Listings", self.scrape_listings_node)
        graph_builder.add_node("Structure Data", self.structure_listings_node)
        graph_builder.add_node("Generate Summary", self.summary_node)
        
        # Define edges
        graph_builder.add_edge(START, "Search Yad2")
        graph_builder.add_edge("Search Yad2", "Extract URLs")
        graph_builder.add_edge("Extract URLs", "Scrape Listings")
        graph_builder.add_edge("Scrape Listings", "Structure Data")
        graph_builder.add_edge("Structure Data", "Generate Summary")
        graph_builder.add_edge("Generate Summary", END)
        
        return graph_builder.compile()


# Example usage function
async def run_discovery(search_query: str = "דירות להשכרה תל אביב"):
    """Run the discovery agent with a search query"""
    
    agent = DiscoveryAgent()
    graph = agent.build_graph()
    
    initial_state = {
        "search_query": search_query,
        "raw_search_results": "",
        "listing_urls": [],
        "raw_listing_pages": [],
        "structured_listings": [],
        "discovery_summary": ""
    }
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    import asyncio
    
    # Test the discovery agent
    async def test_discovery():
        result = await run_discovery("דירות להשכרה תל אביב")
        print("Discovery Results:")
        print(result["discovery_summary"])
        print(f"\nTotal listings: {len(result['structured_listings'])}")
        
        # Print first listing details
        if result["structured_listings"]:
            first_listing = result["structured_listings"][0]
            print(f"\nFirst listing: {first_listing.title}")
            print(f"Price: ₪{first_listing.price:,}")
            print(f"Location: {first_listing.location.neighborhood}")
            print(f"Rooms: {first_listing.property_details.rooms}")
    
    # Run test
    asyncio.run(test_discovery())