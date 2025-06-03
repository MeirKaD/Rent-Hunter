import json
import logging
import os
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from mcp_use import MCPAgent, MCPClient
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent import RentalListing, ListingLocation, PropertyDetails  # Import from your existing agent

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Price Analysis Models
class NeighborhoodPriceStats(BaseModel):
    neighborhood: str
    median_price_per_sqm: float
    average_price_per_sqm: float
    price_range_min: float
    price_range_max: float
    total_listings: int
    sample_properties: List[Dict] = Field(default_factory=list)


class PriceFairnessScore(BaseModel):
    listing_id: str
    price_per_sqm: float
    neighborhood_median_per_sqm: float
    fairness_score: float  # 0-1 scale
    fairness_category: str  # "Good Deal", "Fair", "Overpriced"
    percentage_vs_median: float
    confidence_level: str  # "High", "Medium", "Low"


class MarketTrend(BaseModel):
    neighborhood: str
    trend_direction: str  # "Rising", "Stable", "Declining"
    trend_strength: float  # -1 to 1
    recent_price_change: float  # Percentage change
    sample_size: int


class PriceAnalysisState(TypedDict):
    input_listings: List[RentalListing]
    neighborhood_stats: Dict[str, NeighborhoodPriceStats]
    price_fairness_scores: List[PriceFairnessScore]
    market_trends: List[MarketTrend]
    analysis_summary: str
    enriched_listings: List[RentalListing]  # Listings with price analysis added


class PriceAnalysisAgent:
    def __init__(self):
        # Initialize LLMs
        self.analysis_llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
        
        # Tel Aviv neighborhood mapping
        self.tel_aviv_neighborhoods = {
            "city center": ["center", "merkaz", "downtown", "dizengoff"],
            "neve tzedek": ["neve tzedek", "neveh tzedek"],
            "florentin": ["florentin"],
            "rothschild": ["rothschild", "sheinkin"],
            "old north": ["old north", "tzafon yashan"],
            "ramat aviv": ["ramat aviv"],
            "jaffa": ["jaffa", "yafo"],
            "bavli": ["bavli"],
            "shapira": ["shapira"],
            "nachalat binyamin": ["nachalat binyamin"],
        }

    def normalize_neighborhood(self, neighborhood: str) -> str:
        """Normalize neighborhood names for consistent analysis"""
        if not neighborhood:
            return "unknown"
        
        neighborhood_lower = neighborhood.lower().strip()
        
        for normalized_name, variations in self.tel_aviv_neighborhoods.items():
            if any(var in neighborhood_lower for var in variations):
                return normalized_name
        
        return neighborhood_lower

    async def gather_market_data_node(self, state: PriceAnalysisState):
        """Gather additional market data for price comparisons"""
        dispatch_custom_event("market_data_status", "Gathering market comparison data...")
        
        input_listings = state["input_listings"]
        neighborhoods = list(set([
            self.normalize_neighborhood(listing.location.neighborhood) 
            for listing in input_listings 
            if listing.location and listing.location.neighborhood
        ]))
        
        # Create MCP client
        client = MCPClient.from_dict(self.bright_data_config)
        agent = MCPAgent(llm=self.fast_llm, client=client, max_steps=20)
        
        market_data = {}
        
        for neighborhood in neighborhoods[:3]:  # Limit to top 3 neighborhoods for efficiency
            dispatch_custom_event("market_search", f"Searching market data for {neighborhood}")
            
            search_prompt = f"""
            Search for recent rental price data and market information for {neighborhood} neighborhood in Tel Aviv.
            
            Use the search_engine tool to find:
            1. Recent rental listings in {neighborhood}, Tel Aviv
            2. Average rental prices per square meter
            3. Market reports or price analysis for Tel Aviv real estate
            4. Property market trends in {neighborhood}
            
            Focus on finding comparable rental properties (apartments) with price per square meter data.
            Look for reliable real estate websites, market reports, or property databases.
            """
            
            try:
                search_results = await agent.run(search_prompt)
                market_data[neighborhood] = search_results
            except Exception as e:
                logger.error(f"Error gathering market data for {neighborhood}: {e}")
                market_data[neighborhood] = f"Error: {str(e)}"
        
        return {"market_data": market_data}

    def calculate_neighborhood_stats_node(self, state: PriceAnalysisState):
        """Calculate price statistics for each neighborhood from available listings"""
        dispatch_custom_event("stats_calculation", "Calculating neighborhood statistics...")
        
        input_listings = state["input_listings"]
        neighborhood_stats = {}
        
        # Group listings by neighborhood
        neighborhood_groups = {}
        for listing in input_listings:
            if listing.price > 0 and listing.property_details.square_meters:
                normalized_neighborhood = self.normalize_neighborhood(listing.location.neighborhood)
                
                if normalized_neighborhood not in neighborhood_groups:
                    neighborhood_groups[normalized_neighborhood] = []
                
                price_per_sqm = listing.price / listing.property_details.square_meters
                neighborhood_groups[normalized_neighborhood].append({
                    "listing": listing,
                    "price_per_sqm": price_per_sqm
                })
        
        # Calculate statistics for each neighborhood
        for neighborhood, listings_data in neighborhood_groups.items():
            prices_per_sqm = [data["price_per_sqm"] for data in listings_data]
            
            if len(prices_per_sqm) >= 2:  # Need at least 2 data points
                stats = NeighborhoodPriceStats(
                    neighborhood=neighborhood,
                    median_price_per_sqm=statistics.median(prices_per_sqm),
                    average_price_per_sqm=statistics.mean(prices_per_sqm),
                    price_range_min=min(prices_per_sqm),
                    price_range_max=max(prices_per_sqm),
                    total_listings=len(listings_data),
                    sample_properties=[
                        {
                            "title": data["listing"].title[:50],
                            "price": data["listing"].price,
                            "sqm": data["listing"].property_details.square_meters,
                            "price_per_sqm": data["price_per_sqm"]
                        }
                        for data in listings_data[:3]  # Keep top 3 as samples
                    ]
                )
                neighborhood_stats[neighborhood] = stats
                
                dispatch_custom_event("neighborhood_analyzed", 
                    f"{neighborhood}: ₪{stats.median_price_per_sqm:.0f}/sqm median")
        
        return {"neighborhood_stats": neighborhood_stats}

    def analyze_price_fairness_node(self, state: PriceAnalysisState):
        """Analyze price fairness for each listing against neighborhood medians"""
        dispatch_custom_event("fairness_analysis", "Analyzing price fairness...")
        
        input_listings = state["input_listings"]
        neighborhood_stats = state["neighborhood_stats"]
        price_fairness_scores = []
        
        for listing in input_listings:
            if listing.price > 0 and listing.property_details.square_meters:
                normalized_neighborhood = self.normalize_neighborhood(listing.location.neighborhood)
                neighborhood_stat = neighborhood_stats.get(normalized_neighborhood)
                
                if neighborhood_stat:
                    listing_price_per_sqm = listing.price / listing.property_details.square_meters
                    neighborhood_median = neighborhood_stat.median_price_per_sqm
                    
                    # Calculate fairness metrics
                    percentage_vs_median = ((listing_price_per_sqm - neighborhood_median) / neighborhood_median) * 100
                    
                    # Determine fairness category based on percentage difference
                    if percentage_vs_median <= -15:
                        fairness_category = "Excellent Deal"
                        fairness_score = 0.9
                    elif percentage_vs_median <= -5:
                        fairness_category = "Good Deal"
                        fairness_score = 0.75
                    elif percentage_vs_median <= 15:
                        fairness_category = "Fair Price"
                        fairness_score = 0.5
                    else:
                        fairness_category = "Overpriced"
                        fairness_score = 0.2
                    
                    # Determine confidence level based on sample size
                    confidence_level = "High" if neighborhood_stat.total_listings >= 5 else \
                                     "Medium" if neighborhood_stat.total_listings >= 3 else "Low"
                    
                    fairness_score_obj = PriceFairnessScore(
                        listing_id=listing.listing_id,
                        price_per_sqm=listing_price_per_sqm,
                        neighborhood_median_per_sqm=neighborhood_median,
                        fairness_score=fairness_score,
                        fairness_category=fairness_category,
                        percentage_vs_median=percentage_vs_median,
                        confidence_level=confidence_level
                    )
                    
                    price_fairness_scores.append(fairness_score_obj)
                    
                    dispatch_custom_event("listing_analyzed", 
                        f"{listing.title[:30]}: {fairness_category} ({percentage_vs_median:+.1f}%)")
        
        return {"price_fairness_scores": price_fairness_scores}

    def enrich_listings_node(self, state: PriceAnalysisState):
        """Add price analysis data to the original listings"""
        dispatch_custom_event("enrichment", "Enriching listings with price analysis...")
        
        input_listings = state["input_listings"]
        price_fairness_scores = state["price_fairness_scores"]
        
        # Create a lookup dictionary for fairness scores
        fairness_lookup = {score.listing_id: score for score in price_fairness_scores}
        
        enriched_listings = []
        
        for listing in input_listings:
            # Create a copy of the listing with additional price analysis
            enriched_listing = listing.model_copy()
            
            # Add price analysis to raw_data if fairness score exists
            fairness_score = fairness_lookup.get(listing.listing_id)
            if fairness_score:
                if not enriched_listing.raw_data:
                    enriched_listing.raw_data = {}
                
                enriched_listing.raw_data["price_analysis"] = {
                    "price_per_sqm": fairness_score.price_per_sqm,
                    "neighborhood_median_per_sqm": fairness_score.neighborhood_median_per_sqm,
                    "fairness_category": fairness_score.fairness_category,
                    "fairness_score": fairness_score.fairness_score,
                    "percentage_vs_median": fairness_score.percentage_vs_median,
                    "confidence_level": fairness_score.confidence_level
                }
                
                # Update the title to include fairness indicator
                fairness_emoji = "🟢" if fairness_score.fairness_category in ["Excellent Deal", "Good Deal"] else \
                               "🟡" if fairness_score.fairness_category == "Fair Price" else "🔴"
                
                enriched_listing.title = f"{fairness_emoji} {enriched_listing.title}"
            
            enriched_listings.append(enriched_listing)
        
        return {"enriched_listings": enriched_listings}

    def generate_analysis_summary_node(self, state: PriceAnalysisState):
        """Generate comprehensive price analysis summary"""
        dispatch_custom_event("summary_generation", "Generating analysis summary...")
        
        neighborhood_stats = state["neighborhood_stats"]
        price_fairness_scores = state["price_fairness_scores"]
        input_listings = state["input_listings"]
        
        # Calculate overall statistics
        total_analyzed = len(price_fairness_scores)
        excellent_deals = len([s for s in price_fairness_scores if s.fairness_category == "Excellent Deal"])
        good_deals = len([s for s in price_fairness_scores if s.fairness_category == "Good Deal"])
        fair_prices = len([s for s in price_fairness_scores if s.fairness_category == "Fair Price"])
        overpriced = len([s for s in price_fairness_scores if s.fairness_category == "Overpriced"])
        
        # Find best deals
        best_deals = sorted(price_fairness_scores, key=lambda x: x.percentage_vs_median)[:3]
        
        # Generate summary
        summary = f"""
        ## 🏢 Price Analysis Summary
        **Analyzed:** {total_analyzed} listings across {len(neighborhood_stats)} neighborhoods
        
        ### 📊 Price Fairness Distribution
        - 🟢 **Excellent/Good Deals:** {excellent_deals + good_deals} listings ({((excellent_deals + good_deals)/max(total_analyzed, 1)*100):.1f}%)
        - 🟡 **Fair Prices:** {fair_prices} listings ({(fair_prices/max(total_analyzed, 1)*100):.1f}%)
        - 🔴 **Overpriced:** {overpriced} listings ({(overpriced/max(total_analyzed, 1)*100):.1f}%)
        
        ### 🏘️ Neighborhood Price Ranges (per sqm)
        """
        
        for neighborhood, stats in neighborhood_stats.items():
            summary += f"\n- **{neighborhood.title()}**: ₪{stats.median_price_per_sqm:.0f} median (₪{stats.price_range_min:.0f}-₪{stats.price_range_max:.0f})"
        
        if best_deals:
            summary += f"\n\n### 💎 Top Opportunities\n"
            for i, deal in enumerate(best_deals, 1):
                listing = next((l for l in input_listings if l.listing_id == deal.listing_id), None)
                if listing:
                    summary += f"{i}. **{listing.title[:50]}** - {deal.percentage_vs_median:+.1f}% vs median\n"
        
        summary += f"\n\n*Analysis completed at {datetime.now().strftime('%H:%M on %B %d, %Y')}*"
        
        dispatch_custom_event("analysis_complete", f"Analyzed {total_analyzed} listings")
        
        return {"analysis_summary": summary}

    def build_graph(self):
        """Build and compile the price analysis graph"""
        graph_builder = StateGraph(PriceAnalysisState)
        
        # Add nodes
        graph_builder.add_node("Gather Market Data", self.gather_market_data_node)
        graph_builder.add_node("Calculate Neighborhood Stats", self.calculate_neighborhood_stats_node)
        graph_builder.add_node("Analyze Price Fairness", self.analyze_price_fairness_node)
        graph_builder.add_node("Enrich Listings", self.enrich_listings_node)
        graph_builder.add_node("Generate Summary", self.generate_analysis_summary_node)
        
        # Define edges
        graph_builder.add_edge(START, "Gather Market Data")
        graph_builder.add_edge("Gather Market Data", "Calculate Neighborhood Stats")
        graph_builder.add_edge("Calculate Neighborhood Stats", "Analyze Price Fairness")
        graph_builder.add_edge("Analyze Price Fairness", "Enrich Listings")
        graph_builder.add_edge("Enrich Listings", "Generate Summary")
        graph_builder.add_edge("Generate Summary", END)
        
        return graph_builder.compile()


# Function to run price analysis on discovered listings
async def run_price_analysis(listings: List[RentalListing]):
    """Run the price analysis agent on a list of rental listings"""
    
    agent = PriceAnalysisAgent()
    graph = agent.build_graph()
    
    initial_state = {
        "input_listings": listings,
        "neighborhood_stats": {},
        "price_fairness_scores": [],
        "market_trends": [],
        "analysis_summary": "",
        "enriched_listings": []
    }
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    import asyncio
    from agent import run_discovery  # Import your existing discovery function
    
    # Test the integrated pipeline
    async def test_integrated_pipeline():
        print("🔍 Running Discovery Agent...")
        discovery_result = await run_discovery("דירות להשכרה תל אביב")
        
        print(f"\n📊 Running Price Analysis on {len(discovery_result['structured_listings'])} listings...")
        price_analysis_result = await run_price_analysis(discovery_result['structured_listings'])
        
        print("\n" + "="*50)
        print("DISCOVERY RESULTS:")
        print(discovery_result["discovery_summary"])
        
        print("\n" + "="*50)
        print("PRICE ANALYSIS RESULTS:")
        print(price_analysis_result["analysis_summary"])
        
        print(f"\n📈 Enriched Listings: {len(price_analysis_result['enriched_listings'])}")
        
        # Show sample enriched listing
        if price_analysis_result['enriched_listings']:
            sample_listing = price_analysis_result['enriched_listings'][0]
            print(f"\nSample Enriched Listing:")
            print(f"Title: {sample_listing.title}")
            print(f"Price: ₪{sample_listing.price:,}")
            if sample_listing.raw_data and "price_analysis" in sample_listing.raw_data:
                analysis = sample_listing.raw_data["price_analysis"]
                print(f"Price/sqm: ₪{analysis['price_per_sqm']:.0f}")
                print(f"Fairness: {analysis['fairness_category']} ({analysis['percentage_vs_median']:+.1f}% vs median)")
    
    # Run integrated test
    asyncio.run(test_integrated_pipeline())