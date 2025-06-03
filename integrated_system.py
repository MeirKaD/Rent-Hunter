import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

# Import your existing agents
from agent import DiscoveryAgent, DiscoveryState, RentalListing
from price_analysis_agent import PriceAnalysisAgent, PriceAnalysisState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Combined state for the multi-agent system
class RentRadarState(TypedDict):
    # Input
    search_query: str
    
    # Discovery Agent outputs
    raw_search_results: str
    listing_urls: List[str]
    raw_listing_pages: List[str]
    structured_listings: List[RentalListing]
    discovery_summary: str
    
    # Price Analysis Agent outputs
    neighborhood_stats: Dict
    price_fairness_scores: List
    enriched_listings: List[RentalListing]
    price_analysis_summary: str
    
    # Final system outputs
    final_summary: str
    recommended_listings: List[RentalListing]


class RentRadarMultiAgent:
    """
    Integrated multi-agent system that combines Discovery and Price Analysis agents
    """
    
    def __init__(self):
        self.discovery_agent = DiscoveryAgent()
        self.price_analysis_agent = PriceAnalysisAgent()
        self.coordinator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def discovery_coordinator_node(self, state: RentRadarState):
        """Coordinate the discovery process"""
        dispatch_custom_event("system_status", "🔍 Starting property discovery...")
        
        search_query = state["search_query"]
        
        # Create discovery state
        discovery_state = {
            "search_query": search_query,
            "raw_search_results": "",
            "listing_urls": [],
            "raw_listing_pages": [],
            "structured_listings": [],
            "discovery_summary": ""
        }
        
        # Run discovery agent
        discovery_graph = self.discovery_agent.build_graph()
        discovery_result = await discovery_graph.ainvoke(discovery_state)
        
        # Extract results for the main state
        return {
            "raw_search_results": discovery_result["raw_search_results"],
            "listing_urls": discovery_result["listing_urls"],
            "raw_listing_pages": discovery_result["raw_listing_pages"],
            "structured_listings": discovery_result["structured_listings"],
            "discovery_summary": discovery_result["discovery_summary"]
        }

    async def price_analysis_coordinator_node(self, state: RentRadarState):
        """Coordinate the price analysis process"""
        dispatch_custom_event("system_status", "📊 Starting price analysis...")
        
        structured_listings = state["structured_listings"]
        
        if not structured_listings:
            return {
                "neighborhood_stats": {},
                "price_fairness_scores": [],
                "enriched_listings": [],
                "price_analysis_summary": "No listings available for price analysis."
            }
        
        # Create price analysis state
        price_analysis_state = {
            "input_listings": structured_listings,
            "neighborhood_stats": {},
            "price_fairness_scores": [],
            "market_trends": [],
            "analysis_summary": "",
            "enriched_listings": []
        }
        
        # Run price analysis agent
        price_analysis_graph = self.price_analysis_agent.build_graph()
        price_analysis_result = await price_analysis_graph.ainvoke(price_analysis_state)
        
        # Extract results for the main state
        return {
            "neighborhood_stats": price_analysis_result["neighborhood_stats"],
            "price_fairness_scores": price_analysis_result["price_fairness_scores"],
            "enriched_listings": price_analysis_result["enriched_listings"],
            "price_analysis_summary": price_analysis_result["analysis_summary"]
        }

    def recommendation_engine_node(self, state: RentRadarState):
        """Generate final recommendations based on discovery and price analysis"""
        dispatch_custom_event("system_status", "🎯 Generating recommendations...")
        
        enriched_listings = state["enriched_listings"]
        price_fairness_scores = state["price_fairness_scores"]
        
        if not enriched_listings:
            return {
                "recommended_listings": [],
                "final_summary": "No recommendations available - no listings were successfully analyzed."
            }
        
        # Create fairness score lookup
        fairness_lookup = {score.listing_id: score for score in price_fairness_scores}
        
        # Score and rank listings
        scored_listings = []
        for listing in enriched_listings:
            fairness_score = fairness_lookup.get(listing.listing_id)
            
            # Calculate recommendation score (0-100)
            recommendation_score = 50  # Base score
            
            if fairness_score:
                # Price fairness component (40 points max)
                if fairness_score.fairness_category == "Excellent Deal":
                    recommendation_score += 40
                elif fairness_score.fairness_category == "Good Deal":
                    recommendation_score += 30
                elif fairness_score.fairness_category == "Fair Price":
                    recommendation_score += 10
                # Overpriced gets 0 additional points
                
                # Confidence component (10 points max)
                if fairness_score.confidence_level == "High":
                    recommendation_score += 10
                elif fairness_score.confidence_level == "Medium":
                    recommendation_score += 5
            
            # Property features component (bonus points)
            if listing.property_details.elevator:
                recommendation_score += 5
            if listing.property_details.parking:
                recommendation_score += 5
            if listing.property_details.balcony:
                recommendation_score += 3
            if listing.property_details.air_conditioning:
                recommendation_score += 2
            
            scored_listings.append((listing, recommendation_score, fairness_score))
        
        # Sort by recommendation score
        scored_listings.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommended_listings = [item[0] for item in scored_listings[:10]]
        
        # Generate final summary
        total_listings = len(enriched_listings)
        excellent_deals = len([s for s in price_fairness_scores if s.fairness_category == "Excellent Deal"])
        good_deals = len([s for s in price_fairness_scores if s.fairness_category == "Good Deal"])
        
        top_score = scored_listings[0][1] if scored_listings else 0
        
        final_summary = f"""
        ## 🏆 Rent Radar TLV - Final Recommendations
        
        **Search Query:** "{state['search_query']}"
        **Analysis Date:** {datetime.now().strftime('%B %d, %Y at %H:%M')}
        
        ### 📈 Results Overview
        - **Total Properties Found:** {total_listings}
        - **Price Analysis Completed:** {len(price_fairness_scores)} listings
        - **Excellent/Good Deals:** {excellent_deals + good_deals} properties
        - **Top Recommendation Score:** {top_score:.0f}/100
        
        ### 🎯 Top Recommendations
        """
        
        for i, (listing, score, fairness_score) in enumerate(scored_listings[:5], 1):
            price_info = ""
            if fairness_score:
                price_info = f" ({fairness_score.fairness_category}, {fairness_score.percentage_vs_median:+.1f}% vs median)"
            
            final_summary += f"""
        **{i}. {listing.title[:60]}**
        - Price: ₪{listing.price:,}/month{price_info}
        - Location: {listing.location.neighborhood or 'Unknown'}, {listing.location.city}
        - Size: {listing.property_details.rooms or '?'} rooms, {listing.property_details.square_meters or '?'} sqm
        - Score: {score:.0f}/100
        - Contact: {listing.contact.phone or 'Not available'}
        """
        
        final_summary += f"""
        
        ### 📞 Next Steps
        1. **Contact immediately** for excellent deals (market moves fast!)
        2. **Schedule viewings** for top-scored properties
        3. **Prepare documentation** for quick application process
        
        *Generated by Rent Radar TLV Multi-Agent System*
        """
        
        dispatch_custom_event("recommendations_ready", f"Generated {len(recommended_listings)} recommendations")
        
        return {
            "recommended_listings": recommended_listings,
            "final_summary": final_summary
        }

    def build_graph(self):
        """Build and compile the multi-agent system graph"""
        graph_builder = StateGraph(RentRadarState)
        
        # Add nodes
        graph_builder.add_node("Discovery", self.discovery_coordinator_node)
        graph_builder.add_node("Price Analysis", self.price_analysis_coordinator_node)
        graph_builder.add_node("Generate Recommendations", self.recommendation_engine_node)
        
        # Define edges - sequential flow
        graph_builder.add_edge(START, "Discovery")
        graph_builder.add_edge("Discovery", "Price Analysis")
        graph_builder.add_edge("Price Analysis", "Generate Recommendations")
        graph_builder.add_edge("Generate Recommendations", END)
        
        return graph_builder.compile()


# Main execution function
async def run_rent_radar(search_query: str = "דירות להשכרה תל אביב"):
    """
    Run the complete Rent Radar TLV system
    """
    
    system = RentRadarMultiAgent()
    graph = system.build_graph()
    
    initial_state = {
        "search_query": search_query,
        # Discovery outputs
        "raw_search_results": "",
        "listing_urls": [],
        "raw_listing_pages": [],
        "structured_listings": [],
        "discovery_summary": "",
        # Price analysis outputs
        "neighborhood_stats": {},
        "price_fairness_scores": [],
        "enriched_listings": [],
        "price_analysis_summary": "",
        # Final outputs
        "final_summary": "",
        "recommended_listings": []
    }
    
    # Run the complete pipeline
    print("🚀 Starting Rent Radar TLV Multi-Agent System...")
    final_state = await graph.ainvoke(initial_state)
    
    return final_state


# Event handler for real-time updates
class RentRadarEventHandler:
    """Handle real-time events from the multi-agent system"""
    
    def __init__(self):
        self.events = []
    
    def handle_event(self, event_type: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        event = f"[{timestamp}] {event_type}: {message}"
        self.events.append(event)
        print(event)


# Utility functions for integration
def export_results_to_json(final_state: RentRadarState, filename: str = None):
    """Export results to JSON file for further processing"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rent_radar_results_{timestamp}.json"
    
    # Convert pydantic models to dict for JSON serialization
    export_data = {
        "search_query": final_state["search_query"],
        "discovery_summary": final_state["discovery_summary"],
        "price_analysis_summary": final_state["price_analysis_summary"],
        "final_summary": final_state["final_summary"],
        "total_listings_found": len(final_state["structured_listings"]),
        "total_recommendations": len(final_state["recommended_listings"]),
        "listings": [listing.model_dump() for listing in final_state["enriched_listings"]],
        "recommendations": [listing.model_dump() for listing in final_state["recommended_listings"]],
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"📁 Results exported to {filename}")
    return filename


def filter_recommendations(final_state: RentRadarState, 
                         max_price: int = None,
                         min_rooms: float = None,
                         neighborhood: str = None,
                         deal_types: List[str] = None) -> List[RentalListing]:
    """Filter recommendations based on user criteria"""
    
    recommendations = final_state["recommended_listings"]
    filtered = []
    
    for listing in recommendations:
        # Price filter
        if max_price and listing.price > max_price:
            continue
        
        # Rooms filter
        if min_rooms and (not listing.property_details.rooms or listing.property_details.rooms < min_rooms):
            continue
        
        # Neighborhood filter
        if neighborhood and neighborhood.lower() not in (listing.location.neighborhood or "").lower():
            continue
        
        # Deal type filter
        if deal_types and listing.raw_data and "price_analysis" in listing.raw_data:
            fairness_category = listing.raw_data["price_analysis"]["fairness_category"]
            if fairness_category not in deal_types:
                continue
        
        filtered.append(listing)
    
    return filtered


if __name__ == "__main__":
    # Example usage and testing
    
    async def test_complete_system():
        """Test the complete integrated system"""
        
        print("="*60)
        print("🏢 RENT RADAR TLV - MULTI-AGENT SYSTEM TEST")
        print("="*60)
        
        # Test 1: Basic system run
        print("\n1️⃣ Testing complete system pipeline...")
        result = await run_rent_radar("דירות להשכרה תל אביב 3 חדרים")
        
        print("\n📊 FINAL RESULTS:")
        print(result["final_summary"])
        
        # Test 2: Export results
        print("\n2️⃣ Exporting results...")
        export_file = export_results_to_json(result)
        
        # Test 3: Filter recommendations
        print("\n3️⃣ Testing recommendation filters...")
        
        # Filter for good deals under ₪8000
        good_deals = filter_recommendations(
            result, 
            max_price=8000, 
            deal_types=["Excellent Deal", "Good Deal"]
        )
        print(f"Good deals under ₪8,000: {len(good_deals)} properties")
        
        # Filter for 3+ rooms in specific neighborhoods
        spacious_apartments = filter_recommendations(
            result, 
            min_rooms=3.0,
            neighborhood="center"
        )
        print(f"3+ rooms in center: {len(spacious_apartments)} properties")
        
        # Test 4: Performance metrics
        print("\n📈 SYSTEM PERFORMANCE:")
        total_listings = len(result["structured_listings"])
        analyzed_listings = len(result["enriched_listings"])
        recommendations = len(result["recommended_listings"])
        
        print(f"Discovery Success Rate: {(total_listings/max(1, len(result['listing_urls'])))*100:.1f}%")
        print(f"Analysis Success Rate: {(analyzed_listings/max(1, total_listings))*100:.1f}%")
        print(f"Recommendation Ratio: {(recommendations/max(1, analyzed_listings))*100:.1f}%")
        
        return result
    
    
    async def test_different_queries():
        """Test system with different search queries"""
        
        test_queries = [
            "דירות להשכרה תל אביב 2 חדרים",
            "apartment rental tel aviv center",
            "דירה להשכרה נווה צדק",
            "דירות להשכרה רמת אביב"
        ]
        
        print("\n4️⃣ Testing different search queries...")
        
        results = {}
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            try:
                result = await run_rent_radar(query)
                results[query] = {
                    "total_found": len(result["structured_listings"]),
                    "recommendations": len(result["recommended_listings"]),
                    "success": True
                }
                print(f"✅ Found {results[query]['total_found']} listings, {results[query]['recommendations']} recommendations")
            except Exception as e:
                results[query] = {"error": str(e), "success": False}
                print(f"❌ Error: {e}")
        
        return results
    
    
    # Run tests
    async def main():
        # Run complete system test
        main_result = await test_complete_system()
        
        # Test with different queries
        # query_results = await test_different_queries()
        
        print("\n" + "="*60)
        print("🎉 MULTI-AGENT SYSTEM TESTING COMPLETE")
        print("="*60)
        
        return main_result
    
    
    # Execute the test
    asyncio.run(main())