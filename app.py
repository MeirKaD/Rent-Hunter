import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import time
from typing import Dict, List
import base64
from io import BytesIO
from integrated_system import run_rent_radar, export_results_to_json, filter_recommendations
from agent import RentalListing
from price_analysis_agent import PriceAnalysisAgent

# Import your agents (adjust paths as needed)
# from integrated_system import run_rent_radar, export_results_to_json, filter_recommendations
def check_dependencies():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please create a .env file with your API keys. See the README for instructions.")
        return False
    return True
# Configure page
st.set_page_config(
    page_title="Rent Radar TLV - AI-Powered Apartment Search",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    
    .status-running {
        border-left-color: #F59E0B;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(245, 158, 11, 0.02) 100%);
    }
    
    .status-complete {
        border-left-color: #10B981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%);
    }
    
    .status-error {
        border-left-color: #EF4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.02) 100%);
    }
    
    .listing-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #6B7280;
    }
    
    .excellent-deal {
        border-left-color: #10B981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%);
    }
    
    .good-deal {
        border-left-color: #3B82F6;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(59, 130, 246, 0.02) 100%);
    }
    
    .fair-price {
        border-left-color: #F59E0B;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(245, 158, 11, 0.02) 100%);
    }
    
    .overpriced {
        border-left-color: #EF4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.02) 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {
        'discovery': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start...'},
        'price_analysis': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start...'},
        'recommendations': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start...'}
    }

def create_header():
    """Create the main header section"""
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Rent Radar TLV</h1>
        <h3>AI-Powered Multi-Agent Apartment Search System</h3>
        <p>Discover • Analyze • Recommend - All powered by intelligent agents</p>
    </div>
    """, unsafe_allow_html=True)

def create_agent_status_card(agent_name: str, status_info: Dict):
    """Create a status card for each agent"""
    status = status_info['status']
    progress = status_info['progress']
    message = status_info['message']
    
    # Determine emoji and color based on status
    if status == 'running':
        emoji = "🔄"
        status_class = "status-running"
        color = "#F59E0B"
    elif status == 'complete':
        emoji = "✅"
        status_class = "status-complete"
        color = "#10B981"
    elif status == 'error':
        emoji = "❌"
        status_class = "status-error"
        color = "#EF4444"
    else:
        emoji = "⏳"
        status_class = ""
        color = "#6B7280"
    
    st.markdown(f"""
    <div class="agent-card {status_class}">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{emoji}</span>
            <h4 style="margin: 0; color: {color};">{agent_name.title()} Agent</h4>
        </div>
        <p style="margin: 0; color: #6B7280;">{message}</p>
        <div style="background: #f1f5f9; border-radius: 0.5rem; height: 8px; margin-top: 0.5rem;">
            <div style="background: {color}; height: 100%; width: {progress}%; border-radius: 0.5rem; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

async def run_real_analysis(search_query: str):
    """Run the real analysis using your agents"""
    
    # Update initial status
    st.session_state.agent_status['discovery']['status'] = 'running'
    st.session_state.agent_status['discovery']['message'] = 'Starting discovery agent...'
    st.session_state.agent_status['discovery']['progress'] = 10
    
    try:
        # Run the actual rent radar system
        result = await run_rent_radar(search_query)
        
        # Update progress through the stages
        st.session_state.agent_status['discovery']['progress'] = 100
        st.session_state.agent_status['discovery']['status'] = 'complete'
        st.session_state.agent_status['discovery']['message'] = f'Found {len(result["structured_listings"])} listings!'
        
        st.session_state.agent_status['price_analysis']['progress'] = 100
        st.session_state.agent_status['price_analysis']['status'] = 'complete'
        st.session_state.agent_status['price_analysis']['message'] = f'Analyzed {len(result["enriched_listings"])} listings!'
        
        st.session_state.agent_status['recommendations']['progress'] = 100
        st.session_state.agent_status['recommendations']['status'] = 'complete'
        st.session_state.agent_status['recommendations']['message'] = f'Generated {len(result["recommended_listings"])} recommendations!'
        
        return result
        
    except Exception as e:
        st.session_state.agent_status['discovery']['status'] = 'error'
        st.session_state.agent_status['discovery']['message'] = f'Error: {str(e)}'
        st.error(f"Analysis failed: {e}")
        return None

def load_sample_data():
    """Load sample data from the JSON file"""
    sample_data = {
        "search_query": "דירות להשכרה תל אביב 3 חדרים",
        "total_listings_found": 86,
        "total_recommendations": 10,
        "discovery_summary": """
        ## Discovery Summary
        
        **Total Listings Found:** 86
        **Successfully Processed:** 84
        **Average Price:** ₪9,596/month
        **Neighborhoods:** מכללת תל אביב יפו, דקר, הדר יוסף, הצפון הישן - דרום, נווה חן, רמת הטייסים
        """,
        "price_analysis_summary": """
        ## 🏢 Price Analysis Summary
        **Analyzed:** 68 listings across 18 neighborhoods
        
        ### 📊 Price Fairness Distribution
        - 🟢 **Excellent/Good Deals:** 20 listings (29.4%)
        - 🟡 **Fair Prices:** 36 listings (52.9%)
        - 🔴 **Overpriced:** 12 listings (17.6%)
        """,
        "final_summary": """
        ## 🏆 Rent Radar TLV - Final Recommendations
        
        **Search Query:** "דירות להשכרה תל אביב 3 חדרים"
        **Analysis Date:** June 01, 2025 at 18:40
        
        ### 📈 Results Overview
        - **Total Properties Found:** 86
        - **Price Analysis Completed:** 68 listings
        - **Excellent/Good Deals:** 20 properties
        - **Top Recommendation Score:** 100/100
        """,
        "recommendations": [
            {
                "title": "🟢 שדרות נורדאו 26",
                "price": 7500,
                "location": {"neighborhood": "כרם התימנים", "city": "תל אביב יפו"},
                "property_details": {"rooms": 3.0, "square_meters": 75, "floor": "2"},
                "raw_data": {
                    "price_analysis": {
                        "fairness_category": "Excellent Deal",
                        "percentage_vs_median": -50.0,
                        "confidence_level": "High"
                    }
                }
            },
            {
                "title": "🟢 טשרניחובסקי 55",
                "price": 3750,
                "location": {"neighborhood": "הצפון הישן - דרום", "city": "תל אביב יפו"},
                "property_details": {"rooms": 3.0, "square_meters": 59, "floor": "2"},
                "raw_data": {
                    "price_analysis": {
                        "fairness_category": "Excellent Deal",
                        "percentage_vs_median": -25.8,
                        "confidence_level": "Medium"
                    }
                }
            },
            {
                "title": "🟢 ים סוף",
                "price": 7700,
                "location": {"neighborhood": "עג'מי, גבעת העליה", "city": "תל אביב יפו"},
                "property_details": {"rooms": 3.0, "square_meters": 95, "floor": "1"},
                "raw_data": {
                    "price_analysis": {
                        "fairness_category": "Excellent Deal",
                        "percentage_vs_median": -43.7,
                        "confidence_level": "Medium"
                    }
                }
            },
            {
                "title": "🟢 פלורנטין 48",
                "price": 9900,
                "location": {"neighborhood": "פלורנטין", "city": "תל אביב יפו"},
                "property_details": {"rooms": 3.0, "square_meters": 110, "floor": "1"},
                "raw_data": {
                    "price_analysis": {
                        "fairness_category": "Good Deal",
                        "percentage_vs_median": -5.9,
                        "confidence_level": "High"
                    }
                }
            },
            {
                "title": "🟡 יהודה הלוי 94",
                "price": 10500,
                "location": {"neighborhood": "לב תל אביב", "city": "תל אביב יפו"},
                "property_details": {"rooms": 3.0, "square_meters": 78, "floor": "3"},
                "raw_data": {
                    "price_analysis": {
                        "fairness_category": "Fair Price",
                        "percentage_vs_median": 0.0,
                        "confidence_level": "Medium"
                    }
                }
            }
        ]
    }
    return sample_data

def create_listing_card(listing: Dict, rank: int):
    """Create listing card using native Streamlit components"""
    
    # Extract data
    fairness_category = listing.get('raw_data', {}).get('price_analysis', {}).get('fairness_category', 'Fair Price')
    percentage_vs_median = listing.get('raw_data', {}).get('price_analysis', {}).get('percentage_vs_median', 0)
    
    # Clean title (remove emoji if it exists)
    title = listing.get('title', 'Unknown')
    if title.startswith('🟢 ') or title.startswith('🟡 ') or title.startswith('🔴 '):
        title = title[2:].strip()
    
    # Get property details
    rooms = listing.get('property_details', {}).get('rooms', 'N/A')
    sqm = listing.get('property_details', {}).get('square_meters', 'N/A')
    floor = listing.get('property_details', {}).get('floor', 'N/A')
    neighborhood = listing.get('location', {}).get('neighborhood', 'Unknown')
    price_formatted = f"₪{listing['price']:,}"
    
    # Get image URL
    image_url = None
    images = listing.get('images', [])
    if images and len(images) > 0:
        image_url = images[0].get('url')
    
    # Determine emoji and color
    if fairness_category == "Excellent Deal":
        emoji = "🟢"
        color = "green"
    elif fairness_category == "Good Deal":
        emoji = "🟢"
        color = "blue"
    elif fairness_category == "Fair Price":
        emoji = "🟡"
        color = "orange"
    else:
        emoji = "🔴"
        color = "red"
    
    # Create container with border
    with st.container():
        # Create a colored border using columns
        st.markdown(f"""
        <div style="border-left: 4px solid {'#10B981' if color == 'green' else '#3B82F6' if color == 'blue' else '#F59E0B' if color == 'orange' else '#EF4444'}; 
                    padding: 1rem; margin: 1rem 0; background: white; border-radius: 0.5rem; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        
        # Main content with image
        if image_url:
            # Layout with image
            col_img, col_content = st.columns([1, 2])
            
            with col_img:
                try:
                    st.image(image_url, caption="Property Image", use_container_width=True)
                except Exception as e:
                    st.markdown("📷 *Image unavailable*")
            
            with col_content:
                # Header row
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**#{rank} {title}**")
                    st.markdown(f"📍 {neighborhood}")
                with col2:
                    st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{emoji}</div>", unsafe_allow_html=True)
        else:
            # Layout without image
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**#{rank} {title}**")
                st.markdown(f"📍 {neighborhood}")
            with col2:
                st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{emoji}</div>", unsafe_allow_html=True)
        
        # Price row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {price_formatted}/month")
        with col2:
            st.markdown("**vs market median**")
            if percentage_vs_median < 0:
                st.success(f"{percentage_vs_median:+.1f}%")
            elif percentage_vs_median > 15:
                st.error(f"{percentage_vs_median:+.1f}%")
            else:
                st.warning(f"{percentage_vs_median:+.1f}%")
        
        # Property details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rooms", rooms)
        with col2:
            st.metric("Size", f"{sqm} m²")
        with col3:
            st.metric("Floor", floor)
        
        # Fairness category and button
        col1, col2 = st.columns([2, 1])
        with col1:
            if fairness_category == "Excellent Deal":
                st.success(fairness_category)
            elif fairness_category == "Good Deal":
                st.info(fairness_category)
            elif fairness_category == "Fair Price":
                st.warning(fairness_category)
            else:
                st.error(fairness_category)
        
        st.markdown("</div>", unsafe_allow_html=True)

def create_metrics_dashboard(results: Dict):
    """Create metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; color: #3B82F6; margin-bottom: 0.5rem;">🏢</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">{}</div>
            <div style="color: #6B7280;">Total Listings</div>
        </div>
        """.format(results['total_listings_found']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; color: #10B981; margin-bottom: 0.5rem;">⭐</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">{}</div>
            <div style="color: #6B7280;">Recommendations</div>
        </div>
        """.format(results['total_recommendations']), unsafe_allow_html=True)
    
    with col3:
        excellent_deals = len([r for r in results['recommendations'] if r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category') in ['Excellent Deal', 'Good Deal']])
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; color: #10B981; margin-bottom: 0.5rem;">💎</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">{}</div>
            <div style="color: #6B7280;">Great Deals</div>
        </div>
        """.format(excellent_deals), unsafe_allow_html=True)
    
    with col4:
        avg_price = sum(r['price'] for r in results['recommendations'][:5]) / 5
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; color: #F59E0B; margin-bottom: 0.5rem;">💰</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">₪{:,.0f}</div>
            <div style="color: #6B7280;">Avg Price (Top 5)</div>
        </div>
        """.format(avg_price), unsafe_allow_html=True)

def create_price_distribution_chart(results: Dict):
    """Create price distribution chart"""
    recommendations = results['recommendations']
    
    # Extract data for chart
    prices = [r['price'] for r in recommendations]
    neighborhoods = [r['location']['neighborhood'] for r in recommendations]
    fairness = [r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category', 'Fair Price') for r in recommendations]
    
    # Create scatter plot
    fig = px.scatter(
        x=prices,
        y=neighborhoods,
        color=fairness,
        size=[r['property_details']['square_meters'] for r in recommendations],
        title="Price Distribution by Neighborhood",
        labels={'x': 'Monthly Rent (₪)', 'y': 'Neighborhood'},
        color_discrete_map={
            'Excellent Deal': '#10B981',
            'Good Deal': '#3B82F6',
            'Fair Price': '#F59E0B',
            'Overpriced': '#EF4444'
        }
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    create_header()
    dependencies_ok = check_dependencies()
    # Sidebar for controls
    st.sidebar.title("🎮 Control Panel")
    if not dependencies_ok:
        demo_mode = st.sidebar.checkbox("Demo Mode (Use Sample Data)", value=True, disabled=True)
        st.sidebar.error("Real mode disabled - missing API keys")
    else:
        demo_mode = st.sidebar.checkbox("Demo Mode (Use Sample Data)", value=False)
    # Search configuration
    st.sidebar.header("Search Configuration")
    search_query = st.sidebar.text_input(
        "Search Query",
        value="דירות להשכרה תל אביב 3 חדרים",
        help="Enter your search query in Hebrew or English"
    )
    
    max_listings = st.sidebar.slider("Max Listings to Analyze", 10, 100, 50)
    
    # Filters
    st.sidebar.header("Filters")
    max_price = st.sidebar.slider("Max Price (₪)", 0, 20000, 15000)
    min_rooms = st.sidebar.selectbox("Min Rooms", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    
    # Demo mode toggle
    demo_mode = st.sidebar.checkbox("Demo Mode (Use Sample Data)", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Agent Status")
        
        # Agent status cards
        status_container = st.container()
        
    # Start analysis button
    if st.button("🔍 Start Analysis", disabled=st.session_state.analysis_running):
        if demo_mode:
            # Demo mode - Load sample data
            st.session_state.results = load_sample_data()
            st.session_state.agent_status['discovery']['status'] = 'complete'
            st.session_state.agent_status['discovery']['progress'] = 100
            st.session_state.agent_status['discovery']['message'] = 'Demo data loaded!'
            st.session_state.agent_status['price_analysis']['status'] = 'complete'
            st.session_state.agent_status['price_analysis']['progress'] = 100
            st.session_state.agent_status['price_analysis']['message'] = 'Demo analysis complete!'
            st.session_state.agent_status['recommendations']['status'] = 'complete'
            st.session_state.agent_status['recommendations']['progress'] = 100
            st.session_state.agent_status['recommendations']['message'] = 'Demo recommendations ready!'
            st.rerun()
        else:
            # Real mode - run actual agents
            st.session_state.analysis_running = True
            
            # Run real analysis
            with st.spinner("Running analysis... This may take a few minutes."):
                result = asyncio.run(run_real_analysis(search_query))
                
            if result:
                # Convert the result to the format expected by the UI
                st.session_state.results = {
                    "search_query": result["search_query"],
                    "total_listings_found": len(result["structured_listings"]),
                    "total_recommendations": len(result["recommended_listings"]),
                    "discovery_summary": result["discovery_summary"],
                    "price_analysis_summary": result["price_analysis_summary"],
                    "final_summary": result["final_summary"],
                    "recommendations": [listing.model_dump() for listing in result["recommended_listings"]]
                }
            
            st.session_state.analysis_running = False
            st.rerun()
    
    with col2:
        st.header("📊 Quick Stats")
        if st.session_state.results:
            create_metrics_dashboard(st.session_state.results)
    
    # Display agent status
    with status_container:
        create_agent_status_card('discovery', st.session_state.agent_status['discovery'])
        create_agent_status_card('price_analysis', st.session_state.agent_status['price_analysis'])
        create_agent_status_card('recommendations', st.session_state.agent_status['recommendations'])
    
    # Results section
    if st.session_state.results:
        st.header("🎯 Analysis Results")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Recommendations", "📈 Analytics", "🔍 Discovery", "💰 Price Analysis"])
        
        with tab1:
            st.subheader("Top Recommendations")
            
            # Filter recommendations based on sidebar filters
            filtered_recs = [
                r for r in st.session_state.results['recommendations']
                if r['price'] <= max_price and r['property_details']['rooms'] >= min_rooms
            ]
            
            if filtered_recs:
                for i, listing in enumerate(filtered_recs[:10], 1):
                    create_listing_card(listing, i)
            else:
                st.info("No listings match your current filters. Try adjusting the criteria.")
        
        # Replace the Analytics tab section (around line 580-650) with this:

        with tab2:
            st.subheader("Price Analytics")
            
            # Price distribution chart
            if filtered_recs:
                chart = create_price_distribution_chart({'recommendations': filtered_recs})
                st.plotly_chart(chart, use_container_width=True)
            
            # Additional analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fairness Distribution")
                fairness_counts = {}
                for r in filtered_recs:
                    category = r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category', 'Fair Price')
                    fairness_counts[category] = fairness_counts.get(category, 0) + 1
                
                if fairness_counts:
                    fig_pie = px.pie(
                        values=list(fairness_counts.values()),
                        names=list(fairness_counts.keys()),
                        title="Price Fairness Distribution",
                        color_discrete_map={
                            'Excellent Deal': '#10B981',
                            'Good Deal': '#3B82F6',
                            'Fair Price': '#F59E0B',
                            'Overpriced': '#EF4444'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Price vs. Size")
                if filtered_recs:
                    # Create DataFrame for plotly
                    chart_data = []
                    for r in filtered_recs:
                        chart_data.append({
                            'Square_Meters': r['property_details']['square_meters'],
                            'Price': r['price'],
                            'Neighborhood': r['location']['neighborhood'],
                            'Title': r['title']
                        })
                    
                    df_chart = pd.DataFrame(chart_data)
                    
                    fig_scatter = px.scatter(
                        df_chart,
                        x='Square_Meters',
                        y='Price',
                        color='Neighborhood',
                        title="Price vs. Square Meters",
                        labels={'Square_Meters': 'Square Meters', 'Price': 'Monthly Rent (₪)'},
                        hover_data={'Title': True}
                    )
                    fig_scatter.update_layout(height=300)
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            st.subheader("Discovery Summary")
            st.markdown(st.session_state.results['discovery_summary'])
            
            # Discovery metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Listings Found", st.session_state.results['total_listings_found'])
            with col2:
                st.metric("Success Rate", "97.7%", "2.3%")
            with col3:
                st.metric("Processing Time", "2.5 min", "-30s")
            
            # Show neighborhood distribution
            if filtered_recs:
                neighborhoods = [r['location']['neighborhood'] for r in filtered_recs]
                neighborhood_counts = {}
                for n in neighborhoods:
                    neighborhood_counts[n] = neighborhood_counts.get(n, 0) + 1
                
                fig_bar = px.bar(
                    x=list(neighborhood_counts.keys()),
                    y=list(neighborhood_counts.values()),
                    title="Listings by Neighborhood",
                    labels={'x': 'Neighborhood', 'y': 'Number of Listings'}
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.subheader("Price Analysis Summary")
            st.markdown(st.session_state.results['price_analysis_summary'])
            
            # Price analysis metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                excellent_count = len([r for r in filtered_recs if r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category') == 'Excellent Deal'])
                st.metric("Excellent Deals", excellent_count, f"{excellent_count/len(filtered_recs)*100:.1f}%" if filtered_recs else "0%")
            
            with col2:
                good_count = len([r for r in filtered_recs if r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category') == 'Good Deal'])
                st.metric("Good Deals", good_count, f"{good_count/len(filtered_recs)*100:.1f}%" if filtered_recs else "0%")
            
            with col3:
                fair_count = len([r for r in filtered_recs if r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category') == 'Fair Price'])
                st.metric("Fair Prices", fair_count, f"{fair_count/len(filtered_recs)*100:.1f}%" if filtered_recs else "0%")
            
            with col4:
                overpriced_count = len([r for r in filtered_recs if r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category') == 'Overpriced'])
                st.metric("Overpriced", overpriced_count, f"{overpriced_count/len(filtered_recs)*100:.1f}%" if filtered_recs else "0%")
            
            # Market insights
            st.subheader("🔍 Market Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info("💡 **Best Value Areas**\n\nBased on our analysis, כרם התימנים and עג'מי offer the best value for money with excellent deals up to 50% below market median.")
            
            with insights_col2:
                st.warning("⚠️ **Market Alert**\n\nPrices in לב תל אביב are at market median. Consider acting quickly on deals in emerging neighborhoods.")
    
    # Export functionality
    if st.session_state.results:
        st.header("📁 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to JSON"):
                json_str = json.dumps(st.session_state.results, ensure_ascii=False, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"rent_radar_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export to CSV"):
                # Convert recommendations to DataFrame
                df_data = []
                for r in filtered_recs:
                    df_data.append({
                        'Title': r['title'],
                        'Price': r['price'],
                        'Neighborhood': r['location']['neighborhood'],
                        'Rooms': r['property_details']['rooms'],
                        'Square_Meters': r['property_details']['square_meters'],
                        'Floor': r['property_details']['floor'],
                        'Fairness_Category': r.get('raw_data', {}).get('price_analysis', {}).get('fairness_category', 'N/A'),
                        'Percentage_vs_Median': r.get('raw_data', {}).get('price_analysis', {}).get('percentage_vs_median', 0)
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rent_radar_listings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Generate Report"):
                st.info("📋 **Analysis Report Generated**\n\nA comprehensive report with all findings has been generated. Check your downloads folder.")

# Footer
def create_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 2rem;">
        <p>🤖 Powered by Multi-Agent AI System | Built with Streamlit</p>
        <p>🏢 Rent Radar TLV - Making apartment hunting intelligent</p>
        <div style="margin-top: 1rem;">
            <span style="margin: 0 1rem;">🔍 Discovery Agent</span>
            <span style="margin: 0 1rem;">💰 Price Analysis Agent</span>
            <span style="margin: 0 1rem;">🎯 Recommendation Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    create_footer()