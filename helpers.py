from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv
from flask import current_app
import requests
from flask import current_app
from datetime import datetime
import random
from typing import Tuple, Dict, List
from collections import defaultdict
# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# API Settings here 
load_dotenv()

app.config['GOOGLE_MAPS_API_KEY'] = os.getenv('GOOGLE_MAPS_API_KEY')
# 1) Geocoding = Convert human-readable address to coordinates using Google Geocoding API

"""    Convert human-readable address to coordinates using Google Geocoding API
       Returns: {'lat': float, 'lng': float, 'formatted_address': str} """

def geocode_address(address: str) -> dict:
    """
    Geocode an address using Google Maps Geocoding API
    Returns: {
        'formatted_address': str,
        'latitude': float,
        'longitude': float,
        'location_type': str,
        'place_id': str,
        'components': dict
    }
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        'address': address,
        'key': current_app.config['GOOGLE_MAPS_API_KEY']
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            raise ValueError(f"Geocoding error: {data.get('error_message', 'Unknown error')}")
        
        # Extract the first result
        result = data['results'][0]
        
        return {
            'formatted_address': result['formatted_address'],
            'latitude': result['geometry']['location']['lat'],
            'longitude': result['geometry']['location']['lng'],
            'location_type': result['geometry']['location_type'],
            'place_id': result['place_id'],
            'components': parse_address_components(result['address_components'])
        }
    except Exception as e:
        current_app.logger.error(f"Geocoding failed for address {address}: {str(e)}")
        raise

def parse_address_components(components: list) -> dict:
    """
    Parse address components into a more usable format
    """
    parsed = {}
    for component in components:
        for type in component['types']:
            parsed[type] = component['long_name']
    return parsed


# 2) Reverse Geocoding = Convert coordinates to human-readable address using Google Geocoding API

"""     Convert coordinates back to human-readable address
        Returns: formatted address string"""

def reverse_geocode(lat: float, lng: float) -> dict:
    """
    Reverse geocode coordinates to get address information
    Returns: Same structure as geocode_address()
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        'latlng': f"{lat},{lng}",
        'key': current_app.config['GOOGLE_MAPS_API_KEY']
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            raise ValueError(f"Reverse geocoding error: {data.get('error_message', 'Unknown error')}")
        
        # Extract the first result
        result = data['results'][0]
        
        return {
            'formatted_address': result['formatted_address'],
            'latitude': lat,
            'longitude': lng,
            'location_type': result['geometry']['location_type'],
            'place_id': result['place_id'],
            'components': parse_address_components(result['address_components'])
        }
    except Exception as e:
        current_app.logger.error(f"Reverse geocoding failed for coordinates {lat},{lng}: {str(e)}")
        raise


# 3) Calculate Route = Calculate route between two coordinates using Google Directions API

"""  Get multiple route options from Directions API with safety analysis
    Returns: List of route dictionaries with safety metadata
                """

def calculate_routes(origin: dict, destination: dict, travel_mode: str = "DRIVING") -> list:
    """
    Calculate multiple route options from origin to destination
    Returns: List of route dictionaries with safety metadata
    
    Args:
        origin: {'lat': float, 'lng': float}
        destination: {'lat': float, 'lng': float} or {'place_id': str}
        travel_mode: DRIVING, WALKING, BICYCLING, TRANSIT
    """
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        'origin': f"{origin['lat']},{origin['lng']}",
        'key': current_app.config['GOOGLE_MAPS_API_KEY'],
        'alternatives': 'true',
        'mode': travel_mode.lower(),
        'units': 'imperial' if current_app.config.get('DISTANCE_UNITS') == 'IMPERIAL' else 'metric'
    }
    
    # Handle both coordinate and place_id destinations
    if 'place_id' in destination:
        params['destination'] = f"place_id:{destination['place_id']}"
    else:
        params['destination'] = f"{destination['lat']},{destination['lng']}"
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            raise ValueError(f"Directions error: {data.get('error_message', 'Unknown error')}")
        
        routes = []
        for route in data['routes']:
            route_data = parse_route_data(route, origin, destination, travel_mode)
            routes.append(route_data)
        
        # Sort routes by duration then distance
        routes.sort(key=lambda x: (x['duration']['value'], x['distance']['value']))
        
        return routes
    except Exception as e:
        current_app.logger.error(f"Route calculation failed: {str(e)}")
        raise

def parse_route_data(route: dict, origin: dict, destination: dict, travel_mode: str) -> dict:
    """Parse raw route data into standardized format with safety analysis"""
    leg = route['legs'][0]  # Assuming single leg routes
    
    # Basic route info
    route_data = {
        'overview_polyline': route['overview_polyline']['points'],
        'bounds': route['bounds'],
        'summary': route['summary'],
        'warnings': route.get('warnings', []),
        'distance': leg['distance'],
        'duration': leg['duration'],
        'start_address': leg['start_address'],
        'end_address': leg['end_address'],
        'steps': [],
        'travel_mode': travel_mode,
        'safety_score': 0,  # Will be calculated
        'safety_factors': {}
    }
    
    # Parse steps
    for step in leg['steps']:
        route_data['steps'].append({
            'distance': step['distance']['text'],
            'duration': step['duration']['text'],
            'instructions': step['html_instructions'],
            'travel_mode': step['travel_mode'],
            'polyline': step['polyline']['points']
        })
    
    # Calculate safety score (using fictional data - replace with real implementation)
    route_data['safety_score'], route_data['safety_factors'] = calculate_route_safety(
        route_data['overview_polyline'], 
        travel_mode,
        datetime.now().hour
    )
    
    return route_data

def calculate_route_safety(polyline: str, travel_mode: str, hour_of_day: int) -> tuple:
    """
    Calculate safety score for a route (fictional implementation)
    Returns: (score: float, factors: dict)
    """
    # In a real app, this would analyze the route segments using safety data
    base_score = 7.0  # Base safety score
    
    # Adjust for time of day
    time_factor = 0.8 if 20 <= hour_of_day <= 6 else 1.0  # Less safe at night
    base_score *= time_factor
    
    # Adjust for travel mode
    mode_factors = {
        'DRIVING': 0.9,
        'WALKING': 1.1,
        'BICYCLING': 1.2,
        'TRANSIT': 1.0
    }
    base_score *= mode_factors.get(travel_mode, 1.0)
    
    # Add some random variation based on polyline (in real app would use actual route analysis)
    variation = (hash(polyline) % 100) / 50  # -1 to 1
    final_score = max(1, min(10, round(base_score + variation, 1)))
    
    # Generate fictional safety factors FICTIONAL !!!!
    factors = {
        'lighting': random.choices(['Good', 'Fair', 'Poor'], weights=[0.6, 0.3, 0.1])[0],
        'crime_reports': max(0, int(random.gauss(2, 1.5))),
        'pedestrian_infrastructure': random.choices(['Excellent', 'Good', 'Fair', 'Poor'], 
                                                 weights=[0.2, 0.4, 0.3, 0.1])[0],
        'traffic_volume': random.choices(['Low', 'Medium', 'High'], 
                                       weights=[0.3, 0.5, 0.2])[0]
    }
    
    return final_score, factors

# 4) Analyze Total Safety = Analyze safety of a route using Google Places API
"""
    Analyze safety along a route's path
    Returns: {'safety_score': float, 'danger_zones': list, 'factors': dict}
    """
import random
from typing import Tuple, Dict, List
from collections import defaultdict

def calculate_route_safety(polyline: str, travel_mode: str, hour_of_day: int) -> Tuple[float, Dict]:
    """
    Analyze safety of a route using Google Places API and other factors
    Returns: (safety_score: float, safety_factors: dict)
    """
    try:
        # Decode the polyline to get coordinates along the route
        coordinates = decode_polyline(polyline)
        
        # Sample points along the route (every 200 meters)
        sample_points = sample_route_coordinates(coordinates, interval_meters=200)
        
        # Initialize safety factors
        safety_factors = {
            'crime_incidents': 0,
            'lighting_quality': [],
            'safe_locations': 0,
            'danger_zones': [],
            'place_types': defaultdict(int)
        }
        
        # Analyze each sample point
        for i, (lat, lng) in enumerate(sample_points):
            # Get safety-related places near this point
            nearby_places = get_nearby_places(lat, lng, radius=100)  # 100 meter radius
            
            # Analyze place types for safety implications
            for place in nearby_places:
                place_type = place.get('types', [])
                safety_factors['place_types'][tuple(place_type)] += 1
                
                # Positive safety indicators
                if any(t in place_type for t in ['police', 'fire_station', 'hospital']):
                    safety_factors['safe_locations'] += 1
                
                # Negative safety indicators
                if any(t in place_type for t in ['bar', 'night_club', 'liquor_store']):
                    safety_factors['danger_zones'].append({
                        'location': {'lat': lat, 'lng': lng},
                        'type': 'nightlife',
                        'distance_along_route': i * 200  # meters
                    })
        
        # Calculate base safety score (0-10 scale)
        base_score = 7.0  # Neutral starting point
        
        # Adjust score based on findings
        safe_location_bonus = min(3, safety_factors['safe_locations'] * 0.2)
        danger_penalty = min(3, len(safety_factors['danger_zones']) * 0.3)
        
        base_score += safe_location_bonus
        base_score -= danger_penalty
        
        # Adjust for time of day
        if 20 <= hour_of_day <= 6:  # Night time
            base_score *= 0.8  # Reduce safety score at night
            
        # Adjust for travel mode vulnerability
        mode_factors = {
            'DRIVING': 1.0,
            'WALKING': 0.9,
            'BICYCLING': 0.85,
            'TRANSIT': 0.95
        }
        base_score *= mode_factors.get(travel_mode, 1.0)
        
        # Ensure score stays within bounds
        final_score = max(1, min(10, round(base_score, 1)))
        
        # Add additional metadata
        safety_factors['lighting_quality'] = estimate_lighting_quality(sample_points, hour_of_day)
        safety_factors['crime_incidents'] = get_crime_data(sample_points)  # Would use real crime API
        
        return final_score, safety_factors
        
    except Exception as e:
        current_app.logger.error(f"Safety analysis failed: {str(e)}")
        # Return default values if analysis fails
        return 5.0, {
            'error': 'Safety analysis incomplete',
            'danger_zones': [],
            'factors': {}
        }
# 5) Format Route = Format route data for frontend display

"""  Format route data for frontend display
    Returns: List of formatted route dictionaries
    """

from flask_caching import Cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)
@cache.cached(timeout=3600, query_string=True)
def format_route(route_data):
    """
    Format route data for frontend display
    """
    if not route_data:
        return []
    
    if isinstance(route_data, dict):
        route_data = [route_data]
    
    formatted_routes = []
    
    for route in route_data:
        distance = route.get('distance', {})
        duration = route.get('duration', {})

        # Handle if distance is a dict (Google API style) or a raw number
        distance_value = distance.get('value') / 1000 if isinstance(distance, dict) and 'value' in distance else distance
        if isinstance(distance_value, (int, float)):
            distance_str = f"{distance_value:.1f} km"
        else:
            distance_str = str(distance)

        # Same for duration
        duration_value = duration.get('value') / 60 if isinstance(duration, dict) and 'value' in duration else duration
        if isinstance(duration_value, (int, float)):
            duration_str = f"{duration_value:.1f} mins"
        else:
            duration_str = str(duration)

        formatted_route = {
            'id': route.get('id'),
            'name': route.get('name', 'Unnamed Route'),
            'start_point': route.get('start_point'),
            'end_point': route.get('end_point'),
            'distance': distance_str,
            'duration': duration_str,
            'waypoints': route.get('waypoints', []),
            'difficulty': route.get('difficulty', 'medium'),
            'rating': route.get('rating', 0),
            'thumbnail': route.get('thumbnail_url', 'default_route.jpg')
        }
        
        if 'description' in route:
            formatted_route['description'] = route['description']
        if 'elevation_gain' in route:
            formatted_route['elevation_gain'] = f"{route['elevation_gain']} m"
            
        formatted_routes.append(formatted_route)
    
    return formatted_routes

# 6) Format Safety = Format safety data for frontend display
"""  Format safety data for frontend display
    Returns: Formatted safety data dictionary
    """
def format_safety(safety_data):
    """
    Format safety data for frontend display
    
    Args:
        safety_data: Raw safety data (dictionary containing various safety metrics)
        
    Returns:
        Formatted safety data dictionary with only necessary fields for frontend
    """
    if not safety_data:
        return {
            'error': 'No safety data available',
            'status': 'unknown'
        }
    
    # Create the base formatted structure
    formatted = {
        'status': safety_data.get('overall_status', 'unknown').lower(),
        'last_updated': safety_data.get('timestamp', ''),
        'alerts': []
    }
    
    # Add overall safety score if available
    if 'safety_score' in safety_data:
        formatted['score'] = f"{safety_data['safety_score']}/100"
    
    # Process alerts if they exist
    if 'alerts' in safety_data and isinstance(safety_data['alerts'], list):
        for alert in safety_data['alerts']:
            formatted_alert = {
                'type': alert.get('type', 'notice'),
                'title': alert.get('title', 'Safety Notice'),
                'message': alert.get('message', ''),
                'severity': alert.get('severity_level', 1),
                'time': alert.get('timestamp', '')
            }
            # Only include relevant fields
            if 'location' in alert:
                formatted_alert['location'] = {
                    'lat': alert['location'].get('lat'),
                    'lng': alert['location'].get('lng')
                }
            formatted['alerts'].append(formatted_alert)
    
    # Add additional metrics if they exist
    optional_fields = {
        'crime_stats': 'crime_statistics',
        'police_presence': 'police_presence_level',
        'lighting': 'lighting_quality'
    }
    
    for frontend_key, backend_key in optional_fields.items():
        if backend_key in safety_data:
            formatted[frontend_key] = safety_data[backend_key]
    
    return formatted
# 7) Get Crime data for coordinates
"""
    Get crime data for coordinates (placeholder for real crime API)
    """
def get_crime_data(coordinates: List[Tuple[float, float]]) -> int:
    """
    Get crime data for coordinates (placeholder for real crime API)
    """ 
    # In a real implementation, this would query a crime data API
    return random.randint(0, 5)  # SIMULATED DATA !!!!!!!!!

# 8) Estimate Lighting Quality
def estimate_lighting_quality(sample_points: List[Tuple[float, float]], hour_of_day: int) -> str:
    """
    Estimate lighting quality based on time of day and sample points
    Returns: 'Good', 'Fair', 'Poor'
    """
    if hour_of_day < 6 or hour_of_day >= 20:  # Night time
        return random.choices(['Good', 'Fair', 'Poor'], weights=[0.2, 0.5, 0.3])[0] # Simulated data !!!!!!!
    else:
        return 'Good'  # Daytime is generally well-lit
if __name__ == '__main__':
    app.run(port=5000)

# 9) Decode Polyline
def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """
    Decode Google Maps polyline string into coordinates
    """
    coordinates = []
    index = lat = lng = 0
    
    while index < len(polyline_str):
        shift = result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat
        
        shift = result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng
        
        coordinates.append((lat * 1e-5, lng * 1e-5))
    
    return coordinates
# 10) Sample Route Coordinates
def sample_route_coordinates(coordinates: List[Tuple[float, float]], interval_meters: int = 200) -> List[Tuple[float, float]]:
    """
    Sample points along the route at regular intervals
    """
    # For simplicity, we'll just take every nth point
    # In a real implementation, you'd calculate actual distance between points
    sample_rate = max(1, len(coordinates) // 20)
    return coordinates[::sample_rate]

# 11) Get Nearby Places
def get_nearby_places(lat: float, lng: float, radius: int) -> List[Dict]:
    """
    Get nearby places using Google Places API
    """
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    params = {
        'location': f"{lat},{lng}",
        'radius': radius,
        'key': current_app.config['GOOGLE_MAPS_API_KEY']
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            current_app.logger.warning(f"Places API error: {data.get('error_message', 'Unknown')}")
            return []
        
        return data['results']
    except Exception as e:
        current_app.logger.error(f"Failed to fetch nearby places: {str(e)}")
        return []