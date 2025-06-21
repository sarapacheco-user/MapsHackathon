
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from helpers import (
    geocode_address,
    parse_address_components,
    reverse_geocode,
    calculate_routes,
    calculate_route_safety,
    parse_route_data,
    format_route,
    format_safety,
    get_crime_data,
    estimate_lighting_quality,
    decode_polyline,
    sample_route_coordinates,
    get_nearby_places
)

api_blueprint = Blueprint('api', __name__)

# Geocode API
@api_blueprint.route('/geocode', methods=['GET'])
def api_geocode():
    """
    Geocode an address to coordinates.
    
    Query Parameters:
        address (str): Human-readable address (e.g., "123 Main St, New York").
    
    Returns:
        JSON response with formatted address, coordinates, and place details.
    """
    address = request.args.get('address')

    if not address:
        return jsonify({
            "error": "Address parameter is required",
            "status": "error"
        }), 400
    
    try:
        result = geocode_address(address)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except Exception as e:
        current_app.logger.error(f"Geocode error: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred",
            "status": "error"
        }), 500

# Reverse Geocode API
@api_blueprint.route('/reverse-geocode', methods=['GET'])
def api_reverse_geocode():
    """
    Convert coordinates to a human-readable address.
    
    Query Parameters:
        lat (float): Latitude coordinate
        lng (float): Longitude coordinate
    
    Returns:
        JSON response with formatted address and location details.
    """
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    
    if lat is None or lng is None:
        return jsonify({
            "error": "Both lat and lng parameters are required",
            "status": "error"
        }), 400
    
    try:
        result = reverse_geocode(lat, lng)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except Exception as e:
        current_app.logger.error(f"Reverse geocode error: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred",
            "status": "error"
        }), 500
    # Calculate Route
@api_blueprint.route('/routes', methods=['POST'])
def calculate_route():
    data = request.get_json()
    try:
        routes = calculate_routes(
            data['origin'],
            data['destination'],
            data.get('travel_mode', 'DRIVING')
        )

        if not routes:
            return jsonify({"error": "No routes found", "status": "error"}), 404

        # Defensive filter for safety_score
        routes = [r for r in routes if 'safety_score' in r and isinstance(r['safety_score'], (int, float))]
        if not routes:
            return jsonify({"error": "No valid routes with safety scores found", "status": "error"}), 404

        safest_route = max(routes, key=lambda x: x['safety_score'])

        return jsonify({
            "routes": [format_route(route) for route in routes],
            "safest_route_id": safest_route['summary']
        }), 200

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": error_msg, "status": "error"}), 400

# Route Safety Analysis API
@api_blueprint.route('/route-safety', methods=['POST'])
def analyze_route_safety():
    """
    Analyze safety of a specific route
    Request Body: {
        "polyline": string,
        "travel_mode": "DRIVING/WALKING/etc",
        "hour_of_day": int (optional)
    }
    """
    data = request.get_json()
    try:
        score, factors = calculate_route_safety(
            data['polyline'],
            data['travel_mode'],
            data.get('hour_of_day', datetime.now().hour)
        )
        return jsonify({
            "safety_score": score,
            "factors": format_safety(factors)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Nearby Places API
@api_blueprint.route('/nearby-places', methods=['GET'])
def get_nearby_safety_points():
    """
    Get safety-relevant places near a location
    Query Params:
        lat (float): Latitude
        lng (float): Longitude
        radius (int, optional): Search radius in meters (default: 100)
    """
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        radius = int(request.args.get('radius', 100))
        places = get_nearby_places(lat, lng, radius)
        return jsonify({
            "places": places,
            "safety_relevant": [
                p for p in places 
                if any(t in p.get('types', []) 
                for t in ['police', 'hospital', 'bar', 'night_club'])
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Safest Route API (Enhanced)
@api_blueprint.route('/safest-route', methods=['POST'])
def get_safest_route():
    """
    Find the safest route between two points
    Request Body: {
        "origin": {"lat": float, "lng": float},
        "destination": {"lat": float, "lng": float},
        "travel_mode": "DRIVING/WALKING/BICYCLING/TRANSIT"
    }
    """
    data = request.get_json()
    
    try:
        routes = calculate_routes(
            data['origin'],
            data['destination'],
            data.get('travel_mode', 'DRIVING')
        )
        
        if not routes:
            return jsonify({"error": "No routes found"}), 404
            
        safest_route = max(routes, key=lambda x: x['safety_score'])
        
        return jsonify({
            "route": format_route(safest_route),
            "safety_analysis": {
                "score": safest_route['safety_score'],
                "primary_risks": [
                    k for k, v in safest_route['safety_factors'].items()
                    if isinstance(v, str) and v in ['Poor', 'High'] or
                    (isinstance(v, int) and v > 2)
                ]
            },
            "comparison": {
                "safer_than_worst": round(
                    safest_route['safety_score'] - 
                    min(r['safety_score'] for r in routes), 1
                ),
                "total_alternatives": len(routes) - 1
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
