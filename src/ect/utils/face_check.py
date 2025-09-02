import numpy as np
from typing import List, Tuple


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting algorithm for point-in-polygon test"""
    x, y = point
    n = polygon.shape[0]
    inside = False
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        if ((p1[1] > y) != (p2[1] > y)):
            xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if x <= xinters:
                inside = not inside
    return inside


def point_on_polygon_boundary(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a point lies on the boundary of a polygon"""
    x, y = point
    n = polygon.shape[0]
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        
        # Check if point is on the line segment p1-p2
        if point_on_line_segment(point, p1, p2, tol):
            return True
    return False


def point_on_line_segment(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a point lies on a line segment"""
    # Vector from p1 to p2
    v = p2 - p1
    # Vector from p1 to point
    u = point - p1
    
    # Handle 2D cross product properly for NumPy 2.0+
    if len(u) == 2 and len(v) == 2:
        # 2D cross product: u[0]*v[1] - u[1]*v[0]
        cross_product = u[0] * v[1] - u[1] * v[0]
    else:
        # For higher dimensions, use the magnitude of cross product
        cross_product = np.linalg.norm(np.cross(u, v))
    
    # Check if vectors are collinear (cross product is zero)
    if abs(cross_product) > tol:
        return False
    
    # Check if point is within the segment bounds
    if np.dot(v, v) < tol:  # p1 and p2 are the same point
        return np.linalg.norm(point - p1) < tol
    
    # Project point onto line and check if it's within [0, 1]
    t = np.dot(u, v) / np.dot(v, v)
    return 0 <= t <= 1


def segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if two line segments intersect (including endpoints)"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # Check if segments are degenerate (zero length)
    if np.allclose(p1, p2, atol=tol) or np.allclose(p3, p4, atol=tol):
        return False
    
    # Standard intersection test
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def validate_face_embedding(face_coords: np.ndarray, all_coords: np.ndarray, 
                          face_vertex_indices: List[int], all_vertex_indices: List[int],
                          tol: float = 1e-10) -> Tuple[bool, str]:
    """
    Validate that a face is properly embedded (no vertices in interior, no edge intersections)
    
    Args:
        face_coords: Coordinates of face vertices in order
        all_coords: Coordinates of all vertices in the complex
        face_vertex_indices: Indices of vertices that form the face
        all_vertex_indices: Indices of all vertices in the complex
        tol: Numerical tolerance
        
    Returns:
        (is_valid, error_message)
    """
    # Only validate for 2D faces (projecting higher-dimensional faces to 2D would be complex)
    if face_coords.shape[1] > 2:
        # For 3D+ faces, we'd need more sophisticated geometric analysis
        # For now, just check basic self-intersection in the first 2 dimensions
        face_coords_2d = face_coords[:, :2]
    else:
        face_coords_2d = face_coords
    
    # Check 1: No other vertices inside the face (2D projection)
    for i, vertex_idx in enumerate(all_vertex_indices):
        if vertex_idx in face_vertex_indices:
            continue
            
        if i >= len(all_coords):
            continue  # Skip if vertex index is out of bounds
            
        vertex_coord = all_coords[i]
        
        # Project to 2D if needed
        if vertex_coord.shape[0] > 2:
            vertex_coord_2d = vertex_coord[:2]
        else:
            vertex_coord_2d = vertex_coord
        
        # Check if vertex is strictly inside (not on boundary)
        if point_in_polygon(vertex_coord_2d, face_coords_2d):
            # Double-check it's not on the boundary
            if not point_on_polygon_boundary(vertex_coord_2d, face_coords_2d, tol):
                return False, f"Vertex {vertex_idx} is inside face interior"
    
    # Check 2: Face edges don't self-intersect (for non-convex faces)
    n = len(face_coords_2d)
    if n < 3:
        return False, "Face must have at least 3 vertices"
    
    for i in range(n):
        edge1_start = face_coords_2d[i]
        edge1_end = face_coords_2d[(i + 1) % n]
        
        # Check against non-adjacent edges
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:  # Skip last edge vs first edge
                continue
                
            edge2_start = face_coords_2d[j]
            edge2_end = face_coords_2d[(j + 1) % n]
            
            if segments_intersect(edge1_start, edge1_end, edge2_start, edge2_end, tol):
                return False, f"Face edges intersect: edge {i}-{(i+1)%n} intersects edge {j}-{(j+1)%n}"
    
    return True, ""


def validate_edge_embedding(edge_coords: np.ndarray, all_coords: np.ndarray,
                          edge_vertex_indices: List[int], all_vertex_indices: List[int],
                          tol: float = 1e-10) -> Tuple[bool, str]:
    """
    Validate that an edge is properly embedded (no vertices on interior)
    
    Args:
        edge_coords: Coordinates of edge endpoints
        all_coords: Coordinates of all vertices in the complex  
        edge_vertex_indices: Indices of vertices that form the edge
        all_vertex_indices: Indices of all vertices in the complex
        tol: Numerical tolerance
        
    Returns:
        (is_valid, error_message)
    """
    if len(edge_coords) != 2:
        return False, "Edge must have exactly 2 vertices"
    
    p1, p2 = edge_coords[0], edge_coords[1]
    
    # Check that no other vertices lie on the edge interior
    for i, vertex_idx in enumerate(all_vertex_indices):
        if vertex_idx in edge_vertex_indices:
            continue
            
        vertex_coord = all_coords[i]
        
        # Check if vertex is on the edge (excluding endpoints)
        if point_on_line_segment(vertex_coord, p1, p2, tol):
            # Make sure it's not just very close to an endpoint
            if (np.linalg.norm(vertex_coord - p1) > tol and 
                np.linalg.norm(vertex_coord - p2) > tol):
                return False, f"Vertex {vertex_idx} lies on edge interior"
    
    return True, ""
